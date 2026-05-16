"""
Billboard Hot 100 – Calibración del modelo de Soh & Bradlow-Fader
==================================================================

Modelo
------
Para cada canción i en el chart de la semana t, definimos:

    tau_i = t - t_deb_i          (semanas desde el debut de la canción i)

    S(tau) = A / (1 + (1/p0 - 1) * exp(-tau / t0))
             Curva logística que modela el "buzz" intrínseco de la canción.

    tau_c_i = mu + beta1 * x1_i + beta2 * x2_i
             Constante de decaimiento de memoria, personalizada por canción.

    V_i(t) = sum_{tau=0}^{tau_i} S(tau) * (tau+1)^{-1} * exp(-tau / tau_c_i)
             Utilidad acumulada de la canción i en la semana t.

La probabilidad del ranking observado (posiciones 1..49 del top-50) es:

    P = prod_{k=1}^{49}  exp(V_{pi(k)}(t)) / sum_{j=k}^{49} exp(V_{pi(j)}(t))

donde pi(k) es la canción en la posición k (de mejor a peor ranking).
Maximizamos la log-verosimilitud sumada sobre todos los charts del CSV.

Parámetros: A, p0, t0, mu, beta1, beta2

CSV de entrada
--------------
Columnas (separadas por ,):
    fecha_chart , fecha_debut , titulo , artista , x2 , x1 , ranking

    - fecha_chart, fecha_debut : formato ISO (YYYY-MM-DD)
    - x2 : 0 o 1 (colaboración)
    - x1 : entero (hits previos del artista)
    - ranking : entero 1..N (posición en ese chart)

Uso
---
    python billboard_model.py --csv datos.csv [opciones]

    --csv      Ruta al CSV (requerido)
    --sep      Separador del CSV (default: |)
    --n_weeks  Número de semanas de billboard a usar para calibración (default: todas)
    --out      Prefijo de archivos de salida (default: "output")
    --n_starts Número de puntos de inicio para la optimización multi-start (default: 20)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CARGA Y PREPROCESAMIENTO
# ─────────────────────────────────────────────

def load_data(csv_path: str, sep: str = ",", n_weeks: int = None) -> list[dict]:
    """
    Lee el CSV y devuelve una lista de charts.

    Cada elemento es un dict:
        {
            "fecha":  pd.Timestamp,
            "songs":  lista de dicts ordenada por ranking (1 = mejor),
                      cada dict: {tau, x1, x2, titulo}
        }

    tau = número de semanas que lleva la canción en la lista cuando
          se publica ese chart (0 = semana de debut).
    """
    df = pd.read_csv(
    csv_path,
    sep=sep,
    header=0,  # la primera fila es el encabezado
    names=["fecha_chart", "fecha_debut", "titulo", "artista", "x2", "x1", "ranking"],
    )
    df["fecha_chart"] = pd.to_datetime(df["fecha_chart"], format="ISO8601")
    df["fecha_debut"] = pd.to_datetime(df["fecha_debut"], format="ISO8601")

    # Billboard usa semanas; calculamos tau en semanas enteras
    df["tau"] = ((df["fecha_chart"] - df["fecha_debut"]).dt.days / 7).round().astype(int)
    df["tau"] = df["tau"].clip(lower=0)  # seguridad: nunca negativo

    df["tau"].to_csv('taus.csv', sep=',')
    # Ordenamos y agrupamos por semana de chart
    df = df.sort_values(["fecha_chart", "ranking"])

    charts_raw = []
    for fecha, grupo in df.groupby("fecha_chart"):
        # Nos quedamos solo con las top-50 (ajuste de Soh)
        grupo = grupo[grupo["ranking"] <= 50].copy()
        grupo = grupo.sort_values("ranking")

        songs = [
            {
                "titulo": row["titulo"],
                "tau":    int(row["tau"]),
                "x1":    float(row["x1"]),
                "x2":    float(row["x2"]),
            }
            for _, row in grupo.iterrows()
        ]
        if len(songs) >= 2:          # necesitamos al menos 2 canciones
            charts_raw.append({"fecha": fecha, "songs": songs})

    charts_raw.sort(key=lambda c: c["fecha"])

    if n_weeks is not None:
        charts_raw = charts_raw[:n_weeks]

    print(f"  Semanas cargadas: {len(charts_raw)}")
    print(f"  Rango temporal:   {charts_raw[0]['fecha'].date()} → {charts_raw[-1]['fecha'].date()}")
    charts_raw = charts_raw[8:]  # descartar las primeras 8 semanas (~2 meses)
    return charts_raw


# ─────────────────────────────────────────────
# 2. FUNCIONES DEL MODELO
# ─────────────────────────────────────────────

def S(tau_arr: np.ndarray, A: float, p0: float, t0: float) -> np.ndarray:
    """
    Curva logística de buzz: S(tau) = A / (1 + (1/p0 - 1)*exp(-tau/t0))
    tau_arr: array de enteros 0,1,...,tau_max
    """
    return A / (1.0 + (1.0 / p0 - 1.0) * np.exp(-tau_arr / t0))

def dS(tau_arr: np.ndarray, A: float, p0: float, t0: float) -> np.ndarray:
    """
    La derivada de S
    """
    return (-A / (1.0 + (1.0 / p0 - 1.0) * np.exp(-tau_arr / t0))**2)*(-1/p0 + 1)/t0* np.exp(-tau_arr / t0)


def compute_V(tau_max: int, x1: float, x2: float,
              A: float, p0: float, t0: float,
              mu: float, beta1: float, beta2: float) -> float:
    """
    V_i(t) = sum_{tau=0}^{tau_max}  S(tau) * (tau+1)^{-1} * exp(-tau / tau_c)

    tau_c = mu + beta1*x1 + beta2*x2  (debe ser > 0 para que tenga sentido físico)
    """
    tau_c = mu + beta1 * x1 + beta2 * x2
    if tau_c <= 0:
        return -np.inf                  # configuración inválida

    taus = np.arange(0, tau_max + 1, dtype=float)
    s_vals = S(tau_max - taus, A, p0, t0)   # S(t-τ): decrece con τ
    decay  = np.exp(-taus / tau_c)
    weight = 1.0 / (taus + 1.0)
    return float(np.sum(s_vals * weight * decay))


def log_likelihood_chart(songs: list[dict],
                         A: float, p0: float, t0: float,
                         mu: float, beta1: float, beta2: float) -> float:
    """
    Log-verosimilitud de un chart individual.

    Fórmula (Plackett-Luce sobre top-50, ajuste de Soh):
        log P = sum_{k=0}^{N-2}  [ V_{pi(k)} - log( sum_{j=k}^{N-1} exp(V_{pi(j)}) ) ]

    donde pi(k) es la canción en posición k (0-indexed, mejor primero).
    """
    N = len(songs)
    # Calcula V para cada canción
    Vs = np.array([
        compute_V(s["tau"], s["x1"], s["x2"], A, p0, t0, mu, beta1, beta2)
        for s in songs
    ])

    if np.any(~np.isfinite(Vs)):
        return -np.inf

    # Acumula la log-verosimilitud con el truco log-sum-exp para estabilidad numérica
    ll = 0.0
    for k in range(N - 1):           # k = 0 .. N-2
        remaining = Vs[k:]            # V del ganador + todos los que compiten con él
        max_v = remaining.max()
        log_denom = max_v + np.log(np.sum(np.exp(remaining - max_v)))
        ll += Vs[k] - log_denom

    return ll

# def log_likelihood_chart(songs, A, p0, t0, mu, beta1, beta2):
    
#     # Masa extra por canciones no observadas (Bradlow-Fader: n1=n2=50)
#     # Su valor esperado se aproxima con la media de los V observados

#     N = len(songs)
#     # Calcula V para cada canción
#     Vs = np.array([
#         compute_V(s["tau"], s["x1"], s["x2"], A, p0, t0, mu, beta1, beta2)
#         for s in songs
#     ])

#     if np.any(~np.isfinite(Vs)):
#         return -np.inf
#     n_unobserved = 50  # n1 + n2 de B&F
#     mean_V = Vs.mean()
#     extra_mass = n_unobserved * np.exp(mean_V)
    
#     ll = 0.0
#     for k in range(N - 1):
#         remaining = Vs[k:]
#         max_v = remaining.max()
#         log_denom = max_v + np.log(
#             np.sum(np.exp(remaining - max_v)) + extra_mass * np.exp(-max_v)
#         )
#         ll += Vs[k] - log_denom
#     return ll


def total_log_likelihood(params: np.ndarray, charts: list[dict]) -> float:
    """Suma de log-verosimilitudes sobre todos los charts."""
    A, p0, t0, mu, beta1, beta2 = params

    # Restricciones de dominio (devolvemos -inf si se violan)
    if A <= 0 or p0 <= 0 or p0 >= 1 or t0 <= 0:
        return -np.inf

    ll = 0.0
    for chart in charts:
        ll += log_likelihood_chart(chart["songs"], A, p0, t0, mu, beta1, beta2)
        if not np.isfinite(ll):
            return -np.inf

    return ll


def neg_ll(params: np.ndarray, charts: list[dict]) -> float:
    """Negativo de la log-verosimilitud (para minimizar)."""
    return -total_log_likelihood(params, charts)


# ─────────────────────────────────────────────
# 3. OPTIMIZACIÓN
# ─────────────────────────────────────────────

# Límites de búsqueda para cada parámetro:
#   (A, p0, t0, mu, beta1, beta2)
BOUNDS = [
    (0.1, 50.0),     # A       – amplitud del buzz
    (1e-4, 1),   # p0      – probabilidad inicial logística
    (0.5,  100.0),     # t0      – semivida logística (semanas)
    (0.1,  50.0),     # mu      – base de tau_c
    (-10.0,  10.0),     # beta1   – efecto de hits previos
    (-10.0,  10.0),     # beta2   – efecto de colaboración
]

PARAM_NAMES = ["A", "p0", "t0", "mu", "beta1", "beta2"]


def optimize(charts: list[dict], n_starts: int = 20, seed: int = 42) -> dict:
    """
    Estrategia de optimización en dos etapas:

    1. Evolución diferencial (búsqueda global) – robusta frente a mínimos locales.
       Explora el espacio de parámetros de forma estocástica.

    2. L-BFGS-B (refinamiento local) – parte desde el mejor punto global
       y afina con gradiente numérico.

    Devuelve un dict con los parámetros óptimos, el valor de LL y el histórico
    de convergencia de la etapa global.
    """
    print("\n[1/2] Evolución diferencial (búsqueda global)…")

    history = {"iteration": [], "neg_ll": []}

    def callback(xk, convergence):
        val = neg_ll(xk, charts)
        history["iteration"].append(len(history["iteration"]))
        history["neg_ll"].append(-val)   # guardamos LL (no neg_LL)
        print(f"  iter {history['iteration'][-1]:4d} | LL = {-val:.4f}")

    result_global = differential_evolution(
        neg_ll,
        bounds=BOUNDS,
        args=(charts,),
        strategy="best1bin",
        maxiter=500,
        popsize=15,
        tol=1e-8,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=seed,
        callback=callback,
        polish=False,        # el pulido fino lo hacemos con L-BFGS-B
        workers=1,
    )

    x0 = result_global.x
    print(f"\n  Mejor LL global:  {-result_global.fun:.6f}")
    print(f"  Parámetros:       {dict(zip(PARAM_NAMES, x0))}")

    print("\n[2/2] Refinamiento local (L-BFGS-B)…")
    result_local = minimize(
        neg_ll,
        x0=x0,
        args=(charts,),
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )

    best_params = result_local.x
    best_ll     = -result_local.fun
    print(f"  LL final:         {best_ll:.6f}")
    print(f"  Convergencia:     {'OK' if result_local.success else result_local.message}")

    return {
        "params":  dict(zip(PARAM_NAMES, best_params)),
        "ll":      best_ll,
        "history": history,
    }


# ─────────────────────────────────────────────
# 4. PREDICCIÓN VS RANKING REAL
# ─────────────────────────────────────────────

def predict_rankings(charts: list[dict], params: dict) -> pd.DataFrame:
    """
    Para cada (semana, canción) devuelve:
        - ranking_real   : posición observada (1 = mejor)
        - ranking_pred   : posición predicha por el modelo (basada en V_i)
        - V              : utilidad acumulada
        - prob_rank1     : P(canción es #1) = softmax(V)[0]
    """
    A, p0, t0, mu, beta1, beta2 = (
        params["A"], params["p0"], params["t0"],
        params["mu"], params["beta1"], params["beta2"],
    )

    rows = []
    for chart in charts:
        songs = chart["songs"]
        Vs = np.array([
            compute_V(s["tau"], s["x1"], s["x2"], A, p0, t0, mu, beta1, beta2)
            for s in songs
        ])
        # ranking predicho: mayor V → mejor posición
        order = np.argsort(-Vs)          # índices de mayor a menor V
        pred_ranks = np.empty_like(order)
        pred_ranks[order] = np.arange(1, len(order) + 1)

        softmax_V = np.exp(Vs - Vs.max())
        softmax_V /= softmax_V.sum()

        for idx, s in enumerate(songs):
            rows.append({
                "fecha":        chart["fecha"].date(),
                "titulo":       s["titulo"],
                "ranking_real": idx + 1,          # ya estaban ordenados por ranking
                "ranking_pred": int(pred_ranks[idx]),
                "V":            round(Vs[idx], 6),
                "prob_rank1":   round(softmax_V[idx], 6),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 5. VISUALIZACIONES
# ─────────────────────────────────────────────

def plot_convergence(history: dict, out_prefix: str):
    """Curva de convergencia de la evolución diferencial."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["iteration"], history["neg_ll"], color="#2563eb", lw=1.8)
    ax.set_xlabel("Iteración (evolución diferencial)")
    ax.set_ylabel("Log-verosimilitud")
    ax.set_title("Convergencia de la optimización global")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{out_prefix}_convergencia.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada: {path}")


def plot_ll_per_week(charts: list[dict], params: dict, out_prefix: str):
    """Log-verosimilitud por semana de chart."""
    A, p0, t0, mu, beta1, beta2 = (
        params["A"], params["p0"], params["t0"],
        params["mu"], params["beta1"], params["beta2"],
    )

    fechas, lls = [], []
    for chart in charts:
        ll = log_likelihood_chart(chart["songs"], A, p0, t0, mu, beta1, beta2)
        fechas.append(chart["fecha"])
        lls.append(ll)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(fechas, lls, width=5, color="#2563eb", alpha=0.7)
    ax.axhline(np.mean(lls), color="#dc2626", lw=1.5, linestyle="--", label=f"Media = {np.mean(lls):.2f}")
    ax.set_xlabel("Semana del chart")
    ax.set_ylabel("Log-verosimilitud")
    ax.set_title("Log-verosimilitud por semana de Billboard")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = f"{out_prefix}_ll_por_semana.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada: {path}")


def plot_pred_vs_real(df_pred: pd.DataFrame, out_prefix: str):
    """
    Tres paneles:
        (a) Scatter ranking predicho vs real
        (b) Distribución del error absoluto de ranking
        (c) Error medio absoluto por posición real
    """
    df = df_pred.copy()
    df["error"] = (df["ranking_pred"] - df["ranking_real"]).abs()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Scatter
    ax = axes[0]
    ax.scatter(df["ranking_real"], df["ranking_pred"],
               alpha=0.25, s=8, color="#2563eb")
    lim = max(df["ranking_real"].max(), df["ranking_pred"].max()) + 1
    ax.plot([1, lim], [1, lim], "r--", lw=1, label="Perfecto")
    ax.set_xlabel("Ranking real")
    ax.set_ylabel("Ranking predicho")
    ax.set_title("Predicho vs Real")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Histograma de error absoluto
    ax = axes[1]
    ax.hist(df["error"], bins=30, color="#2563eb", alpha=0.8, edgecolor="white")
    ax.axvline(df["error"].mean(), color="#dc2626", lw=1.5, linestyle="--",
               label=f"MAE = {df['error'].mean():.2f}")
    ax.set_xlabel("Error absoluto de ranking")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución del error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) MAE por posición real
    ax = axes[2]
    mae_by_pos = df.groupby("ranking_real")["error"].mean()
    ax.bar(mae_by_pos.index, mae_by_pos.values, color="#2563eb", alpha=0.8)
    ax.set_xlabel("Posición real")
    ax.set_ylabel("MAE")
    ax.set_title("Error medio por posición")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Predicciones del modelo vs Rankings reales", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = f"{out_prefix}_predicciones.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada: {path}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Calibración modelo Billboard Hot 100")
    parser.add_argument("--csv",      default="billboard_analisis6.csv",          help="Ruta al CSV de datos")
    parser.add_argument("--sep",      default=",",            help="Separador del CSV (default: ,)")
    parser.add_argument("--n_weeks",  type=int, default=None, help="Número de semanas a usar (default: todas)")
    parser.add_argument("--out",      default="output",       help="Prefijo de archivos de salida")
    parser.add_argument("--n_starts", type=int, default=20,   help="Multi-starts para evolución diferencial")
    args = parser.parse_args()

    print("=" * 60)
    print("  MODELO BILLBOARD HOT 100  –  Calibración")
    print("=" * 60)

    # 1. Cargar datos
    print("\n→ Cargando datos…")
    charts = load_data(args.csv, sep=args.sep, n_weeks=args.n_weeks)

    # 1.5. Buscar errores
    taus = [s["tau"] for chart in charts for s in chart["songs"]]
    print(f"tau: min={min(taus)}, max={max(taus)}, media={np.mean(taus):.1f}, mediana={np.median(taus):.1f}")

    x1s = [s["x1"] for chart in charts for s in chart["songs"]]
    x2s = [s["x2"] for chart in charts for s in chart["songs"]]
    print(f"x1: min={min(x1s)}, max={max(x1s)}, media={np.mean(x1s):.1f}")
    print(f"x2: proporción colaboraciones={np.mean(x2s):.2f}")
    # Distribución de taus
    import matplotlib.pyplot as plt
    plt.hist(taus, bins=50)
    plt.xlabel("tau (semanas en lista)")
    plt.title("Distribución de tau")
    plt.savefig("diagnostico_tau.png")


    # Inspecciona V para una canción concreta con parámetros razonables
    cancion_test = charts[10]["songs"][0]  # primera canción del chart 11
    print(cancion_test)

    A, p0, t0, mu = 1.0, 0.1, 10.0, 5.0

    taus = np.arange(0, cancion_test["tau"] + 1, dtype=float)

    # Versión ACTUAL (corregida): S(t - tau)
    s_nuevo = A / (1 + (1/p0 - 1) * np.exp(-(cancion_test["tau"] - taus) / t0))

    # Versión ANTERIOR (bug): S(tau)
    s_viejo = A / (1 + (1/p0 - 1) * np.exp(-taus / t0))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(taus, s_nuevo, label="S(t-τ) — correcto")
    axes[0].plot(taus, s_viejo, label="S(τ) — bug", linestyle="--")
    axes[0].set_title("Comparación S evaluado en τ vs t-τ")
    axes[0].legend()

    decay = np.exp(-taus / mu)
    weight = 1.0 / (taus + 1.0)
    V_nuevo = np.cumsum(s_nuevo * weight * decay)
    V_viejo = np.cumsum(s_viejo * weight * decay)
    axes[1].plot(taus, V_nuevo, label="V con S(t-τ)")
    axes[1].plot(taus, V_viejo, label="V con S(τ)", linestyle="--")
    axes[1].set_title("V acumulado resultante")
    axes[1].legend()
    plt.savefig("diagnostico_V.png")
    # 2. Optimizar
    print("\n→ Optimizando parámetros…")
    result = optimize(charts, n_starts=args.n_starts)

    # 3. Mostrar parámetros óptimos
    print("\n" + "=" * 60)
    print("  PARÁMETROS ÓPTIMOS")
    print("=" * 60)
    for name, val in result["params"].items():
        print(f"  {name:>7s} = {val:.6f}")
    print(f"\n  Log-verosimilitud total = {result['ll']:.4f}")
    print(f"  LL por semana (media)   = {result['ll'] / len(charts):.4f}")

    # 4. Predicciones
    print("\n→ Generando predicciones…")
    df_pred = predict_rankings(charts, result["params"])

    mae = (df_pred["ranking_pred"] - df_pred["ranking_real"]).abs().mean()
    pct_top3 = (
        (df_pred["ranking_real"] <= 3) & (df_pred["ranking_pred"] <= 3)
    ).mean() * 100
    print(f"  MAE global:              {mae:.2f} posiciones")
    print(f"  % top-3 correctamente:   {pct_top3:.1f}%")

    # 5. Guardar predicciones a CSV
    pred_path = f"{args.out}_predicciones.csv"
    df_pred.to_csv(pred_path, index=False)
    print(f"  Predicciones guardadas:  {pred_path}")

    # 6. Gráficas
    print("\n→ Generando gráficas…")
    plot_convergence(result["history"], args.out)
    plot_ll_per_week(charts, result["params"], args.out)
    plot_pred_vs_real(df_pred, args.out)

    print("\n✓ Proceso completado.")


if __name__ == "__main__":
    main()

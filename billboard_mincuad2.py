"""
Billboard Hot 100 - Estimación por mínimos cuadrados + visualización
=====================================================================
Ajusta V(t) ~ 1/ranking directamente por mínimos cuadrados ponderados.
Incluye función de visualización interactiva de curvas reales vs estimadas.

Uso:
    pip install pandas numpy scipy pyarrow matplotlib
    python billboard_mincuad.py --csv datos.csv --output resultados_mc.parquet
    python billboard_mincuad.py --csv datos.csv --output resultados_mc.parquet \\
                                --plot "Shape of You" "Ed Sheeran"
"""

import argparse
import warnings
import multiprocessing

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")          # sin GUI al ajustar; se activa solo al visualizar
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

FECHA_INICIO = pd.Timestamp("2013-06-01")
MIN_SEMANAS  = 4


# ═════════════════════════════════════════════════════════════════
# 1. MODELO
# ═════════════════════════════════════════════════════════════════

def compute_V_vec(T, A, p0, t0, tau_c):
    """V(k) = Σ S'(τ)·Θ(k−τ) vectorizado con np.convolve."""
    n      = T + 1
    taus   = np.arange(n, dtype=np.float64)
    c      = 1.0 / p0 - 1.0
    e_term = np.exp(-taus / t0)
    sp     = A * c * e_term / (t0 * (1.0 + c * e_term) ** 2)
    theta  = np.exp(-taus / tau_c) / (1.0 + taus)
    return np.convolve(sp, theta)[:n]


# ═════════════════════════════════════════════════════════════════
# 2. FUNCIÓN OBJETIVO: mínimos cuadrados ponderados sobre 1/ranking
# ═════════════════════════════════════════════════════════════════

def mse_weighted(params, t_rels, y_obs, weights):
    """
    params  : [log_A, logit_p0, log_t0, log_tau_c]
    t_rels  : semanas relativas al debut
    y_obs   : 1/ranking observado (Zipf)
    weights : pesos por semana (aquí usamos 1/ranking para dar más peso
              a las semanas en posiciones altas, que son más informativas)

    Minimiza Σ w_k · (V(t_k) − y_obs_k)²
    """
    log_A, logit_p0, log_t0, log_tau_c = params

    A     = np.exp(log_A)
    p0    = 1.0 / (1.0 + np.exp(-logit_p0))
    t0    = np.exp(log_t0)
    tau_c = np.exp(log_tau_c)

    try:
        V_all = compute_V_vec(int(t_rels.max()), A, p0, t0, tau_c)
    except Exception:
        return 1e12

    residuals = V_all[t_rels] - y_obs
    return float(np.sum(weights * residuals ** 2))


# ═════════════════════════════════════════════════════════════════
# 3. AJUSTE POR CANCIÓN
# ═════════════════════════════════════════════════════════════════

def fit_song_mc(args_tuple):
    """
    args_tuple: (song_key, t_rels, y_obs, weights, x1, x2, x3, n_semanas, ranking_debut)
    """
    song_key, t_rels, y_obs, weights, x1, x2, x3, n_semanas, ranking_debut = args_tuple
    titulo, artista = song_key

    t_rels  = np.asarray(t_rels,   dtype=np.int32)
    y_obs   = np.asarray(y_obs,    dtype=np.float64)
    weights = np.asarray(weights,  dtype=np.float64)

    # Inicialización inteligente de p0 a partir del ranking de debut
    p0_init   = min(0.95, max(0.05, 1.0 / ranking_debut))
    logit_p0i = np.log(p0_init / (1.0 - p0_init))

    # A inicial: valor de V debería estar en el rango de 1/ranking,
    # que es pequeño. Empezamos con A ~ max(y_obs).
    A_init   = float(np.max(y_obs))
    log_A_i  = np.log(max(A_init, 1e-6))

    starts = [
        [log_A_i,      logit_p0i,              np.log(5.0),  np.log(5.0)],
        [log_A_i,      logit_p0i,              np.log(3.0),  np.log(2.0)],
        [log_A_i,      np.log(0.05 / 0.95),    np.log(8.0),  np.log(8.0)],
        [log_A_i,      np.log(0.5 / 0.5),      np.log(3.0),  np.log(3.0)],
        [log_A_i,      logit_p0i,              np.log(5.0),  np.log(20.0)],
    ]

    bounds = [
        (-10, 10),   # log_A:     sin restricción fuerte (y_obs puede ser pequeño)
        (-6,   6),   # logit_p0:  p0 en (0.002, 0.998)
        (-2,   5),   # log_t0:    t0 en (0.13, 148) semanas
        (-2,   5),   # log_tau_c: tau_c en (0.13, 148) semanas
    ]

    best_mse = np.inf
    best_x   = None

    for x0 in starts:
        try:
            res = minimize(
                mse_weighted,
                x0,
                args=(t_rels, y_obs, weights),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.fun < best_mse:
                best_mse = res.fun
                best_x   = res.x
        except Exception:
            continue

    if best_x is None:
        return None

    log_A, logit_p0, log_t0, log_tau_c = best_x
    return {
        "titulo":        titulo,
        "artista":       artista,
        "x1": x1, "x2": x2, "x3": x3,
        "A":             float(np.exp(log_A)),
        "p0":            float(1.0 / (1.0 + np.exp(-logit_p0))),
        "t0":            float(np.exp(log_t0)),
        "tau_c":         float(np.exp(log_tau_c)),
        "mse":           float(best_mse),
        "n_semanas":     n_semanas,
        "ranking_debut": ranking_debut,
    }


# ═════════════════════════════════════════════════════════════════
# 4. CONSTRUIR ARGUMENTOS POR CANCIÓN
# ═════════════════════════════════════════════════════════════════

def build_song_args_mc(df):
    """
    Para mínimos cuadrados no necesitamos denominadores:
    cada canción se ajusta independientemente sobre su propia curva 1/ranking.
    """
    song_args = []

    for (titulo, artista), grp in df.groupby(["titulo", "artista"]):
        grp = grp.sort_values("fecha_chart")
        n   = len(grp)
        if n < MIN_SEMANAS:
            continue

        debut_ts      = pd.Timestamp(grp["fecha_debut"].iloc[0])
        fechas        = grp["fecha_chart"].values
        rankings      = grp["ranking"].values.astype(int)
        ranking_debut = int(rankings[0])
        x1 = float(grp["x1"].iloc[0])
        x2 = float(grp["x2"].iloc[0])
        x3 = float(grp["x3"].iloc[0])

        t_rels = np.array(
            [(pd.Timestamp(f) - debut_ts).days // 7 for f in fechas],
            dtype=np.int32,
        )

        # Objetivo: V(t) ≈ 1/ranking  (Zipf)
        y_obs   = 1.0 / rankings.astype(float)

        # Pesos: dar más importancia a semanas en posiciones altas
        # (posición 1 pesa 50x más que posición 50)
        weights = y_obs.copy()   # w_k = 1/r_k

        song_args.append(
            ((titulo, artista), t_rels, y_obs, weights, x1, x2, x3, n, ranking_debut)
        )

    return song_args


# ═════════════════════════════════════════════════════════════════
# 5. ETAPA 2: REGRESIÓN DE tau_c Y ESTIMACIÓN MVLN
# ═════════════════════════════════════════════════════════════════

def stage2_regression(params_df):
    X = np.column_stack([
        np.ones(len(params_df)),
        params_df["x1"].values,
        params_df["x2"].values,
        params_df["x3"].values,
    ])
    y          = params_df["tau_c"].values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu, b1, b2, b3 = coeffs
    residuals  = y - X @ coeffs
    r2         = 1 - np.var(residuals) / np.var(y)
    return mu, b1, b2, b3, r2


def estimate_mvln(params_df):
    X = pd.DataFrame({
        "log_A":     np.log(params_df["A"]),
        "logit_p0":  np.log(params_df["p0"] / (1.0 - params_df["p0"])),
        "log_t0":    np.log(params_df["t0"]),
        "log_tau_c": np.log(params_df["tau_c"]),
    })
    return X.mean().values, X.cov().values, X.columns.tolist()


# ═════════════════════════════════════════════════════════════════
# 6. VISUALIZACIÓN
# ═════════════════════════════════════════════════════════════════

def plot_cancion(params_df, df, titulo, artista, guardar=None):
    """
    Grafica el ranking real y la curva V(t) estimada para una canción.
    Si guardar=None muestra la gráfica; si guardar="ruta.png" la guarda.
    """
    # Buscar parámetros
    mask = (params_df["titulo"] == titulo) & (params_df["artista"] == artista)
    if not mask.any():
        print(f"No encontrada: '{titulo}' — '{artista}'")
        print("Títulos disponibles (muestra):")
        print(params_df["titulo"].sample(10).tolist())
        return

    row = params_df[mask].iloc[0]

    # Datos reales
    grp = df[
        (df["titulo"] == titulo) & (df["artista"] == artista)
    ].sort_values("fecha_chart")

    debut_ts = pd.Timestamp(grp["fecha_debut"].iloc[0])
    t_rels   = np.array(
        [(pd.Timestamp(f) - debut_ts).days // 7 for f in grp["fecha_chart"].values],
        dtype=np.int32,
    )
    rankings = grp["ranking"].values.astype(int)
    y_obs    = 1.0 / rankings.astype(float)

    # V(t) estimada en todos los puntos del rango
    T     = int(t_rels.max())
    V_all = compute_V_vec(T, row["A"], row["p0"], row["t0"], row["tau_c"])
    t_all = np.arange(T + 1)

    matplotlib.use("TkAgg" if guardar is None else "Agg")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{titulo}  —  {artista}", fontsize=13, fontweight="bold")

    # Panel izquierdo: ranking real (eje y invertido: 1 arriba)
    ax = axes[0]
    ax.plot(t_rels, rankings, "o-", color="steelblue", label="Ranking real")
    ax.invert_yaxis()
    ax.set_xlabel("Semanas desde debut")
    ax.set_ylabel("Posición en lista")
    ax.set_title("Ranking observado")
    ax.grid(True, alpha=0.3)

    # Panel derecho: V(t) real (1/ranking) vs estimada
    ax = axes[1]
    ax.plot(t_rels, y_obs,    "o",  color="steelblue", label="1/ranking observado", zorder=3)
    ax.plot(t_all,  V_all,    "-",  color="tomato",    label="V(t) estimada", linewidth=2)
    ax.set_xlabel("Semanas desde debut")
    ax.set_ylabel("Popularidad (1/ranking)")
    ax.set_title("Ajuste del modelo")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Anotar parámetros
    txt = (f"A={row['A']:.3f}  p0={row['p0']:.3f}\n"
           f"t0={row['t0']:.2f}  τc={row['tau_c']:.2f}\n"
           f"MSE={row['mse']:.2e}  n={row['n_semanas']} sem.")
    axes[1].text(0.98, 0.97, txt, transform=axes[1].transAxes,
                 fontsize=8, va="top", ha="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if guardar:
        plt.savefig(guardar, dpi=150)
        print(f"Gráfica guardada en {guardar}")
    else:
        plt.show()

    plt.close()


def plot_multiples(params_df, df, n=6, criterio="mse", guardar=None):
    """
    Grafica las n canciones con mejor o peor MSE para diagnóstico rápido.
    criterio: 'mse' (mejores ajustes) o 'mse_peor' (peores ajustes)
    """
    ascending = criterio == "mse"
    seleccion = params_df.sort_values("mse", ascending=ascending).head(n)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig.suptitle(
        f"{'Mejores' if ascending else 'Peores'} {n} ajustes por MSE",
        fontsize=13, fontweight="bold"
    )

    for ax, (_, row) in zip(axes, seleccion.iterrows()):
        titulo  = row["titulo"]
        artista = row["artista"]

        grp = df[
            (df["titulo"] == titulo) & (df["artista"] == artista)
        ].sort_values("fecha_chart")

        debut_ts = pd.Timestamp(grp["fecha_debut"].iloc[0])
        t_rels   = np.array(
            [(pd.Timestamp(f) - debut_ts).days // 7 for f in grp["fecha_chart"].values],
            dtype=np.int32,
        )
        rankings = grp["ranking"].values.astype(int)
        y_obs    = 1.0 / rankings.astype(float)

        T     = int(t_rels.max())
        V_all = compute_V_vec(T, row["A"], row["p0"], row["t0"], row["tau_c"])

        ax.plot(t_rels, y_obs,              "o",  color="steelblue", markersize=4)
        ax.plot(np.arange(T + 1), V_all,   "-",  color="tomato", linewidth=1.5)
        ax.set_title(f"{titulo[:25]}\n{artista[:20]}", fontsize=8)
        ax.set_xlabel("Semanas", fontsize=7)
        ax.set_ylabel("1/rank", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=150)
        print(f"Gráfica guardada en {guardar}")
    else:
        plt.show()
    plt.close()


# ═════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True)
    parser.add_argument("--output", default="resultados_mc.parquet")
    parser.add_argument("--cores",  type=int,
                        default=max(1, multiprocessing.cpu_count() - 1))
    # Visualización: si se pasan --plot titulo artista, solo grafica (sin reajustar)
    parser.add_argument("--plot",   nargs=2, metavar=("TITULO", "ARTISTA"),
                        help="Graficar una canción específica desde resultados ya guardados")
    parser.add_argument("--plot-top",   action="store_true",
                        help="Mostrar las 6 canciones con mejor ajuste")
    parser.add_argument("--plot-peor",  action="store_true",
                        help="Mostrar las 6 canciones con peor ajuste")
    parser.add_argument("--guardar",    default=None,
                        help="Ruta para guardar la gráfica en vez de mostrarla")
    args = parser.parse_args()

    # ── Leer CSV ─────────────────────────────────────────────────
    print(f"Leyendo {args.csv}...")
    df = pd.read_csv(args.csv, parse_dates=["fecha_chart", "fecha_debut"]).rename(columns={
        "exitos_previos_artista": "x1",
        "colaboracion":           "x2",
        "es_navidena":            "x3",
    })
    df["fecha_chart"] = df["fecha_chart"].dt.normalize()
    df["fecha_debut"] = df["fecha_debut"].dt.normalize()
    df = df[df["fecha_chart"] >= FECHA_INICIO].copy()
    df = df[df["ranking"] <= 50].copy()
    df["ranking"] = df["ranking"].astype(int)

    # ── Modo visualización (sin reajustar) ────────────────────────
    if args.plot or args.plot_top or args.plot_peor:
        import os
        if not os.path.exists(args.output):
            print(f"No se encuentra {args.output}. Ejecuta primero sin --plot.")
            return
        params_df = pd.read_parquet(args.output)
        matplotlib.use("TkAgg")

        if args.plot:
            plot_cancion(params_df, df, args.plot[0], args.plot[1], guardar=args.guardar)
        if args.plot_top:
            plot_multiples(params_df, df, criterio="mse",      guardar=args.guardar)
        if args.plot_peor:
            plot_multiples(params_df, df, criterio="mse_peor", guardar=args.guardar)
        return

    # ── Modo ajuste ───────────────────────────────────────────────
    print(f"  Filas     : {len(df)}")
    print(f"  Semanas   : {df['fecha_chart'].nunique()}")
    print(f"  Canciones : {df.groupby(['titulo','artista']).ngroups}")

    print("\nPreparando argumentos...")
    song_args = build_song_args_mc(df)
    print(f"  Canciones a ajustar: {len(song_args)}")

    chunksize = max(1, len(song_args) // (args.cores * 4))
    print(f"\n[ETAPA 1] Ajustando con {args.cores} cores (chunksize={chunksize})...")
    with multiprocessing.Pool(processes=args.cores) as pool:
        results = pool.map(fit_song_mc, song_args, chunksize=chunksize)

    results = [r for r in results if r is not None]
    print(f"  Ajustadas: {len(results)}")

    params_df = pd.DataFrame(results)
    params_df.to_parquet(args.output, index=False)
    print(f"  Guardado en: {args.output}")

    # Estadísticas de ajuste
    print(f"\n  MSE mediano : {params_df['mse'].median():.4e}")
    print(f"  MSE medio   : {params_df['mse'].mean():.4e}")
    print(f"\n  Distribución de parámetros:")
    print(params_df[["A", "p0", "t0", "tau_c"]].describe().round(4).to_string())

    print("\n[ETAPA 2] Regresión OLS de tau_c...")
    mu, b1, b2, b3, r2 = stage2_regression(params_df)
    print(f"  mu            : {mu:.4f}")
    print(f"  beta1 (hits)  : {b1:.4f}")
    print(f"  beta2 (colab) : {b2:.4f}")
    print(f"  beta3 (navid) : {b3:.4f}")
    print(f"  R²            : {r2:.4f}")

    betas_out = args.output.replace(".parquet", "_betas.parquet")
    pd.DataFrame({
        "parametro": ["mu", "beta1", "beta2", "beta3"],
        "valor":     [mu, b1, b2, b3],
    }).to_parquet(betas_out, index=False)

    print("\n[MVLN] Estimando distribución poblacional...")
    eta, Sigma, col_names = estimate_mvln(params_df)
    print("  Eta:", dict(zip(col_names, eta.round(4))))

    eta_out   = args.output.replace(".parquet", "_eta.parquet")
    sigma_out = args.output.replace(".parquet", "_Sigma.parquet")
    pd.DataFrame({"parametro": col_names, "eta": eta}).to_parquet(eta_out, index=False)
    pd.DataFrame(Sigma, index=col_names, columns=col_names).to_parquet(sigma_out)

    print(f"\n─── Archivos generados ─────────────────────────────────────")
    print(f"  params = pd.read_parquet('{args.output}')")
    print(f"  betas  = pd.read_parquet('{betas_out}')")
    print(f"\n─── Visualización ──────────────────────────────────────────")
    print(f"  # Canción específica:")
    print(f"  python billboard_mincuad.py --csv {args.csv} --output {args.output} \\")
    print(f'      --plot "Titulo" "Artista"')
    print(f"  # Mejores ajustes:")
    print(f"  python billboard_mincuad.py --csv {args.csv} --output {args.output} --plot-top")
    print(f"  # Peores ajustes:")
    print(f"  python billboard_mincuad.py --csv {args.csv} --output {args.output} --plot-peor")


if __name__ == "__main__":
    main()

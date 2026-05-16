"""
Billboard Hot 100 - Estimación por máxima verosimilitud (dos etapas)
=====================================================================
Igual que la versión anterior pero con inicialización inteligente de p0:
  - p0 inicial estimado a partir del ranking de debut de cada canción
  - Múltiples starts que cubren tanto canciones que suben como las que debutan en pico

Uso:
    pip install pandas numpy scipy pyarrow
    python billboard_verosimilitud.py --csv datos.csv --output resultados_vero.parquet
"""

import argparse
import warnings
import multiprocessing

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

FECHA_INICIO = pd.Timestamp("2013-06-01")
MIN_SEMANAS  = 4

# ═════════════════════════════════════════════════════════════════
# 1. MODELO
# ═════════════════════════════════════════════════════════════════

def compute_V_vec(T, A, p0, t0, tau_c):
    n      = T + 1
    taus   = np.arange(n, dtype=np.float64)
    c      = 1.0 / p0 - 1.0
    e_term = np.exp(-taus / t0)
    sp     = A * c * e_term / (t0 * (1.0 + c * e_term) ** 2)
    theta  = np.exp(-taus / tau_c) / (1.0 + taus)
    return np.convolve(sp, theta)[:n]


# ═════════════════════════════════════════════════════════════════
# 2. LOG-VEROSIMILITUD
# ═════════════════════════════════════════════════════════════════

def neg_log_likelihood(params, t_rels, denom_bases):
    log_A, logit_p0, log_t0, log_tau_c = params

    A     = np.exp(log_A)
    p0    = 1.0 / (1.0 + np.exp(-logit_p0))
    t0    = np.exp(log_t0)
    tau_c = np.exp(log_tau_c)

    try:
        V_all = compute_V_vec(int(t_rels.max()), A, p0, t0, tau_c)
    except Exception:
        return 1e12

    Vi        = V_all[t_rels]
    log_denom = np.logaddexp(np.log(np.maximum(denom_bases, 1e-300)), Vi)
    return -np.sum(Vi - log_denom)


# ═════════════════════════════════════════════════════════════════
# 3. AJUSTE POR CANCIÓN con inicialización inteligente de p0
# ═════════════════════════════════════════════════════════════════

def fit_song(args_tuple):
    song_key, t_rels, denom_bases, x1, x2, x3, n_semanas, ranking_debut = args_tuple
    titulo, artista = song_key

    t_rels      = np.asarray(t_rels,      dtype=np.int32)
    denom_bases = np.asarray(denom_bases, dtype=np.float64)

    # ── Inicialización inteligente de p0 ──────────────────────────
    # Si la canción debuta en posición alta (ranking_debut pequeño),
    # p0 debería ser cercano a 1. Usamos 1/ranking_debut normalizado
    # por 1/1 (mejor posición posible) como proxy de p0 inicial.
    # logit(p0_init) = log(p0_init / (1 - p0_init))
    p0_from_debut = min(0.95, max(0.05, 1.0 / ranking_debut))
    logit_p0_debut = np.log(p0_from_debut / (1.0 - p0_from_debut))

    # p0 bajo: canción que sube antes de su pico
    p0_low    = 0.05
    logit_low = np.log(p0_low / (1.0 - p0_low))

    # p0 medio
    p0_mid    = 0.3
    logit_mid = np.log(p0_mid / (1.0 - p0_mid))

    # Puntos de inicio: combinamos debut-informado con los genéricos
    starts = [
        # Informado por el debut
        [0.0,  logit_p0_debut, np.log(5.0),  np.log(5.0)],
        # Canción que sube despacio
        [0.0,  logit_low,      np.log(8.0),  np.log(5.0)],
        # Canción que debuta cerca del pico
        [0.0,  logit_mid,      np.log(3.0),  np.log(3.0)],
        # Memoria larga
        [-0.5, logit_p0_debut, np.log(5.0),  np.log(20.0)],
        # Memoria corta
        [0.5,  logit_p0_debut, np.log(3.0),  np.log(1.0)],
    ]

    bounds = [
        (-5,  5),    # log_A:     A en (0.007, 148)
        (-6,  6),    # logit_p0:  p0 en (0.002, 0.998)
        (-2,  5),    # log_t0:    t0 en (0.13, 148) semanas
        (-2,  5),    # log_tau_c: tau_c en (0.13, 148) semanas
    ]

    best_ll = np.inf
    best_x  = None

    for x0 in starts:
        try:
            res = minimize(
                neg_log_likelihood,
                x0,
                args=(t_rels, denom_bases),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-7},
            )
            if res.fun < best_ll:
                best_ll = res.fun
                best_x  = res.x
        except Exception:
            continue

    if best_x is None:
        return None

    log_A, logit_p0, log_t0, log_tau_c = best_x
    return {
        "titulo":         titulo,
        "artista":        artista,
        "x1": x1, "x2": x2, "x3": x3,
        "A":              float(np.exp(log_A)),
        "p0":             float(1.0 / (1.0 + np.exp(-logit_p0))),
        "t0":             float(np.exp(log_t0)),
        "tau_c":          float(np.exp(log_tau_c)),
        "log_likelihood": float(-best_ll),
        "n_semanas":      n_semanas,
        "ranking_debut":  ranking_debut,
    }


# ═════════════════════════════════════════════════════════════════
# 4. PRECÁLCULO DE DENOMINADORES
# ═════════════════════════════════════════════════════════════════

def precompute_denominators(df):
    exp_inv = np.zeros(52, dtype=np.float64)
    for j in range(1, 52):
        exp_inv[j] = np.exp(1.0 / j)

    week_totals = (
        df.groupby("fecha_chart")["ranking"]
        .apply(lambda rs: float(np.sum(exp_inv[rs.values])))
        .to_dict()
    )

    df2 = df.copy()
    df2["_exp_inv_pos"] = exp_inv[df2["ranking"].values]
    contrib = (
        df2.set_index(["fecha_chart", "titulo", "artista"])["_exp_inv_pos"]
        .to_dict()
    )
    return week_totals, contrib, exp_inv


# ═════════════════════════════════════════════════════════════════
# 5. CONSTRUIR ARGUMENTOS POR CANCIÓN
# ═════════════════════════════════════════════════════════════════

def build_song_args(df, week_totals, contrib, exp_inv):
    exp51     = float(exp_inv[51])
    song_args = []

    for (titulo, artista), grp in df.groupby(["titulo", "artista"]):
        grp = grp.sort_values("fecha_chart")
        n   = len(grp)
        if n < MIN_SEMANAS:
            continue

        debut_ts      = pd.Timestamp(grp["fecha_debut"].iloc[0])
        fechas        = grp["fecha_chart"].values
        ranking_debut = int(grp["ranking"].iloc[0])   # posición en la primera semana
        x1 = float(grp["x1"].iloc[0])
        x2 = float(grp["x2"].iloc[0])
        x3 = float(grp["x3"].iloc[0])

        t_rels = np.array(
            [(pd.Timestamp(f) - debut_ts).days // 7 for f in fechas],
            dtype=np.int32,
        )

        denom_bases = np.array([
            week_totals[f]
            - contrib.get((f, titulo, artista), 0.0)
            + exp51
            for f in fechas
        ], dtype=np.float64)

        song_args.append(
            ((titulo, artista), t_rels, denom_bases, x1, x2, x3, n, ranking_debut)
        )

    return song_args


# ═════════════════════════════════════════════════════════════════
# 6. ETAPA 2: REGRESIÓN DE tau_c Y ESTIMACIÓN MVLN
# ═════════════════════════════════════════════════════════════════

def stage2_regression(params_df):
    X = np.column_stack([
        np.ones(len(params_df)),
        params_df["x1"].values,
        params_df["x2"].values,
        params_df["x3"].values,
    ])
    y      = params_df["tau_c"].values
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu, b1, b2, b3 = coeffs
    residuals = y - X @ coeffs
    r2 = 1 - np.var(residuals) / np.var(y)
    return mu, b1, b2, b3, r2


def estimate_mvln(params_df):
    X = pd.DataFrame({
        "log_A":     np.log(params_df["A"]),
        "logit_p0":  np.log(params_df["p0"] / (1.0 - params_df["p0"])),
        "log_t0":    np.log(params_df["t0"]),
        "log_tau_c": np.log(params_df["tau_c"]),
    })
    return X.mean().values, X.cov().values, X.columns.tolist()


# COMPROBACIÓN
import matplotlib.pyplot as plt

# def plot_cancion(params_df, titulo, artista, df_original):
#     row = params_df[
#         (params_df["titulo"] == titulo) &
#         (params_df["artista"] == artista)
#     ].iloc[0]

#     grp = df_original[
#         (df_original["titulo"] == titulo) &
#         (df_original["artista"] == artista)
#     ].sort_values("fecha_chart")

#     debut = pd.Timestamp(grp["fecha_debut"].iloc[0])
#     t_rels = [(pd.Timestamp(f) - debut).days // 7
#               for f in grp["fecha_chart"].values]
#     rankings = grp["ranking"].values

#     T = max(t_rels)
#     V = compute_V_vec(T, row["A"], row["p0"], row["t0"], row["tau_c"])

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#     # Ranking real (invertido para que arriba = mejor)
#     ax1.plot(t_rels, rankings, "o-")
#     ax1.invert_yaxis()
#     ax1.set_title(f"Ranking real — {titulo}")
#     ax1.set_xlabel("Semanas desde debut")
#     ax1.set_ylabel("Posición")

#     # V(t) estimada (mayor V = mejor posición)
#     ax2.plot(range(T + 1), V)
#     ax2.set_title(f"V(t) estimada — {titulo}")
#     ax2.set_xlabel("Semanas desde debut")
#     ax2.set_ylabel("V(t)")
#     print(f'A: ${row["A"]}, p0: ${row["p0"]}, t0: ${row["t0"]}, tau_c: ${row["tau_c"]}')

#     plt.tight_layout()
#     plt.show()

def plot_cancion(params_df, df, titulo, artista, guardar=None):
    """Grafica ranking real y V(t) estimada para una canción concreta."""
    mask = (params_df["titulo"] == titulo) & (params_df["artista"] == artista)
    if not mask.any():
        print(f"No encontrada: '{titulo}' — '{artista}'")
        print("Muestra de títulos disponibles:")
        print(params_df["titulo"].sample(min(10, len(params_df))).tolist())
        return

    row = params_df[mask].iloc[0]
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
    t_all = np.arange(T + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{titulo}  —  {artista}", fontsize=13, fontweight="bold")

    # Panel izquierdo: ranking real
    ax = axes[0]
    ax.plot(t_rels, rankings, "o-", color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Semanas desde debut")
    ax.set_ylabel("Posición en lista")
    ax.set_title("Ranking observado")
    ax.grid(True, alpha=0.3)

    # Panel derecho: 1/ranking vs V(t)
    ax = axes[1]
    ax.plot(t_rels, y_obs, "o",  color="steelblue", label="1/ranking obs.", zorder=3)
    ax.plot(t_all,  V_all, "-",  color="tomato",    label="V(t) estimada", linewidth=2)
    ax.set_xlabel("Semanas desde debut")
    ax.set_ylabel("Popularidad (1/ranking)")
    ax.set_title("Ajuste del modelo")
    ax.legend()
    ax.grid(True, alpha=0.3)

    txt = (f"A={row['A']:.3f}  p0={row['p0']:.3f}\n"
           f"t0={row['t0']:.2f}  τc={row['tau_c']:.2f}\n"
           f"n={row['n_semanas']} sem.")
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


# ═════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True)
    parser.add_argument("--output", default="resultados_vero.parquet")
    parser.add_argument("--cores",  type=int,
                        default=max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

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

    print(f"  Filas     : {len(df)}")
    print(f"  Semanas   : {df['fecha_chart'].nunique()}")
    print(f"  Canciones : {df.groupby(['titulo','artista']).ngroups}")

    print("\nPrecalculando denominadores...")
    week_totals, contrib, exp_inv = precompute_denominators(df)

    print("Preparando argumentos...")
    song_args = build_song_args(df, week_totals, contrib, exp_inv)
    print(f"  Canciones a ajustar: {len(song_args)}")

    chunksize = max(1, len(song_args) // (args.cores * 4))
    print(f"\n[ETAPA 1] Ajustando con {args.cores} cores (chunksize={chunksize})...")
    with multiprocessing.Pool(processes=args.cores) as pool:
        results = pool.map(fit_song, song_args, chunksize=chunksize)

    results = [r for r in results if r is not None]
    print(f"  Ajustadas: {len(results)}")

    params_df = pd.DataFrame(results)
    params_df.to_parquet(args.output, index=False)
    print(f"  Guardado en: {args.output}")

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
    print(f"  eta    = pd.read_parquet('{eta_out}')")
    print(f"  Sigma  = pd.read_parquet('{sigma_out}').values")

    plot_cancion(params_df, df, 'Animals', 'Maroon 5')


if __name__ == "__main__":
    main()

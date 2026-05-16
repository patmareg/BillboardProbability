"""
Billboard Hot 100 - Estimación de parámetros en dos etapas
===========================================================

ETAPA 1: Por cada canción, optimiza A, p0, t0, tau_c individualmente
         maximizando la log-verosimilitud del ranking observado.

ETAPA 2: Regresión lineal de tau_c ~ mu + b1*x1 + b2*x2 + b3*x3
         para obtener betas globales.

         Finalmente estima (eta, Sigma) de la MVLN poblacional sobre
         [log_A, logit_p0, log_t0, log_tau_c].

Velocidad:
  - compute_V completamente vectorizado (np.convolve en C)
  - Denominadores precalculados una sola vez antes del paralelo
  - Lookup O(1) con dicts nativos
  - multiprocessing.Pool con chunksize adaptativo

Uso:
    pip install pandas numpy scipy pyarrow statsmodels
    python billboard_fitting.py --csv datos.csv --output resultados.parquet
"""

import argparse
import warnings
import multiprocessing

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────
FECHA_INICIO = pd.Timestamp("2013-06-01")   # ignorar primeros 5 meses
MIN_SEMANAS  = 4                             # canciones con menos semanas se descartan


# ═════════════════════════════════════════════════════════════════
# 1. MODELO
# ═════════════════════════════════════════════════════════════════

def compute_V_vec(T, A, p0, t0, tau_c):
    """
    V(k) = Σ_{τ=0}^{k} S'(τ) · Θ(k−τ),  k = 0,…,T

    S'(τ)  = A·(1/p0−1)·exp(−τ/t0) / [t0·(1+(1/p0−1)·exp(−τ/t0))²]
    Θ(lag) = exp(−lag/τ_c) / (1+lag)

    Implementado como convolución discreta con np.convolve (C interno).
    Devuelve array de longitud T+1.
    """
    n      = T + 1
    taus   = np.arange(n, dtype=np.float64)
    c      = 1.0 / p0 - 1.0
    e_term = np.exp(-taus / t0)
    sp     = A * c * e_term / (t0 * (1.0 + c * e_term) ** 2)
    theta  = np.exp(-taus / tau_c) / (1.0 + taus)
    return np.convolve(sp, theta)[:n]


# ═════════════════════════════════════════════════════════════════
# 2. LOG-VEROSIMILITUD DE UNA CANCIÓN
# ═════════════════════════════════════════════════════════════════

def neg_log_likelihood(params, t_rels, denom_bases):
    """
    params      : [log_A, logit_p0, log_t0, log_tau_c]
    t_rels      : array int  — semanas relativas al debut (0-indexed)
    denom_bases : array float — Σ_{j≠i} exp(1/j) + exp(1/51) por semana
                  (constante durante la optimización)

    log L = Σ_k [ V_i(t_k) − log( denom_base_k + exp(V_i(t_k)) ) ]
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

    Vi        = V_all[t_rels]
    log_denom = np.logaddexp(np.log(np.maximum(denom_bases, 1e-300)), Vi)
    return -np.sum(Vi - log_denom)


# ═════════════════════════════════════════════════════════════════
# 3. AJUSTE DE UNA CANCIÓN  (función de nivel módulo para pickle)
# ═════════════════════════════════════════════════════════════════

def fit_song(args_tuple):
    """
    args_tuple: (song_key, t_rels, denom_bases, x1, x2, x3, n_semanas)
    Devuelve dict con parámetros ajustados o None si falla.
    """
    song_key, t_rels, denom_bases, x1, x2, x3, n_semanas = args_tuple
    titulo, artista = song_key

    t_rels      = np.asarray(t_rels,      dtype=np.int32)
    denom_bases = np.asarray(denom_bases, dtype=np.float64)

    # Tres puntos de inicio para evitar mínimos locales
    starts = [
        [ 0.0,  0.0,  np.log(5.0),  np.log(5.0)],
        [ 0.5, -2.0,  np.log(3.0),  np.log(2.0)],
        [-0.5,  2.0,  np.log(10.0), np.log(20.0)],
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
                bounds=[
                    (-5,  5),    # log_A:     A entre e^-5 ≈ 0.007 y e^5 ≈ 148
                    (-6,  6),    # logit_p0:  p0 entre 0.002 y 0.998
                    (-2,  5),    # log_t0:    t0 entre 0.13 y 148 semanas
                    (-2,  5),    # log_tau_c: tau_c entre 0.13 y 148 semanas
                ],
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
    }


# ═════════════════════════════════════════════════════════════════
# 4. PRECÁLCULO DE DENOMINADORES  (O(N), una sola vez)
# ═════════════════════════════════════════════════════════════════

def precompute_denominators(df):
    """
    Devuelve:
      week_totals : dict  fecha → float   (Σ exp(1/j) de todos en lista)
      contrib     : dict  (fecha, titulo, artista) → float  (exp(1/pos_i))
      exp_inv     : array exp(1/j) para j = 0..51
    """
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
# 5. CONSTRUIR ARGUMENTOS POR CANCIÓN  (O(N), una sola vez)
# ═════════════════════════════════════════════════════════════════

def build_song_args(df, week_totals, contrib, exp_inv):
    exp51     = float(exp_inv[51])
    song_args = []

    for (titulo, artista), grp in df.groupby(["titulo", "artista"]):
        grp = grp.sort_values("fecha_chart")
        n   = len(grp)
        if n < MIN_SEMANAS:
            continue

        debut_ts = pd.Timestamp(grp["fecha_debut"].iloc[0])
        fechas   = grp["fecha_chart"].values
        x1 = float(grp["x1"].iloc[0])
        x2 = float(grp["x2"].iloc[0])
        x3 = float(grp["x3"].iloc[0])

        t_rels = np.array(
            [(pd.Timestamp(f) - debut_ts).days // 7 for f in fechas],
            dtype=np.int32,
        )

        # denom_base = total_exp[semana] − exp(1/pos_i) + exp(1/51)
        denom_bases = np.array([
            week_totals[f]
            - contrib.get((f, titulo, artista), 0.0)
            + exp51
            for f in fechas
        ], dtype=np.float64)

        song_args.append(
            ((titulo, artista), t_rels, denom_bases, x1, x2, x3, n)
        )

    return song_args


# ═════════════════════════════════════════════════════════════════
# 6. ETAPA 2: REGRESIÓN DE tau_c Y ESTIMACIÓN MVLN
# ═════════════════════════════════════════════════════════════════

def stage2_regression(params_df):
    """
    Regresión OLS: tau_c ~ mu + b1*x1 + b2*x2 + b3*x3
    Devuelve mu, b1, b2, b3 y los residuos.
    """
    X = np.column_stack([
        np.ones(len(params_df)),
        params_df["x1"].values,
        params_df["x2"].values,
        params_df["x3"].values,
    ])
    y = params_df["tau_c"].values

    # OLS: β = (X'X)^{-1} X'y
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    mu, b1, b2, b3 = coeffs
    residuals = y - X @ coeffs
    return mu, b1, b2, b3, residuals


def estimate_mvln(params_df):
    """
    Transforma [A, p0, t0, tau_c] a escala log/logit y estima
    (eta, Sigma) como media y covarianza muestral (MLE normal multivariante).
    """
    X = pd.DataFrame({
        "log_A":     np.log(params_df["A"]),
        "logit_p0":  np.log(params_df["p0"] / (1.0 - params_df["p0"])),
        "log_t0":    np.log(params_df["t0"]),
        "log_tau_c": np.log(params_df["tau_c"]),
    })
    eta   = X.mean().values
    Sigma = X.cov().values
    return eta, Sigma, X.columns.tolist()

# COMPROBACIÓN
import matplotlib.pyplot as plt

def plot_cancion(params_df, titulo, artista, df_original):
    row = params_df[
        (params_df["titulo"] == titulo) &
        (params_df["artista"] == artista)
    ].iloc[0]

    grp = df_original[
        (df_original["titulo"] == titulo) &
        (df_original["artista"] == artista)
    ].sort_values("fecha_chart")

    debut = pd.Timestamp(grp["fecha_debut"].iloc[0])
    t_rels = [(pd.Timestamp(f) - debut).days // 7
              for f in grp["fecha_chart"].values]
    rankings = grp["ranking"].values

    T = max(t_rels)
    V = compute_V_vec(T, row["A"], row["p0"], row["t0"], row["tau_c"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Ranking real (invertido para que arriba = mejor)
    ax1.plot(t_rels, rankings, "o-")
    ax1.invert_yaxis()
    ax1.set_title(f"Ranking real — {titulo}")
    ax1.set_xlabel("Semanas desde debut")
    ax1.set_ylabel("Posición")

    # V(t) estimada (mayor V = mejor posición)
    ax2.plot(range(T + 1), V)
    ax2.set_title(f"V(t) estimada — {titulo}")
    ax2.set_xlabel("Semanas desde debut")
    ax2.set_ylabel("V(t)")
    print(f'A: ${row["A"]}, p0: ${row["p0"]}, t0: ${row["t0"]}, tau_c: ${row["tau_c"]}')

    plt.tight_layout()
    plt.show()

# ═════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Billboard two-stage fitting")
    parser.add_argument("--csv",    required=True)
    parser.add_argument("--output", default="parametros_canciones.parquet")
    parser.add_argument("--cores",  type=int,
                        default=max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

    # ── Leer y filtrar ────────────────────────────────────────────
    print(f"Leyendo {args.csv}...")
    df = pd.read_csv(
        args.csv,
        parse_dates=["fecha_chart", "fecha_debut"],
    ).rename(columns={
        "exitos_previos_artista": "x1",
        "colaboracion":           "x2",
        "es_navidena":            "x3",
    })
    df["fecha_chart"] = df["fecha_chart"].dt.normalize()
    df["fecha_debut"] = df["fecha_debut"].dt.normalize()
    df = df[df["fecha_chart"] >= FECHA_INICIO].copy()
    df = df[df["ranking"] <= 50].copy()
    df["ranking"] = df["ranking"].astype(int)
    print("Columnas detectadas:", df.columns.tolist())

    n_canciones = df.groupby(["titulo", "artista"]).ngroups
    print(f"  Filas        : {len(df)}")
    print(f"  Semanas      : {df['fecha_chart'].nunique()}")
    print(f"  Canciones    : {n_canciones}")

    # ── Precálculos (una sola vez) ────────────────────────────────
    print("\nPrecalculando denominadores...")
    week_totals, contrib, exp_inv = precompute_denominators(df)

    print("Preparando argumentos por canción...")
    song_args = build_song_args(df, week_totals, contrib, exp_inv)
    print(f"  Canciones a ajustar: {len(song_args)}")

    # ── ETAPA 1: ajuste paralelo ──────────────────────────────────
    chunksize = max(1, len(song_args) // (args.cores * 4))
    print(f"\n[ETAPA 1] Ajustando con {args.cores} cores "
          f"(chunksize={chunksize})...")

    with multiprocessing.Pool(processes=args.cores) as pool:
        results = pool.map(fit_song, song_args, chunksize=chunksize)

    results = [r for r in results if r is not None]
    print(f"  Canciones ajustadas: {len(results)}")

    params_df = pd.DataFrame(results)
    params_df.to_parquet(args.output, index=False)
    print(f"  Parámetros individuales → {args.output}")

    # ── ETAPA 2: regresión de tau_c ───────────────────────────────
    print("\n[ETAPA 2] Regresión OLS de tau_c sobre covariables...")
    mu, b1, b2, b3, residuals = stage2_regression(params_df)

    print(f"  mu (intercepto) : {mu:.4f}")
    print(f"  beta1 (hits prev): {b1:.4f}")
    print(f"  beta2 (colab)    : {b2:.4f}")
    print(f"  beta3 (navidad)  : {b3:.4f}")
    print(f"  R² ajustado      : "
          f"{1 - np.var(residuals)/np.var(params_df['tau_c'].values):.4f}")

    betas_df = pd.DataFrame({
        "parametro": ["mu", "beta1", "beta2", "beta3"],
        "valor":     [mu, b1, b2, b3],
    })
    betas_out = args.output.replace(".parquet", "_betas.parquet")
    betas_df.to_parquet(betas_out, index=False)
    print(f"  Betas → {betas_out}")

    # ── MVLN poblacional ──────────────────────────────────────────
    print("\n[MVLN] Estimando distribución poblacional...")
    eta, Sigma, col_names = estimate_mvln(params_df)

    print("\n  Eta (escala transformada):")
    for name, val in zip(col_names, eta):
        print(f"    {name:12s}: {val:.4f}")

    print("\n  Sigma:")
    sigma_df = pd.DataFrame(Sigma, index=col_names, columns=col_names)
    print(sigma_df.round(4).to_string())

    eta_out   = args.output.replace(".parquet", "_eta.parquet")
    sigma_out = args.output.replace(".parquet", "_Sigma.parquet")
    pd.DataFrame({"parametro": col_names, "eta": eta}).to_parquet(eta_out,   index=False)
    sigma_df.to_parquet(sigma_out)
    print(f"\n  Eta   → {eta_out}")
    print(f"  Sigma → {sigma_out}")

    # ── Instrucciones de lectura ──────────────────────────────────
    print("\n─── Cómo leer los resultados ───────────────────────────────")
    print("import pandas as pd, numpy as np")
    print(f"params = pd.read_parquet('{args.output}')          # parámetros por canción")
    print(f"betas  = pd.read_parquet('{betas_out}')            # mu, beta1, beta2, beta3")
    print(f"eta    = pd.read_parquet('{eta_out}')              # vector eta MVLN")
    print(f"Sigma  = pd.read_parquet('{sigma_out}').values     # matriz Sigma MVLN")

    betas = pd.read_parquet("resultados_betas.parquet")
    print(betas)

    plot_cancion(params_df, 'Drivers License', 'Olivia Rodrigo', df)


if __name__ == "__main__":
    main()

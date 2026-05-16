import matplotlib.pyplot as plt
from billboard_fitting import compute_V_vec

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

    plt.tight_layout()
    plt.show()

plot_cancion('resultados.parquet', 'Something in the water', 'Carrie Underwood', 'billboard_analisis7.csv')



# import pandas as pd
# import numpy as np

# params = pd.read_parquet("resultados.parquet")
# print(params[["A", "p0", "t0", "tau_c", "log_likelihood", "n_semanas"]].describe())
# print("\nCanciones con tau_c > 1000:")
# print(params[params["tau_c"] > 1000][["titulo", "artista", "tau_c", "n_semanas"]])
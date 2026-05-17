"""
Predicciones para la semana siguiente usando error Gumbel
Modelo: ω = V + ε, con ε ~ Gumbel(0, sigma_epsilon)
Uso: python predecir_semana_futura.py --parametros_csv chart_con_parametros.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import gumbel_r
import argparse

# ============================================
# FUNCIONES DEL MODELO
# ============================================

def S(tau_arr, A, p0, t0):
    """Curva logística: S(tau) = A / (1 + (1/p0 - 1)*exp(-tau/t0))"""
    return A / (1.0 + (1.0 / p0 - 1.0) * np.exp(-tau_arr / t0))

def dS(tau_arr, A, p0, t0):
    """Derivada de S respecto a tau"""
    exp_term = np.exp(-tau_arr / t0)
    coef = (1.0 / p0 - 1.0)
    denom = 1.0 + coef * exp_term
    return (A * coef / t0) * exp_term / (denom ** 2)

def compute_V(tau_max: int, tau_c: float, A: float, p0: float, t0: float) -> float:
    """
    V(t) = Σ_{τ=0}^{τ_max} S'(τ) · (1 + τ_max - τ)^{-1} · exp(-(τ_max - τ)/τ_c)
    
    donde:
    - τ_max = t - td (tiempo desde debut)
    - τ_c: parámetro de decaimiento (directo, no se calcula)
    """
    if tau_c <= 0:
        return -np.inf
    
    taus = np.arange(0, tau_max + 1, dtype=float)
    
    # S'(τ)
    s_prime_vals = dS(taus, A, p0, t0)
    
    # (1 + τ_max - τ)^{-1}
    tiempo_restante = tau_max - taus
    weight = 1.0 / (1.0 + tiempo_restante)
    
    # exp(-(τ_max - τ)/τ_c)
    decay = np.exp(-tiempo_restante / tau_c)
    
    return float(np.sum(s_prime_vals * weight * decay))


# ============================================
# PREDICCIÓN CON ERROR GUMBEL
# ============================================

def predecir_semana_futura(fila_params, semanas_actual, 
                           sigma_epsilon=0.5, n_simulaciones=5000):
    """
    Predice ω para la semana siguiente (t+1)
    Modelo: ω = V + ε, con ε ~ Gumbel(0, sigma_epsilon)
    
    Parámetros:
    - fila_params: Serie/row con columnas A, p0, t0, tau_c
    - semanas_actual: τ_max actual = t - td
    - sigma_epsilon: escala del error Gumbel (ε ~ Gumbel(0, sigma_epsilon))
    """
    # Extraer parámetros
    A = float(fila_params['A'])
    p0 = float(fila_params['p0'])
    t0 = float(fila_params['t0'])
    tau_c = float(fila_params['tau_c'])
    
    print(f"   τ_c = {tau_c:.4f}")
    
    # Semana futura
    t_futuro = semanas_actual + 1
    
    # Calcular V(t) para semana actual y futura
    V_actual = compute_V(semanas_actual, tau_c, A, p0, t0)
    V_futuro = compute_V(t_futuro, tau_c, A, p0, t0)
    
    if V_futuro <= 0 or np.isinf(V_futuro):
        return None
    
    # ✅ CORRECTO: ω = V + ε (sin logaritmos ni exponenciales)
    errores = gumbel_r.rvs(loc=0, scale=sigma_epsilon, size=n_simulaciones)
    omega_sim = V_futuro + errores
    
    return {
        'V_actual': V_actual,
        'V_tendencia_futura': V_futuro,
        'tau_c': tau_c,
        'omega_media': np.mean(omega_sim),
        'omega_mediana': np.median(omega_sim),
        'omega_p5': np.percentile(omega_sim, 5),
        'omega_p95': np.percentile(omega_sim, 95),
        'prob_crecimiento': np.mean(omega_sim > V_futuro),
    }


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main_predecir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parametros_csv", default="chart_con_parametros.csv",
                        help="CSV con parámetros extraídos (debe tener A, p0, t0, tau_c)")
    parser.add_argument("--output", default="predicciones_semana_futura.csv")
    parser.add_argument("--sigma_epsilon", type=float, default=0.5,
                        help="Escala del error Gumbel (típicamente 0.3-1.0)")
    parser.add_argument("--n_simulaciones", type=int, default=5000)
    args = parser.parse_args()
    
    # Cargar parámetros extraídos
    df_params = pd.read_csv(args.parametros_csv)
    
    print(f"Columnas disponibles: {df_params.columns.tolist()}")
    
    # Filtrar solo canciones con parámetros válidos (A, p0, t0, tau_c)
    columnas_necesarias = ['A', 'p0', 't0', 'tau_c']
    df_validos = df_params.dropna(subset=columnas_necesarias)
    
    print(f"Canciones con parámetros completos: {len(df_validos)}")
    
    if len(df_validos) == 0:
        print("\n❌ No hay canciones con parámetros válidos.")
        print("   Asegúrate que el CSV tenga las columnas: A, p0, t0, tau_c")
        return
    
    resultados = []
    
    for _, row in df_validos.iterrows():
        print(f"\n--- Procesando: {row['song']} - {row['artist']} ---")
        
        # Obtener semanas en lista
        semanas = row.get('semanas_en_lista', 7)
        print(f"   Semanas en lista: {semanas}")
        
        # Predecir (sin x1,x2,x3 porque tau_c ya está dado)
        pred = predecir_semana_futura(
            row, semanas,
            sigma_epsilon=args.sigma_epsilon,
            n_simulaciones=args.n_simulaciones
        )
        
        if pred:
            resultados.append({
                'rank': row['rank'],
                'song': row['song'],
                'artist': row['artist'],
                'semanas_actual': semanas,
                'tau_c': pred['tau_c'],
                'V_actual': pred['V_actual'],
                'V_tendencia_futura': pred['V_tendencia_futura'],
                'omega_pred_medio': pred['omega_media'],
                'omega_pred_mediana': pred['omega_mediana'],
                'omega_ic90_inf': pred['omega_p5'],
                'omega_ic90_sup': pred['omega_p95'],
                'prob_crecimiento': pred['prob_crecimiento'],
            })
            print(f"   ✅ ω medio = {pred['omega_media']:.2f}")
            print(f"   📈 Prob. crecimiento = {pred['prob_crecimiento']:.1%}")
        else:
            print(f"   ❌ Predicción fallida (V futuro inválido)")
    
    # Guardar resultados
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(args.output, index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Predicciones guardadas en {args.output}")
        print(f"{'='*60}")
        
        # Mostrar top 10
        print("\n🎯 TOP 10 PREDICCIONES PARA LA PRÓXIMA SEMANA")
        print(f"{'='*60}")
        
        top10 = df_resultados.nsmallest(10, 'rank')
        for _, r in top10.iterrows():
            print(f"\n#{r['rank']:3d} | {r['song'][:30]:30s} | {r['artist'][:20]:20s}")
            print(f"     τ_c = {r['tau_c']:.4f}")
            print(f"     V(t) actual = {r['V_actual']:.2f} → V(t+1) tendencia = {r['V_tendencia_futura']:.2f}")
            print(f"     ω esperado: {r['omega_pred_medio']:8.2f} (IC90%: {r['omega_ic90_inf']:6.2f}-{r['omega_ic90_sup']:6.2f})")
            print(f"     Probabilidad de superar tendencia: {r['prob_crecimiento']:.1%}")
        
        # Estadísticas
        print(f"\n{'='*60}")
        print("📊 ESTADÍSTICAS GLOBALES")
        print(f"{'='*60}")
        print(f"Canciones predichas: {len(df_resultados)}")
        print(f"Rango ω medio: [{df_resultados['omega_pred_medio'].min():.2f}, {df_resultados['omega_pred_medio'].max():.2f}]")
        print(f"Prob. crecimiento promedio: {df_resultados['prob_crecimiento'].mean():.1%}")
    else:
        print("\n❌ No se generaron predicciones")

if __name__ == "__main__":
    main_predecir()
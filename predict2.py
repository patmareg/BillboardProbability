import numpy as np
import pandas as pd
import argparse

# ============================================
# FUNCIONES DEL MODELO (con tau_c directo)
# ============================================

def S(tau_arr, A, p0, t0):
    return A / (1.0 + (1.0 / p0 - 1.0) * np.exp(-tau_arr / t0))

def dS(tau_arr, A, p0, t0):
    exp_term = np.exp(-tau_arr / t0)
    coef = (1.0 / p0 - 1.0)
    denom = 1.0 + coef * exp_term
    return (A * coef / t0) * exp_term / (denom ** 2)

def compute_V(tau_max, tau_c, A, p0, t0):
    """
    V(t) = Σ_{τ=0}^{τ_max} S'(τ) · (1 + τ_max - τ)^{-1} · e^{-(τ_max - τ)/τ_c}
    
    Parámetros:
    - tau_max: tiempo desde debut (t - td)
    - tau_c: parámetro de decaimiento (directo, no calculado desde covariables)
    - A, p0, t0: parámetros de la curva S
    """
    if tau_c <= 0:
        return -np.inf
    
    taus = np.arange(0, tau_max + 1, dtype=float)
    
    # S'(τ)
    s_prime_vals = dS(taus, A, p0, t0)
    
    # (1 + τ_max - τ)^{-1}
    tiempo_restante = tau_max - taus
    weight = 1.0 / (1.0 + tiempo_restante)
    
    # e^{-(τ_max - τ)/τ_c}
    decay = np.exp(-tiempo_restante / tau_c)
    
    return float(np.sum(s_prime_vals * weight * decay))

# ============================================
# CÁLCULO DE PROBABILIDADES (FÓRMULA CERRADA)
# ============================================

def prob_primera_posicion(V_vals):
    """
    P(canción i = #1) = e^{V_i} / Σ e^{V_j}
    """
    exp_V = np.exp(V_vals - np.max(V_vals))
    return exp_V / np.sum(exp_V)

def prob_mejor_que(V_i, V_j):
    """
    P(canción i supera a canción j) = 1 / (1 + e^{-(V_i - V_j)})
    """
    diff = V_i - V_j
    return 1.0 / (1.0 + np.exp(-diff))

def prob_ranking_completo(V_vals, orden):
    """
    Probabilidad de un ranking específico
    orden: lista de índices de mejor a peor
    """
    prob = 1.0
    restantes = list(range(len(V_vals)))
    
    for idx in orden:
        if idx not in restantes:
            return 0.0
        V_restantes = [V_vals[j] for j in restantes]
        sum_exp = np.sum(np.exp(V_restantes - np.max(V_restantes)))
        prob_actual = np.exp(V_vals[idx] - np.max(V_restantes)) / sum_exp
        prob *= prob_actual
        restantes.remove(idx)
    
    return prob

# ============================================
# PREDICCIÓN CON PARÁMETROS DIRECTOS
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='resultados_mc.parquet',
                        help='Archivo con parámetros (debe tener A, p0, t0, tau_c)')
    parser.add_argument('--csv_chart', type=str, default='billboard_hot_100.csv',
                        help='CSV con datos del chart')
    parser.add_argument('--output', type=str, default='probabilidades_exactas.csv',
                        help='Archivo de salida')
    args = parser.parse_args()
    
    # Cargar datos
    df_chart = pd.read_csv(args.csv_chart)
    
    if args.params.endswith('.parquet'):
        df_params = pd.read_parquet(args.params)
    else:
        df_params = pd.read_csv(args.params)
    
    print("Columnas en archivo de parámetros:", df_params.columns.tolist())
    
    # Verificar columnas necesarias
    columnas_necesarias = ['A', 'p0', 't0', 'tau_c']
    for col in columnas_necesarias:
        if col not in df_params.columns:
            print(f"❌ Error: Columna '{col}' no encontrada en parámetros")
            return
    
    # Obtener último chart
    if 'date' in df_chart.columns:
        df_chart['date'] = pd.to_datetime(df_chart['date'])
        ultimo_chart = df_chart[df_chart['date'] == df_chart['date'].max()].copy()
        print(f"\n📅 Última semana: {df_chart['date'].max().date()}")
    else:
        ultimo_chart = df_chart.copy()
    
    # Calcular V(t) para cada canción
    resultados = []
    V_vals = []
    canciones_info = []
    
    for _, row in ultimo_chart.iterrows():
        titulo = row['song'].strip()
        artista = row['artist'].strip()
        
        # Buscar parámetros
        match = df_params[
            (df_params['titulo'].str.lower() == titulo.lower()) &
            (df_params['artista'].str.lower() == artista.lower())
        ]
        
        if match.empty:
            match = df_params[df_params['titulo'].str.lower() == titulo.lower()]
        
        if match.empty:
            print(f"⚠️ Sin parámetros: {titulo} - {artista}")
            continue
        
        p = match.iloc[0]
        
        # Obtener semanas desde debut
        semanas = row.get('semanas_en_lista', 7)
        if 'debut_date' in row and 'date' in row:
            debut = pd.to_datetime(row['debut_date'])
            actual = pd.to_datetime(row['date'])
            semanas = (actual - debut).days // 7
        
        # Calcular V(t) para la semana FUTURA (t+1)
        tau_max = semanas + 1
        V_futuro = compute_V(
            tau_max=tau_max,
            tau_c=p['tau_c'],
            A=p['A'],
            p0=p['p0'],
            t0=p['t0']
        )
        
        V_vals.append(V_futuro)
        canciones_info.append({
            'rank': row['rank'],
            'song': titulo,
            'artist': artista,
            'semanas': semanas,
            'tau_c': p['tau_c'],
            'A': p['A'],
            'p0': p['p0'],
            't0': p['t0']
        })
    
    if len(V_vals) == 0:
        print("❌ No se encontraron canciones con parámetros")
        return
    
    # Convertir a array
    V_vals = np.array(V_vals)
    
    # Calcular probabilidades
    prob_num1 = prob_primera_posicion(V_vals)
    
    # Ordenar por V para ranking más probable
    ranking_mp = np.argsort(-V_vals)
    
    # Construir resultados
    for i, info in enumerate(canciones_info):
        resultados.append({
            'rank_actual': info['rank'],
            'song': info['song'],
            'artist': info['artist'],
            'semanas_actual': info['semanas'],
            'tau_c': info['tau_c'],
            'A': info['A'],
            'p0': info['p0'],
            't0': info['t0'],
            'V_tendencia_futura': V_vals[i],
            'prob_num1': prob_num1[i],
            'ranking_esperado': ranking_mp.tolist().index(i) + 1
        })
    
    # Ordenar por rank actual y guardar
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values('rank_actual')
    df_resultados.to_csv(args.output, index=False)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("🎯 PROBABILIDADES EXACTAS (con τ_c directo)")
    print("="*70)
    
    # Top 5 con mayor probabilidad de ser #1
    top5 = df_resultados.nlargest(5, 'prob_num1')
    print("\n📊 Canciones con mayor probabilidad de ser #1 la próxima semana:")
    for _, row in top5.iterrows():
        print(f"\n  #{row['rank_actual']}: {row['song']} - {row['artist']}")
        print(f"     Probabilidad #1: {row['prob_num1']:.2%}")
        print(f"     V(t+1) esperado: {row['V_tendencia_futura']:.2f}")
        print(f"     τ_c: {row['tau_c']:.2f} | A: {row['A']:.2f} | semanas: {row['semanas_actual']}")
    
    # Ranking esperado para el top 10 actual
    print("\n📋 Ranking esperado para la próxima semana (según V):")
    ranking_esperado = df_resultados.sort_values('ranking_esperado')
    for _, row in ranking_esperado.head(10).iterrows():
        flecha = "↑" if row['ranking_esperado'] < row['rank_actual'] else ("↓" if row['ranking_esperado'] > row['rank_actual'] else "→")
        print(f"  #{row['ranking_esperado']:2d} {flecha} (actual #{row['rank_actual']:2d}): {row['song'][:35]}")

if __name__ == "__main__":
    main()
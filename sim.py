"""
Simulación de trayectoria de una canción usando el modelo de Bradlow & Fader

Proceso:
1. Elegir semana aleatoria (2013-2021) y canción del TOP 5 real con parámetros reales
2. SEMANA 0: Usa el Top 100 REAL del Billboard (sin generar canciones nuevas)
3. Semanas siguientes: Cada semana se crean N canciones NUEVAS con parámetros ∼ MVLN(η, Σ)
4. Los parámetros asignados a cada canción se mantienen para siempre
5. Repetir hasta que la canción elegida salga del Top 100

Uso: python simular_trayectoria_cancion.py --nuevas_canciones 15
"""

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r, multivariate_normal
import argparse
from datetime import datetime, timedelta
import random
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# FUNCIONES DEL MODELO
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
    tau_max = t - td (semanas desde debut)
    """
    if tau_c <= 0 or A <= 0 or p0 <= 0 or t0 <= 0:
        return -np.inf
    
    if tau_max < 0:
        return 0.0
    
    tau_max = min(tau_max, 200)
    
    taus = np.arange(0, tau_max + 1, dtype=float)
    s_prime_vals = dS(taus, A, p0, t0)
    tiempo_restante = tau_max - taus
    weight = 1.0 / (1.0 + tiempo_restante)
    decay = np.exp(-tiempo_restante / tau_c)
    
    return float(np.sum(s_prime_vals * weight * decay))


# ============================================
# CARGA DE DATOS
# ============================================

def cargar_eta_Sigma(eta_path, Sigma_path):
    eta_df = pd.read_parquet(eta_path)
    if 'eta' in eta_df.columns:
        eta = eta_df['eta'].values
    else:
        eta = eta_df.values.flatten()
    eta = np.array([float(x) for x in eta])
    
    Sigma_df = pd.read_parquet(Sigma_path)
    if isinstance(Sigma_df, pd.DataFrame):
        Sigma = Sigma_df.values.astype(float)
    else:
        Sigma = np.array(Sigma_df).astype(float)
    
    if len(eta) != 4:
        eta = eta[:4] if len(eta) > 4 else np.pad(eta, (0, 4 - len(eta)), constant_values=np.log(50))
    if Sigma.shape[0] != 4 or Sigma.shape[1] != 4:
        Sigma = np.eye(4) * 0.25
    
    return eta, Sigma

def sample_params_from_mvln(eta, Sigma, n_samples=1):
    log_params = multivariate_normal.rvs(mean=eta, cov=Sigma, size=n_samples)
    
    if n_samples == 1:
        log_params = log_params.reshape(1, -1)
    
    params = []
    for log_p in log_params:
        log_A, logit_p0, log_t0, log_tau_c = log_p
        A = np.exp(log_A)
        p0 = 1.0 / (1.0 + np.exp(-logit_p0))
        t0 = np.exp(log_t0)
        tau_c = np.exp(log_tau_c)
        params.append([A, p0, t0, tau_c])
    
    return np.array(params)

def cargar_datos_billboard(csv_path):
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def cargar_parametros_reales(parametros_csv_path):
    df = pd.read_csv(parametros_csv_path)
    return df

def obtener_canciones_semana(df, fecha):
    return df[df['date'] == fecha].copy()

def obtener_top5_reales(semana_df):
    """Obtiene las 5 primeras canciones del Top real"""
    return semana_df.sort_values('rank').head(5).copy()

def obtener_top50_reales(semana_df):
    """Obtiene las 50 primeras canciones del Top real"""
    return semana_df.sort_values('rank').head(50).copy()

def obtener_top100_reales(semana_df):
    """Obtiene las 100 primeras canciones del Top real"""
    return semana_df.sort_values('rank').head(100).copy()

def obtener_semanas_desde_debut(df_chart, titulo, artista, fecha_actual):
    historial = df_chart[
        (df_chart['song'].str.lower() == titulo.lower()) &
        (df_chart['artist'].str.lower() == artista.lower()) &
        (df_chart['date'] <= fecha_actual)
    ]
    
    if historial.empty:
        return 0
    
    historial = historial.sort_values('date')
    fechas = historial['date'].values
    fecha_actual_dt = pd.to_datetime(fecha_actual)
    
    if fecha_actual_dt in fechas:
        posicion = np.where(fechas == fecha_actual_dt)[0][0]
        return posicion + 1
    else:
        return len(historial)

def obtener_parametros_cancion_real(df_params_reales, titulo, artista):
    """
    Obtiene parámetros reales solo si son válidos y no son nan.
    Retorna None si no existen o son inválidos.
    """
    match = df_params_reales[
        (df_params_reales['song'].str.lower() == titulo.lower()) &
        (df_params_reales['artist'].str.lower() == artista.lower())
    ]
    
    if match.empty:
        return None
    
    p = match.iloc[0]
    
    # Verificar que los valores NO sean nan
    if pd.isna(p['A']) or pd.isna(p['p0']) or pd.isna(p['t0']) or pd.isna(p['tau_c']):
        return None
    
    try:
        A = float(p['A'])
        p0 = float(p['p0'])
        t0 = float(p['t0'])
        tau_c = float(p['tau_c'])
        
        # Verificar que sean números válidos (no inf, no nan)
        if np.isnan(A) or np.isnan(p0) or np.isnan(t0) or np.isnan(tau_c):
            return None
        if np.isinf(A) or np.isinf(p0) or np.isinf(t0) or np.isinf(tau_c):
            return None
        
        # Filtros para parámetros válidos (razonables)
        if A < 10 or A > 500:
            return None
        if p0 < 0.01 or p0 > 0.5:
            return None
        if t0 < 0.5 or t0 > 30:
            return None
        if tau_c < 0.5 or tau_c > 50:
            return None
        
        return {'A': A, 'p0': p0, 't0': t0, 'tau_c': tau_c}
    except (ValueError, TypeError):
        return None


# ============================================
# SELECCIÓN DE CANCIÓN INICIAL (TOP 5)
# ============================================

def seleccionar_cancion_inicial(df_chart, df_params_reales, fecha_inicio=None, max_intentos=100):
    """
    Elige semana aleatoria y canción del TOP 5 real con parámetros reales válidos.
    Si una canción no tiene parámetros válidos, prueba con otra.
    Si ninguna canción del Top 5 tiene parámetros, cambia de fecha.
    """
    
    intentos = 0
    
    while intentos < max_intentos:
        
        if fecha_inicio is None:
            start_date = datetime(2013, 1, 1)
            end_date = datetime(2021, 12, 31)
            fecha_aleatoria = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        else:
            fecha_aleatoria = pd.to_datetime(fecha_inicio)
        
        days_until_saturday = (5 - fecha_aleatoria.weekday()) % 7
        fecha_aleatoria = fecha_aleatoria + timedelta(days=days_until_saturday)
        
        fechas_disponibles = df_chart['date'].unique()
        fechas_cercanas = fechas_disponibles[fechas_disponibles >= fecha_aleatoria]
        
        if len(fechas_cercanas) == 0:
            fechas_cercanas = fechas_disponibles
        
        if len(fechas_cercanas) == 0:
            intentos += 1
            continue
        
        fecha_candidata = fechas_cercanas[0]
        
        semana_df = obtener_canciones_semana(df_chart, fecha_candidata)
        top5_real = obtener_top5_reales(semana_df)
        
        canciones_con_params = []
        canciones_sin_params = []
        
        for _, row in top5_real.iterrows():
            titulo = row['song'].strip()
            artista = row['artist'].strip()
            
            params = obtener_parametros_cancion_real(df_params_reales, titulo, artista)
            
            if params is not None:
                canciones_con_params.append({
                    'rank': row['rank'],
                    'song': titulo,
                    'artist': artista,
                    'A': params['A'],
                    'p0': params['p0'],
                    't0': params['t0'],
                    'tau_c': params['tau_c']
                })
            else:
                canciones_sin_params.append({
                    'rank': row['rank'],
                    'song': titulo,
                    'artist': artista
                })
        
        if len(canciones_con_params) > 0:
            print(f"  ✅ Semana encontrada: {fecha_candidata.date()}")
            print(f"  Canciones en TOP 5 con parámetros válidos: {len(canciones_con_params)}")
            
            for c in canciones_con_params:
                print(f"     ✅ #{c['rank']} - {c['song']} - {c['artist']}")
            
            # Mostrar canciones sin parámetros (para diagnóstico)
            if len(canciones_sin_params) > 0:
                print(f"  Canciones sin parámetros válidos (ignoradas):")
                for c in canciones_sin_params:
                    print(f"     ❌ #{c['rank']} - {c['song']} - {c['artist']}")
            
            cancion_elegida = random.choice(canciones_con_params)
            print(f"\n  ✨ Canción elegida: #{cancion_elegida['rank']} - {cancion_elegida['song']}")
            return fecha_candidata, cancion_elegida
        else:
            print(f"  ⚠️ Semana {fecha_candidata.date()}: ninguna canción del TOP 5 tiene parámetros válidos. Reintentando...")
            intentos += 1
            
            # Avanzar a la siguiente semana
            siguientes_fechas = fechas_disponibles[fechas_disponibles > fecha_candidata]
            if len(siguientes_fechas) > 0:
                fecha_inicio = siguientes_fechas[0]
    
    raise ValueError(f"No se encontró ninguna canción del TOP 5 con parámetros válidos después de {max_intentos} intentos")


# ============================================
# SIMULACIÓN DE RANKING
# ============================================

def simular_ranking(canciones_con_V, sigma_epsilon=1.0):
    if len(canciones_con_V) == 0:
        return [], {}
    
    n = len(canciones_con_V)
    errores = gumbel_r.rvs(loc=0, scale=sigma_epsilon, size=n)
    
    for i, cancion in enumerate(canciones_con_V):
        cancion['omega'] = cancion['V'] + errores[i]
    
    ranking = sorted(canciones_con_V, key=lambda x: x['omega'], reverse=True)
    
    posiciones = {}
    for pos, cancion in enumerate(ranking):
        clave = f"{cancion['song']}|{cancion['artist']}"
        posiciones[clave] = pos + 1
    
    return ranking, posiciones


# ============================================
# SIMULACIÓN DE UNA TRAYECTORIA
# ============================================

def simular_trayectoria(sim_id, fecha_inicio, cancion_elegida, df_chart, 
                        df_params_reales, eta, Sigma, 
                        sigma_epsilon=1.0, max_semanas=200, 
                        nuevas_canciones_por_semana=15, verbose=True):
    """
    Simula la trayectoria completa:
    - SEMANA 0: Usa el Top 100 REAL del Billboard (sin generar canciones nuevas)
    - Semanas siguientes: Cada semana se crean N canciones nuevas con MVLN
    """
    
    titulo = cancion_elegida['song']
    artista = cancion_elegida['artist']
    A_real = cancion_elegida['A']
    p0_real = cancion_elegida['p0']
    t0_real = cancion_elegida['t0']
    tau_c_real = cancion_elegida['tau_c']
    rank_inicial = cancion_elegida['rank']
    
    posiciones = []
    fecha_actual = fecha_inicio
    
    # Diccionario para almacenar parámetros de todas las canciones
    parametros_canciones = {}
    
    # Registrar la canción elegida
    clave_cancion = f"{titulo}|{artista}"
    parametros_canciones[clave_cancion] = {
        'A': A_real, 'p0': p0_real, 't0': t0_real, 'tau_c': tau_c_real
    }
    
    if verbose and sim_id == 1:
        print(f"     Rank inicial: #{rank_inicial}")
        print(f"     Parámetros: A={A_real:.2f}, p0={p0_real:.4f}, t0={t0_real:.2f}, tau_c={tau_c_real:.2f}")
        print(f"     Nuevas canciones por semana: {nuevas_canciones_por_semana}")
    
    # =========================================================
    # SEMANA 0: USAR DATOS REALES DEL BILLBOARD (NO GENERAR NUEVAS)
    # =========================================================
    
    semana_df = obtener_canciones_semana(df_chart, fecha_actual)
    top100_real = obtener_top100_reales(semana_df)
    
    lista_semana0 = []
    
    for _, row in top100_real.iterrows():
        clave = f"{row['song']}|{row['artist']}"
        
        # Obtener parámetros reales si existen
        params = obtener_parametros_cancion_real(df_params_reales, row['song'], row['artist'])
        
        if params is not None:
            parametros_canciones[clave] = params
            A_comp = params['A']
            p0_comp = params['p0']
            t0_comp = params['t0']
            tau_c_comp = params['tau_c']
            
            semanas_comp = obtener_semanas_desde_debut(df_chart, row['song'], row['artist'], fecha_actual)
            V_comp = compute_V(semanas_comp, tau_c_comp, A_comp, p0_comp, t0_comp)
            
            if V_comp > 0:
                lista_semana0.append({
                    'song': row['song'],
                    'artist': row['artist'],
                    'V': V_comp
                })
        else:
            # Si no hay parámetros reales, estimar V a partir del ranking
            rank = row['rank']
            V_comp = 1.0 / rank if rank > 0 else 0.01
            lista_semana0.append({
                'song': row['song'],
                'artist': row['artist'],
                'V': V_comp
            })
    
    # Añadir la canción elegida
    V_cancion = compute_V(0, tau_c_real, A_real, p0_real, t0_real)
    lista_semana0.append({
        'song': titulo,
        'artist': artista,
        'V': V_cancion
    })
    
    # Simular ranking para semana 0
    ranking_semana0, posiciones_dict = simular_ranking(lista_semana0.copy(), sigma_epsilon)
    posicion = posiciones_dict.get(clave_cancion, None)
    
    if posicion is not None:
        posiciones.append(posicion)
        if verbose:
            print(f"     Semana 0 (real): posición {posicion}")
    
    # Verificar si sale en semana 0
    if posicion and posicion > 100:
        if verbose:
            print(f"     🚪 Sale en semana 0 (ranking real)")
        return {
            'sim_id': sim_id,
            'cancion': titulo,
            'artista': artista,
            'semanas': 1,
            'posiciones': posiciones
        }
    
    # =========================================================
    # SEMANAS SIGUIENTES (1, 2, 3...): GENERAR NUEVAS CANCIONES
    # =========================================================
    
    semana_actual_num = 1
    semanas_desde_debut = 1
    V_cancion = compute_V(semanas_desde_debut, tau_c_real, A_real, p0_real, t0_real)
    
    if V_cancion <= 0:
        if verbose:
            print(f"     ❌ V inválido en semana 1")
        return None
    
    while semana_actual_num < max_semanas:
        
        # Avanzar a la siguiente semana real
        fechas_futuras = df_chart[df_chart['date'] > fecha_actual]['date'].unique()
        if len(fechas_futuras) > 0:
            fecha_actual = fechas_futuras[0]
        else:
            break
        
        semana_df = obtener_canciones_semana(df_chart, fecha_actual)
        top100_real = obtener_top100_reales(semana_df)
        
        nueva_lista = []
        nuevas_generadas = 0
        
        for idx, (_, row) in enumerate(top100_real.iterrows()):
            clave = f"{row['song']}|{row['artist']}"
            
            # Si la canción no tiene parámetros asignados aún
            if clave not in parametros_canciones:
                # Las primeras N canciones se generan como nuevas
                if nuevas_generadas < nuevas_canciones_por_semana:
                    params = sample_params_from_mvln(eta, Sigma, n_samples=1)[0]
                    A_comp, p0_comp, t0_comp, tau_c_comp = params
                    A_comp = min(max(A_comp, 10), 150)
                    p0_comp = min(max(p0_comp, 0.05), 0.3)
                    t0_comp = min(max(t0_comp, 1), 15)
                    tau_c_comp = min(max(tau_c_comp, 1), 20)
                    
                    parametros_canciones[clave] = {
                        'A': A_comp, 'p0': p0_comp, 't0': t0_comp, 'tau_c': tau_c_comp
                    }
                    nuevas_generadas += 1
                else:
                    # El resto usan MVLN pero no se cuentan como nuevas
                    params = sample_params_from_mvln(eta, Sigma, n_samples=1)[0]
                    A_comp, p0_comp, t0_comp, tau_c_comp = params
                    A_comp = min(max(A_comp, 10), 150)
                    p0_comp = min(max(p0_comp, 0.05), 0.3)
                    t0_comp = min(max(t0_comp, 1), 15)
                    tau_c_comp = min(max(tau_c_comp, 1), 20)
                    
                    parametros_canciones[clave] = {
                        'A': A_comp, 'p0': p0_comp, 't0': t0_comp, 'tau_c': tau_c_comp
                    }
            
            # Obtener parámetros y calcular V
            params = parametros_canciones[clave]
            semanas_comp = obtener_semanas_desde_debut(df_chart, row['song'], row['artist'], fecha_actual)
            V_comp = compute_V(semanas_comp, params['tau_c'], params['A'], params['p0'], params['t0'])
            
            if V_comp > 0:
                nueva_lista.append({
                    'song': row['song'],
                    'artist': row['artist'],
                    'V': V_comp
                })
        
        # Mostrar estadísticas de nuevas canciones
        if verbose and sim_id == 1 and semana_actual_num == 1:
            print(f"     Semana 1: {nuevas_generadas} nuevas canciones generadas")
        
        # Añadir la canción elegida
        nueva_lista.append({
            'song': titulo,
            'artist': artista,
            'V': V_cancion
        })
        
        # Simular ranking
        ranking_simulado, posiciones_dict = simular_ranking(nueva_lista.copy(), sigma_epsilon)
        posicion = posiciones_dict.get(clave_cancion, None)
        
        if posicion is None:
            break
        
        posiciones.append(posicion)
        
        if verbose and (semana_actual_num % 10 == 0 or semana_actual_num < 5):
            print(f"     Semana {semana_actual_num}: posición {posicion}")
        
        # CRITERIO DE PARADA
        if posicion > 100:
            if verbose:
                print(f"     🚪 Sale en semana {semana_actual_num} (posición {posicion})")
            break
        
        # Preparar siguiente semana
        semana_actual_num += 1
        semanas_desde_debut += 1
        V_cancion = compute_V(semanas_desde_debut, tau_c_real, A_real, p0_real, t0_real)
        
        if V_cancion <= 0:
            if verbose:
                print(f"     ❌ V inválido en semana {semana_actual_num}")
            break
    
    if len(posiciones) == 0:
        return None
    
    return {
        'sim_id': sim_id,
        'cancion': titulo,
        'artista': artista,
        'rank_inicial': rank_inicial,
        'semanas': len(posiciones),
        'posiciones': posiciones
    }


# ============================================
# FUNCIÓN PARA LIMPIAR PARÁMETROS (OPCIONAL)
# ============================================

def limpiar_parametros(df_params_reales):
    """Limpia el DataFrame de parámetros eliminando filas con nan o valores inválidos"""
    df_clean = df_params_reales.copy()
    
    # Eliminar filas con nan
    df_clean = df_clean.dropna(subset=['A', 'p0', 't0', 'tau_c'])
    
    # Filtrar valores razonables
    df_clean = df_clean[
        (df_clean['A'] > 10) & (df_clean['A'] < 500) &
        (df_clean['p0'] > 0.01) & (df_clean['p0'] < 0.5) &
        (df_clean['t0'] > 0.5) & (df_clean['t0'] < 30) &
        (df_clean['tau_c'] > 0.5) & (df_clean['tau_c'] < 50)
    ]
    
    return df_clean


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Simulación de trayectoria de canciones Billboard')
    parser.add_argument('--csv', default='billboard_hot_100.csv',
                        help='CSV con datos del Billboard')
    parser.add_argument('--params_reales', default='chart_con_parametros.csv',
                        help='CSV con parámetros reales de canciones')
    parser.add_argument('--eta_path', default='resultados_mc_eta.parquet',
                        help='Archivo con eta (medias MVLN)')
    parser.add_argument('--Sigma_path', default='resultados_mc_Sigma.parquet',
                        help='Archivo con Sigma (covarianza MVLN)')
    parser.add_argument('--sigma_epsilon', type=float, default=1.0,
                        help='Escala del error Gumbel')
    parser.add_argument('--n_simulaciones', type=int, default=100,
                        help='Número de simulaciones')
    parser.add_argument('--max_semanas', type=int, default=200,
                        help='Máximo de semanas por simulación')
    parser.add_argument('--nuevas_canciones', type=int, default=15,
                        help='Número de canciones nuevas generadas por semana (default: 15)')
    parser.add_argument('--output_dir', default='resultados_simulaciones',
                        help='Directorio para guardar resultados')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Mostrar información detallada')
    parser.add_argument('--clean_params', action='store_true', default=False,
                        help='Limpiar parámetros automáticamente (eliminar nan)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("📂 CARGANDO DATOS")
    print("="*60)
    
    df_chart = cargar_datos_billboard(args.csv)
    df_params_reales = cargar_parametros_reales(args.params_reales)
    
    print(f"  Billboard: {len(df_chart)} registros")
    print(f"  Parámetros originales: {len(df_params_reales)} canciones")
    
    # Limpiar parámetros si se solicita
    if args.clean_params:
        df_params_reales = limpiar_parametros(df_params_reales)
        print(f"  Parámetros después de limpiar: {len(df_params_reales)} canciones")
    
    validos = df_params_reales.dropna(subset=['A', 'p0', 't0', 'tau_c'])
    print(f"  Canciones con parámetros reales válidos: {len(validos)}")
    
    if len(validos) == 0:
        print("\n❌ ERROR: No hay parámetros reales válidos")
        print("   Ejecuta primero: python extraer_chart.py")
        print("   O usa --clean_params para limpiar automáticamente")
        return
    
    try:
        eta, Sigma = cargar_eta_Sigma(args.eta_path, args.Sigma_path)
        print(f"  eta (MVLN): {eta.round(4)}")
    except Exception as e:
        print(f"  Error cargando eta/Sigma: {e}")
        print("  Usando valores por defecto...")
        eta = np.array([np.log(80), np.log(0.1), np.log(5), np.log(8)])
        Sigma = np.eye(4) * 0.3
    
    print("\n" + "="*60)
    print("🎲 SELECCIONANDO CANCIÓN INICIAL (TOP 5)")
    print("="*60)
    
    try:
        fecha_inicio, cancion = seleccionar_cancion_inicial(df_chart, df_params_reales)
        print(f"  Fecha: {fecha_inicio.date()}")
        print(f"  Canción elegida: #{cancion['rank']} - {cancion['song']} - {cancion['artist']}")
        print(f"  Parámetros: A={cancion['A']:.2f}, p0={cancion['p0']:.4f}, "
              f"t0={cancion['t0']:.2f}, tau_c={cancion['tau_c']:.2f}")
    except ValueError as e:
        print(f"\n❌ {e}")
        return
    
    print("\n" + "="*60)
    print(f"🚀 {args.n_simulaciones} SIMULACIONES")
    print(f"   sigma_epsilon = {args.sigma_epsilon}")
    print(f"   Nuevas canciones por semana = {args.nuevas_canciones}")
    print("   Semana 0: usa Top 100 REAL del Billboard")
    print("   Canción elegida del TOP 5 real")
    print("="*60)
    
    resultados = []
    
    for sim in range(1, args.n_simulaciones + 1):
        if args.verbose:
            print(f"\n  Simulación {sim}/{args.n_simulaciones}:")
        
        res = simular_trayectoria(
            sim, fecha_inicio, cancion, df_chart, df_params_reales,
            eta, Sigma, args.sigma_epsilon, args.max_semanas, 
            args.nuevas_canciones, args.verbose
        )
        
        if res and res['semanas'] > 0:
            resultados.append(res)
            if args.verbose:
                print(f"    ✅ {res['semanas']} semanas")
        else:
            if args.verbose:
                print(f"    ❌ Fallida")
    
    if len(resultados) == 0:
        print("\n❌ No se generaron resultados")
        return
    
    # Guardar resultados
    df_res = pd.DataFrame([{
        'simulacion': r['sim_id'],
        'cancion': r['cancion'],
        'artista': r['artista'],
        'rank_inicial': r['rank_inicial'],
        'semanas': r['semanas']
    } for r in resultados])
    df_res.to_csv(os.path.join(args.output_dir, 'resumen.csv'), index=False)
    
    # Guardar evolución
    evolucion = []
    for r in resultados:
        for semana, pos in enumerate(r['posiciones'], 1):
            evolucion.append({
                'simulacion': r['sim_id'],
                'semana': semana,
                'posicion': pos
            })
    df_evol = pd.DataFrame(evolucion)
    df_evol.to_csv(os.path.join(args.output_dir, 'evolucion.csv'), index=False)
    
    # Estadísticas
    semanas = [r['semanas'] for r in resultados]
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS")
    print("="*60)
    print(f"  Simulaciones exitosas: {len(resultados)}/{args.n_simulaciones}")
    print(f"  Media de semanas: {np.mean(semanas):.1f}")
    print(f"  Mediana: {np.median(semanas):.1f}")
    print(f"  Mínimo: {min(semanas)}")
    print(f"  Máximo: {max(semanas)}")
    print(f"  Desv. estándar: {np.std(semanas):.1f}")
    
    print(f"\n✅ Resultados guardados en {args.output_dir}")

if __name__ == "__main__":
    main()
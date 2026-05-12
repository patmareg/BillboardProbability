# CAMBIOS: Eliminar imports que ya no se necesitan
# import billboard  # <--- ELIMINADO
import csv
from datetime import datetime, timedelta  # <--- timedelta todavia se usa para fechas? No, pero lo dejamos por si acaso
# import time  # <--- ELIMINADO (ya no hay pausas)
from collections import defaultdict
import re
import pandas as pd  # <--- NUEVO IMPORT

def es_colaboracion(artista):
    """
    Detecta si hay colaboracion en el nombre del artista.
    """
    artista_lower = artista.lower()
    
    # Patrones de colaboracion
    patrones_colab = [
        r'\bfeat\.?\b',           # feat, feat.
        r'\bfeat\.?uring\b',      # featuring
        r'\bfeaturing\b',         # featuring
        r'\bft\.?\b',             # ft, ft.
        r'\bwith\b',              # with
        r' & ',                   # & con espacios
        r' x ',                   # x entre artistas
        r'\band\b',               # and como palabra completa
    ]
    
    for patron in patrones_colab:
        if re.search(patron, artista_lower):
            return 1
    
    # Caso especial: comas que separan artistas
    if ', ' in artista_lower:
        partes = artista_lower.split(', ')
        if len(partes) > 2:
            return 1
        if len(partes) == 2 and (' ' in partes[1] or len(partes[1]) > 15):
            return 1
    
    return 0

def separar_artistas(artista_str):
    """
    Separa artistas en colaboraciones.
    """
    artista_str = artista_str.strip()
    artista_lower = artista_str.lower()
    
    # Lista de separadores
    separadores = [
        (r' featuring ', 'featuring'),
        (r' feat\.? ', 'feat'),
        (r' ft\.? ', 'ft'),
        (r' & ', '&'),
        (r' with ', 'with'),
        (r' x ', 'x'),
        (r', ', 'comma'),
        (r' and ', 'and'),
    ]
    
    for sep, nombre in separadores:
        if re.search(sep, artista_lower):
            partes = re.split(sep, artista_str, flags=re.IGNORECASE)
            artistas = []
            for p in partes:
                p_limpio = p.strip()
                for sufijo in [' feat', ' featuring', ' ft', ' with']:
                    p_limpio = p_limpio.split(sufijo)[0]
                p_limpio = p_limpio.strip()
                if p_limpio and len(p_limpio) > 1:
                    artistas.append(p_limpio)
            if len(artistas) > 1:
                return artistas
    
    return [artista_str.strip()]

# ============ VERIFICACION DE FUNCION ============
print("=" * 70)
print("VERIFICANDO FUNCION DE COLABORACION")
print("=" * 70)

casos_prueba = [
    ("Taylor Swift", 0),
    ("Kendrick Lamar Featuring Drake", 1),
    ("Kendrick Lamar featuring Drake", 1),
    ("Post Malone feat. Morgan Wallen", 1),
    ("Ed Sheeran & Justin Bieber", 1),
    ("Drake ft. Future", 1),
    ("Mariah Carey", 0),
    ("Bad Bunny, Jhay Cortez", 1),
    ("The Beatles", 0),
    ("Macklemore & Ryan Lewis", 1),
    ("Lady Gaga with Ariana Grande", 1),
]

print("Probando deteccion:")
print("-" * 70)
for artista, esperado in casos_prueba:
    resultado = es_colaboracion(artista)
    estado = "OK" if resultado == esperado else "ERROR"
    tipo = "COLAB" if resultado == 1 else "SOLO"
    print(f"{estado:5} | {tipo:5} | {artista[:50]:50} -> {resultado} (esperado {esperado})")

# ============ CONFIGURACION PRINCIPAL ============
print("\n" + "=" * 60)
print("EXTRACCION BILLBOARD HOT 100 (2013-2024)")
print("=" * 60)

# CAMBIO: Cargar dataset en lugar de generar fechas y hacer scraping
print("Cargando archivo billboard_hot_100.csv...")
df = pd.read_csv('billboard_hot_100.csv')

# Verificar nombres de columnas (ajustar segun el archivo)
# Las columnas tipicas son: date, rank, song, artist
df = df.rename(columns={
    'date': 'fecha',
    'rank': 'posicion',
    'song': 'titulo',
    'artist': 'artista'
})

# Convertir fecha y filtrar desde 2013 hasta 2024
df['fecha'] = pd.to_datetime(df['fecha'])
df = df[(df['fecha'] >= '2013-01-01') & (df['fecha'] <= '2024-12-31')]

# Ordenar por fecha (cronologico)
df = df.sort_values('fecha')

print(f"Registros cargados: {len(df)}")
print(f"Rango de fechas: {df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}")
print("-" * 60)

# ============ INICIALIZAR ============
canciones_por_artista = defaultdict(set)
debut_cancion = {}
resultados = []

# ============ PROCESAR CADA REGISTRO ============
# CAMBIO: Iterar sobre el DataFrame en lugar de semanas
for idx, row in df.iterrows():
    fecha_top_str = row['fecha'].strftime('%Y-%m-%d')
    titulo = row['titulo']
    artista_original = row['artista']
    posicion = int(row['posicion'])
    
    if not titulo or not artista_original:
        continue
    
    artistas_individuales = separar_artistas(artista_original)
    
    if len(artistas_individuales) > 1:
        artista_limpio = " & ".join(artistas_individuales)
    else:
        artista_limpio = artistas_individuales[0]
    
    es_colab = es_colaboracion(artista_original)
    
    exitos_previos = 0
    for artista in artistas_individuales:
        exitos_previos += len(canciones_por_artista[artista])
    
    clave_cancion = f"{titulo.lower()}|{artista_limpio.lower()}"
    if clave_cancion not in debut_cancion:
        debut_cancion[clave_cancion] = fecha_top_str
    
    # Mostrar progreso cada 10000 registros
    if len(resultados) % 10000 == 0:
        print(f"Procesados {len(resultados)} registros...")
    
    resultados.append({
        'fecha_chart': fecha_top_str,
        'fecha_debut': debut_cancion[clave_cancion],
        'titulo': titulo,
        'artista': artista_limpio,
        'colaboracion': es_colab,
        'exitos_previos_artista': exitos_previos,
        'ranking': posicion
    })
    
    for artista in artistas_individuales:
        canciones_por_artista[artista].add(titulo)

# ============ GUARDAR RESULTADOS ============
with open('billboard_analisis.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['fecha_chart', 'fecha_debut', 'titulo', 'artista', 'colaboracion', 'exitos_previos_artista', 'ranking'])
    writer.writeheader()
    writer.writerows(resultados)

# ============ VERIFICAR TAYLOR SWIFT ============
print("\n" + "=" * 70)
print("VERIFICANDO CANCIONES DE TAYLOR SWIFT")
print("=" * 70)

taylor_songs = [r for r in resultados if 'taylor' in r['artista'].lower()]
for r in taylor_songs[:15]:
    estado = "COLAB" if r['colaboracion'] else "SOLO"
    print(f"{r['fecha_chart']} | #{r['ranking']:2} | {r['titulo'][:35]:35} | {r['artista'][:25]:25} | {estado}")

# ============ BUSCAR POETIC JUSTICE ============
print("\n" + "=" * 70)
print("BUSCANDO 'Poetic Justice' (Kendrick Lamar Featuring Drake)")
print("=" * 70)

poetic = [r for r in resultados if 'poetic' in r['titulo'].lower()]
for r in poetic:
    estado = "COLAB" if r['colaboracion'] else "SOLO"
    print(f"Titulo: {r['titulo']}")
    print(f"Artista guardado: {r['artista']}")
    print(f"Colaboracion: {r['colaboracion']} ({estado})")
    print(f"Ranking: {r['ranking']}")

# ============ ESTADISTICAS FINALES ============
print("\n" + "=" * 60)
print("EXTRACCION COMPLETADA")
print("=" * 60)
print(f"Total de registros: {len(resultados)}")
print(f"Archivo: billboard_analisis.csv")

total_colabs = sum(1 for r in resultados if r['colaboracion'] == 1)
if len(resultados) > 0:
    print(f"\nEstadisticas de colaboraciones:")
    print(f"   Total canciones: {len(resultados)}")
    print(f"   Colaboraciones: {total_colabs} ({(total_colabs/len(resultados))*100:.1f}%)")
    print(f"   Solistas: {len(resultados)-total_colabs}")
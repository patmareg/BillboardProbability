import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import time

def es_colaboracion(artista):
    """Detecta si un artista incluye colaboraciones en su nombre"""
    patrones = ['feat', '&', ' and ', ' with ', ',', 'Feat', 'Featuring', ' Ft ']
    for patron in patrones:
        if patron in artista:
            return 1
    return 0

def extraer_semana(fecha_str):
    """
    Extrae el Billboard Hot 100 para una fecha específica
    Args:
        fecha_str: Fecha en formato YYYY-MM-DD (ej: '2024-12-07')
    
    Returns:
        Lista de diccionarios con canciones, artistas y posiciones
    """
    try:
        # Construir URL
        url = f"https://www.billboard.com/charts/hot-100/{fecha_str}/"
        
        # Headers para evitar bloqueos
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"  Error HTTP: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        resultados = []
        
        # Método 1: Buscar elementos de chart
        chart_items = soup.select('div.o-chart-results-list-row-container')
        
        if not chart_items:
            # Intentar selectores alternativos
            chart_items = soup.select('ul.o-chart-results-list-row')
        
        for item in chart_items:
            try:
                # Extraer canción
                song_elem = item.select_one('h3.c-title')
                if not song_elem:
                    song_elem = item.select_one('h3#title-of-a-story')
                
                # Extraer artista
                artist_elem = item.select_one('span.c-label')
                if not artist_elem:
                    artist_elem = item.select_one('li.lrv-u-width-100p span.c-label')
                
                # Extraer posición
                rank_elem = item.select_one('span.c-label.a-font-bold')
                if not rank_elem:
                    rank_elem = item.select_one('[data-rank]')
                
                if song_elem and artist_elem:
                    cancion = song_elem.get_text(strip=True)
                    artista = artist_elem.get_text(strip=True)
                    rank = rank_elem.get_text(strip=True) if rank_elem else '0'
                    
                    # Limpiar textos
                    cancion = cancion.replace('\n', '').replace('\t', '').strip()
                    artista = artista.replace('\n', '').replace('\t', '').strip()
                    
                    try:
                        rank_int = int(rank)
                    except:
                        rank_int = 0
                    
                    if cancion and artista and rank_int > 0:
                        resultados.append({
                            'fecha': fecha_str,
                            'posicion': rank_int,
                            'cancion': cancion,
                            'artista': artista,
                            'es_colaboracion': es_colaboracion(artista)
                        })
            
            except Exception as e:
                continue
        
        # Si no encontramos datos con los selectores principales, intentar método alternativo
        if not resultados:
            # Método 2: Buscar por listas
            all_songs = soup.select('li.o-chart-results-list__item')
            
            for i, item in enumerate(all_songs, 1):
                try:
                    song_elem = item.select_one('h3.c-title')
                    if not song_elem:
                        song_elem = item.select_one('h3')
                    
                    artist_elem = item.select_one('span.c-label')
                    
                    if song_elem and artist_elem:
                        resultados.append({
                            'fecha': fecha_str,
                            'posicion': i,
                            'cancion': song_elem.get_text(strip=True),
                            'artista': artist_elem.get_text(strip=True),
                            'es_colaboracion': es_colaboracion(artist_elem.get_text(strip=True))
                        })
                except:
                    continue
        
        return resultados
    
    except Exception as e:
        print(f"  Error: {e}")
        return []

def generar_fechas():
    """
    Genera lista de fechas (jueves) desde 2013 hasta 2024
    """
    # Billboard actualiza los jueves
    fecha_fin = datetime(2024, 12, 26)   # Última semana completa de 2024
    fecha_inicio = datetime(2013, 1, 3)   # Primera semana de 2013
    
    fechas = []
    fecha_actual = fecha_fin
    
    while fecha_actual >= fecha_inicio:
        fechas.append(fecha_actual.strftime('%Y-%m-%d'))
        fecha_actual -= timedelta(days=7)
    
    return fechas

# ============ EJECUCIÓN PRINCIPAL ============
print("=" * 60)
print("EXTRACCIÓN BILLBOARD HOT 100 (2013 - 2024)")
print("=" * 60)

# Generar fechas
fechas = generar_fechas()
print(f"\n Semanas a procesar: {len(fechas)}")

# Variable para almacenar todos los datos
todos_los_datos = []

print("\n Iniciando extracción...")
print("-" * 60)

# Procesar semana por semana
for i, fecha in enumerate(fechas):
    print(f" Semana {i+1}/{len(fechas)}: {fecha}")
    
    datos_semana = extraer_semana(fecha)
    
    if datos_semana:
        todos_los_datos.extend(datos_semana)
        print(f"  {len(datos_semana)} canciones extraídas")
    else:
        print(f" No se encontraron datos")
    
    # Pausa para no bloquear el servidor
    time.sleep(0.5)

# Guardar resultados
if todos_los_datos:
    with open('billboard_colaboraciones.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['fecha', 'posicion', 'cancion', 'artista', 'es_colaboracion'])
        writer.writeheader()
        writer.writerows(todos_los_datos)
    
    print("\n" + "=" * 60)
    print("EXTRACCIÓN COMPLETADA")
    print("=" * 60)
    print(f" Total de registros guardados: {len(todos_los_datos)}")
    print(f" Archivo: billboard_colaboraciones.csv")
    
    # Mostrar vista previa
    print("\n📋 VISTA PREVIA (primeras 10 canciones):")
    print("-" * 65)
    for row in todos_los_datos[:10]:
        colab = "COLABORACIÓN" if row['es_colaboracion'] else "SOLISTA"
        print(f"#{row['posicion']:3} | {row['cancion'][:35]:35} | {row['artista'][:25]:25} | {colab}")
else:
    print("\n No se encontraron datos. Revisa tu conexión a internet.")
    print("   Si el problema persiste, Billboard puede haber cambiado su estructura.")
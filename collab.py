import billboard
import pandas as pd
import time
from datetime import datetime
import re

def es_colaboracion(artista):
    # artista se refiere a los artistas de la canción
    # Detecta si hay feat, &, and, with, o coma en el nombre del artista
    patrones = [r'[Ff]eat', r'&', r' and ', r' [Ww]ith ', r',']
    for patron in patrones:
        if re.search(patron, artista):
            return 1
    return 0

# Extraer datos desde 2013
print("Extrayendo datos...")
fecha_limite = datetime(2013, 1, 5)
chart = billboard.ChartData('hot-100')
datos = []

while chart and chart.previousDate:
    fecha_chart = datetime.strptime(chart.date, '%Y-%m-%d')
    if fecha_chart < fecha_limite:
        break
    
    for cancion in chart:
        datos.append({
            'fecha': chart.date,
            'posicion': cancion.rank,
            'cancion': cancion.title,
            'artista': cancion.artist,
            'es_colaboracion': es_colaboracion(cancion.artist)
        })
    
    chart = billboard.ChartData('hot-100', chart.previousDate)
    time.sleep(0.5)

# Guardar en Excel
df = pd.DataFrame(datos)
df.to_excel('billboard_colaboraciones.xlsx', index=False)
print(f" Listo! {len(datos)} canciones guardadas en 'billboard_colaboraciones.xlsx'")
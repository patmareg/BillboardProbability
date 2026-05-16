import csv
from datetime import datetime
from collections import defaultdict
import re
import pandas as pd
import json
print("=" * 60)
print("EXTRACCION BILLBOARD HOT 100 (2013-2024)")
print("=" * 60)

print("Cargando archivo billboard_hot_100.csv...")
df = pd.read_csv('billboard_hot_100.csv')

df = df.rename(columns={
    'date': 'fecha',
    'rank': 'posicion',
    'song': 'titulo',
    'artist': 'artista'
})
df['fecha'] = pd.to_datetime(df['fecha'])
df = df[(df['fecha'] >= '2013-01-01') & (df['fecha'] <= '2024-12-31')]
df = df.sort_values('fecha')

print(f"Registros cargados: {len(df)}")
print(f"Rango de fechas: {df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}")
print("-" * 60)
# Procesar por grupos de semanas
fechas_unicas = df['fecha'].dt.strftime('%Y-%m-%d').unique()
canciones_por_semana = defaultdict(list)
for idx, row in df.iterrows():
    fecha_str = row['fecha'].strftime('%Y-%m-%d')
    canciones_por_semana[fecha_str].append(row)

print(f"Procesando {len(fechas_unicas)} semanas...")

canciones = list()
v_realx = dict() # fechas donde está definida
v_realy = dict() #puestos que toma
for fecha_semana in fechas_unicas:
    for row in canciones_por_semana[fecha_semana]:
        titulo = row['titulo']
        if titulo in canciones:
            v_realx[titulo].append(fecha_semana)
            v_realy[titulo].append(row['posicion'])
        else:
            canciones.append(titulo)
            v_realx[titulo] = [fecha_semana]
            v_realy[titulo] = [row['posicion']]
datos={'v_realx': v_realx, 'v_realy': v_realy}
with open("resultados.json", "w") as f:
    json.dump(datos, f, indent=4)
    

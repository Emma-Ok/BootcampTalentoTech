import pandas as pd
import urllib.request
import tempfile
import os

# URL del dataset
url_csv = "https://raw.githubusercontent.com/Emma-Ok/BootcampTalentoTech/main/beneficiarios.csv"

print(f"Descargando archivo desde: {url_csv}")

# Descargar y guardar el archivo temporalmente
with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
    urllib.request.urlretrieve(url_csv, temp_file.name)
    temp_path = temp_file.name

print(f"Archivo descargado temporalmente en: {temp_path}")

# Examinar las primeras líneas del archivo para detectar problemas
print("\nExaminando las primeras 15 líneas del archivo:")
with open(temp_path, 'r', encoding='utf-8', errors='replace') as file:
    lines = [file.readline() for _ in range(15)]
    for i, line in enumerate(lines):
        print(f"Línea {i+1}: {repr(line)}")

# Intentar leer con diferentes configuraciones
print("\nIntentando cargar con diferentes configuraciones:")

configs = [
    {"sep": ",", "engine": "python"},
    {"sep": ";", "engine": "python"},
    {"sep": "\t", "engine": "python"},
    {"sep": None, "engine": "python"},
    {"sep": ",", "engine": "c", "on_bad_lines": "skip"},
]

for i, config in enumerate(configs):
    try:
        print(f"\nIntento {i+1} con configuración: {config}")
        df = pd.read_csv(temp_path, **config)
        print(f"¡Éxito! Dimensiones del DataFrame: {df.shape}")
        print("Primeras 5 filas:")
        print(df.head())
        print("\nInformación de columnas:")
        print(df.dtypes)
        break
    except Exception as e:
        print(f"Error: {e}")

# Limpiar archivo temporal
os.unlink(temp_path)
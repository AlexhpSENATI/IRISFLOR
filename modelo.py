import pandas as pd

# Intenta leer el archivo CSV
try:
    df = pd.read_csv('iris.csv')  # Asegúrate de que el archivo esté en el mismo directorio
    if df.empty:
        print("El DataFrame está vacío. Asegúrate de que el archivo contenga datos.")
    else:
        print("Datos leídos correctamente:")
        print(df.head())  # Muestra las primeras filas del DataFrame
except FileNotFoundError:
    print("El archivo 'iris.csv' no se encontró.")
except pd.errors.EmptyDataError:
    print("El archivo 'iris.csv' está vacío.")
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
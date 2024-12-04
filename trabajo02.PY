# clasificador_iris.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar el dataset de Iris desde un archivo CSV
try:
    df = pd.read_csv('iris.csv')  # Asegúrate de que el archivo esté en el mismo directorio
except FileNotFoundError:
    print("El archivo 'iris.csv' no se encontró. Asegúrate de que esté en el directorio correcto.")
    exit()
except pd.errors.EmptyDataError:
    print("El archivo 'iris.csv' está vacío. Asegúrate de que contenga datos.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
    exit()

# Paso 2: Verificar que el DataFrame no esté vacío
if df.empty:
    print("El DataFrame está vacío. Asegúrate de que el archivo contenga datos.")
    exit()

# Paso 3: Separar las características y la etiqueta
X = df.drop('species', axis=1)  # Características
y = df['species']  # Etiquetas (especies)

# Paso 4: Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 5: Crear el clasificador de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Paso 6: Entrenar el modelo
clf.fit(X_train, y_train)

# Paso 7: Hacer predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Paso 8: Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# Mostrar el reporte de clasificación
print(classification_report(y_test, y_pred))

# Mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=df['species'].unique(), yticklabels=df['species'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión')
plt.show()

# Paso 9: Visualizar el árbol de decisión (opcional)
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=df['species'].unique())
plt.title('Árbol de Decisión para Clasificación de Iris')
plt.show()
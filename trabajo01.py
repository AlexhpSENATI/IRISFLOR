# clasificador_iris.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar el dataset de Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas (especies)

# Paso 2: Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 3: Crear el clasificador de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Paso 4: Entrenar el modelo
clf.fit(X_train, y_train)

# Paso 5: Hacer predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Paso 6: Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

# Mostrar el reporte de clasificación
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión')
plt.show()

# Paso 7: Visualizar el árbol de decisión (opcional)
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title('Árbol de Decisión para Clasificación de Iris')
plt.show()
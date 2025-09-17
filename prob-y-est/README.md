# Probabilidad y Estadística

Esta carpeta contiene scripts y proyectos relacionados con la materia de Probabilidad y Estadística.

## Contenido

### `red-neuronal-breast-cancer.py`

Este script es una guía práctica que compara dos enfoques de clasificación para el dataset de cáncer de mama de Wisconsin:

1.  **K-Nearest Neighbors (KNN)**: Un modelo basado en instancias.
2.  **Perceptrón Multicapa (MLP)**: Una red neuronal.

- **Objetivo**: Clasificar si un tumor es benigno o maligno.
- **Librerías**: Utiliza `scikit-learn`, `pandas`, `numpy`, `matplotlib` y `seaborn`.

El código está diseñado como un tutorial y explica en detalle los siguientes conceptos:
- La correcta división de datos en entrenamiento, validación y prueba.
- El escalado de variables y cómo evitar la fuga de información (`data leakage`).
- La selección del hiperparámetro `k` en KNN mediante validación cruzada.
- El entrenamiento de un MLP "por épocas" simuladas.
- La implementación de *early stopping* manual para prevenir el sobreajuste.
- La evaluación de clasificadores con métricas como F1-score, ROC-AUC y PR-AUC.
- La visualización de la arquitectura de la red neuronal entrenada.

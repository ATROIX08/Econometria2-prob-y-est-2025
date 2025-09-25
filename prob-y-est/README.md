# Probabilidad y Estadística

Esta carpeta contiene scripts y proyectos relacionados con la materia de Probabilidad y Estadística.

## Contenido

### 1. Comparativa de Modelos: KNN vs. Red Neuronal (`red-neuronal-breast-cancer.py`)

Este script realiza una comparación entre dos modelos de clasificación para el dataset de cáncer de mama de Wisconsin: **K-Nearest Neighbors (KNN)** y un **Perceptrón Multicapa (MLP)**.

- **Objetivo**: Clasificar si un tumor es benigno o maligno y comparar el rendimiento de dos enfoques diferentes.
- **Librerías**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.
- **Conceptos clave**:
    - Selección de hiperparámetros (k en KNN) mediante validación cruzada.
    - Entrenamiento de un MLP con "early stopping" manual para prevenir el sobreajuste.
    - Evaluación exhaustiva de clasificadores (F1-score, ROC-AUC, PR-AUC).
    - Visualización de la arquitectura de la red neuronal.

### 2. XGBoost para Detección de Cáncer de Mama (`xgboost-cancer.py`)

Este script implementa un modelo de **XGBoost** para el mismo problema de clasificación del dataset de cáncer de mama.

- **Objetivo**: Utilizar XGBoost para un problema de clasificación binaria y visualizar los árboles de decisión.
- **Librerías**: `xgboost`, `pandas`, `matplotlib`, `seaborn`.
- **Conceptos clave**:
    - Preparación de datos para XGBoost en un problema binario.
    - Entrenamiento y evaluación de un modelo de XGBoost.
    - Interpretación del modelo a través de la visualización de sus árboles.

## Cómo Ejecutar los Scripts

1. Asegúrate de tener instaladas las librerías del `requirements.txt` del repositorio principal.
2. Navega a la carpeta `src` de esta materia:
   ```bash
   cd prob-y-est/src
   ```
3. Ejecuta el script que desees:
   ```bash
   python red-neuronal-breast-cancer.py
   ```
   o
   ```bash
   python xgboost-cancer.py
   ```

## Visualizaciones Generadas

El script `xgboost-cancer.py` genera visualizaciones de los árboles de decisión. Aquí tienes una muestra:

![Árbol de XGBoost](images/arbol_xgboost_1.png)
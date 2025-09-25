# -*- coding: utf-8 -*-
# XGBoost binario para 'Breast Cancer' (0=maligno, 1=benigno)
# Solo train/test (SIN validación), y solo estos outputs:
# 1) Vista previa del dataset
# 2) Matriz de confusión (gráfica)
# 3) Reporte de clasificación
# 4) Gráficos de árboles específicos


# --- Importación de Librerías ---
# pandas: para manejar los datos en un formato tabular (DataFrame)
import pandas as pd
# xgboost: la librería que contiene el algoritmo que vamos a usar
import xgboost as xgb
# matplotlib y seaborn: para crear visualizaciones (gráficos)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Módulos específicos de Scikit-learn ---
# load_breast_cancer: un dataset de ejemplo clásico para clasificación binaria
from sklearn.datasets import load_breast_cancer
# train_test_split: una función para dividir nuestro dataset en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
# classification_report y confusion_matrix: métricas para evaluar el rendimiento del modelo
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# 1) Cargar dataset y vista previa
# =============================================================================
# Cargamos el conjunto de datos sobre cáncer de mama.
# Este dataset ya viene pre-cargado en la librería scikit-learn.
cancer = load_breast_cancer()

# Convertimos los datos en un DataFrame de pandas para facilitar su manipulación.
# 'cancer.data' contiene las características (features) y 'cancer.feature_names' sus nombres.
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Añadimos la columna objetivo ('target') al DataFrame.
# En este dataset, 0 significa 'maligno' y 1 significa 'benigno'.
df["target"] = cancer.target
target_names = ["maligno", "benigno"] # Guardamos los nombres para los reportes

print("Vista previa del dataset:")
# .head() muestra las primeras 5 filas para que veamos cómo se ven los datos.
print(df.head())

# =============================================================================
# 2) Partición: SOLO train / test (20% test, estratificado)
# =============================================================================
# Separamos las características (X) de la variable objetivo (y).
# X: todas las columnas excepto 'target'.
# y: solo la columna 'target'.
X = df.drop(columns=["target"])
y = df["target"].astype(int)

# Dividimos los datos en un conjunto de entrenamiento (train) y uno de prueba (test).
# El modelo aprenderá de los datos de 'train' y lo evaluaremos con los datos de 'test'.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,    # El 30% de los datos se usará para el conjunto de prueba.
    stratify=y,       # 'stratify=y' asegura que la proporción de 0s y 1s sea la misma en train y test.
                      # Es muy importante en clasificación para evitar desbalances.
    random_state= 42  # Fija una "semilla" para que la división sea siempre la misma si ejecutamos el código de nuevo.
)

# =============================================================================
# 3) DMatrix y parámetros XGBoost
# =============================================================================
# XGBoost tiene una estructura de datos interna llamada DMatrix.
# Es una optimización que le permite manejar los datos de forma muy eficiente.
# Creamos una DMatrix para el entrenamiento (con datos y etiquetas) y otra para el test (solo datos).
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test)

# Aquí definimos los "hiperparámetros" del modelo.
# Estos son los ajustes que controlan cómo aprenderá el algoritmo.
# Más abajo se explican en detalle.
params = {
    # OBJETIVO Y MÉTRICA
    "objective": "binary:logistic", # Tarea de clasificación binaria. La salida será una probabilidad.
    "eval_metric": "logloss",       # Métrica para evaluar el rendimiento: "pérdida logística".

    # PARÁMETROS DE CONTROL DEL MODELO (Regularización)
    "eta": 0.05,                    # Tasa de aprendizaje (learning rate).
    "max_depth": 15,                 # Profundidad máxima de cada árbol.
    "subsample": 0.50,              # Porcentaje de filas a usar por cada árbol.
    "colsample_bytree": 0.90,       # Porcentaje de columnas (features) a usar por cada árbol.

    # PARÁMETROS TÉCNICOS
    "tree_method": "hist",          # Algoritmo para construir los árboles (rápido y eficiente).
    "seed": 42,                     # Silencia los mensajes durante el entrenamiento.
}

# =============================================================================
# 4) Entrenamiento del modelo
# =============================================================================
# La función xgb.train() es la que construye el modelo.
# Le pasamos los parámetros definidos, los datos de entrenamiento y el número de "rondas".
booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=15000  # Número de árboles que se construirán secuencialmente.
)

# =============================================================================
# 5) Predicción en el conjunto de TEST
# =============================================================================
# Usamos el modelo entrenado ('booster') para predecir sobre los datos de test ('dtest').
# ¡IMPORTANTE! La salida no es 0 o 1 directamente, es una probabilidad entre 0 y 1.
# Por ejemplo: [0.01, 0.98, 0.45, ...], donde 0.01 es una alta probabilidad de ser clase 0 (maligno)
# y 0.98 es una alta probabilidad de ser clase 1 (benigno).
y_proba = booster.predict(dtest)

# Para obtener una clase final (0 o 1), aplicamos un umbral.
# El umbral estándar es 0.5: si la probabilidad es >= 0.5, lo clasificamos como 1 (benigno),
# si no, como 0 (maligno).
y_pred = (y_proba >= 0.5).astype(int)

# =============================================================================
# 6) Evaluación: Matriz de confusión y Reporte de clasificación
# =============================================================================
# La matriz de confusión nos dice cuántas veces el modelo acertó y cuántas falló, y de qué forma.
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Usamos seaborn y matplotlib para visualizar la matriz de confusión de forma clara.
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Pred {t}" for t in target_names],
            yticklabels=[f"Real {t}" for t in target_names])
plt.title("Matriz de Confusión (TEST)")
plt.ylabel("Verdadero")
plt.xlabel("Predicho")
plt.tight_layout()
plt.show()

# El reporte de clasificación nos da métricas más detalladas como precisión, recall y f1-score.
print("Reporte de Clasificación (TEST):")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


# 7) Gráfico de Árboles de Decisión Específicos

# Índices de los árboles que queremos visualizar (recuerda que empiezan en 0)
indices_arboles = [0, 4, 499, 14999] # Corresponden al 1ro, 5to, 500vo y 15000vo

# Iteramos sobre cada índice para crear un gráfico por cada árbol
for i in indices_arboles:
    try:
        # Creamos una figura y un eje para el gráfico con un tamaño grande
        fig, ax = plt.subplots(figsize=(30, 15))
        
        # Graficamos el árbol específico en el eje que creamos
        xgb.plot_tree(booster, tree_idx=i, ax=ax)
        
        # Añadimos un título claro
        titulo = f"Visualización del Árbol número {i+1}"
        ax.set_title(titulo, fontsize=20)
        
        # Guardamos la figura en un archivo PNG de alta calidad (300 DPI)
        # El archivo se guardará en la carpeta 'images' que está al mismo nivel que 'src'.
        import os
        output_dir = "../images"
        os.makedirs(output_dir, exist_ok=True)
        nombre_archivo = os.path.join(output_dir, f"arbol_xgboost_{i+1}.png")
        print(f"Guardando el gráfico del árbol {i+1} en '{nombre_archivo}'...")
        plt.savefig(nombre_archivo, dpi=500, bbox_inches='tight')
        
        # Mostramos una vista previa en pantalla (seguirá viéndose pequeña/borrosa aquí)
        plt.show()
        
        # Cerramos la figura para liberar memoria antes de crear la siguiente
        plt.close(fig)

    except Exception as e:
        print(f"No se pudo graficar el árbol {i+1}. Posible error: {e}")
        # Esto puede pasar si el modelo no tiene tantos árboles como se piden.

print("\nProceso de visualización de árboles finalizado.")
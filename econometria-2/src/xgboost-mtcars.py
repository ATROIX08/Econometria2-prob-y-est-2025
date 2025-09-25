# -*- coding: utf-8 -*-
# XGBoost multiclase para predecir 'cyl' (cilindros) en el dataset 'mtcars'
# Solo train/test (SIN validación), y solo estos outputs:
# 1) Vista previa del dataset
# 2) Matriz de confusión (gráfica)
# 3) Reporte de clasificación
# 4) Gráficos de árboles específicos

# --- Importación de Librerías ---
# numpy: para operaciones numéricas, especialmente útil aquí para seleccionar tipos de datos
import numpy as np
# pandas: para manejar los datos en un formato tabular (DataFrame)
import pandas as pd
# xgboost: la librería que contiene el algoritmo que vamos a usar
import xgboost as xgb
# matplotlib y seaborn: para crear visualizaciones (gráficos)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Módulos específicos de Scikit-learn ---
# train_test_split: una función para dividir nuestro dataset en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
# métricas para evaluar el rendimiento del modelo en tareas de clasificación
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================================================================
# 1) Cargar dataset y vista previa
# =============================================================================
# Cargamos el conjunto de datos 'mtcars' desde una URL pública.
# Este es un dataset clásico que contiene datos técnicos de diferentes modelos de coches.
url = "https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"
df = pd.read_csv(url)

# La primera columna del archivo CSV no tiene nombre, pandas la llama 'Unnamed: 0'.
# La renombramos a 'model' para que sea más descriptivo.
df = df.rename(columns={'Unnamed: 0': 'model'})

print("Vista previa del dataset mtcars:")
# .head() muestra las primeras 5 filas para que veamos cómo se ven los datos.
print(df.head(), "\n")

# =============================================================================
# 2) Preparación de datos (X e y)
# =============================================================================
# --- Seleccionar las características (X) ---
# Primero, identificamos todas las columnas que son numéricas.
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Luego, creamos nuestra lista de características (features) excluyendo la variable que queremos predecir ('cyl').
features = [c for c in num_cols if c != "cyl"]
X = df[features].copy() # Creamos una copia para evitar advertencias de pandas

# --- Preparar la variable objetivo (y) ---
# La variable objetivo 'cyl' tiene valores como 4, 6 y 8.
# Los algoritmos de Machine Learning, incluido XGBoost, requieren que las etiquetas de clase
# sean números enteros consecutivos que comiencen en 0 (es decir, 0, 1, 2...).
# Por lo tanto, necesitamos "mapear" nuestros valores originales a este nuevo formato.
y_raw = df["cyl"].astype(int)

# Paso 1: Obtener las clases únicas y ordenarlas. El resultado será [4, 6, 8].
clases = sorted(y_raw.unique())
# Paso 2: Crear un diccionario de mapeo. El resultado será {4: 0, 6: 1, 8: 2}.
mapping = {clase_original: indice for indice, clase_original in enumerate(clases)}
# Paso 3: Aplicar el mapeo a nuestra columna 'cyl' para obtener la variable objetivo final.
y = y_raw.map(mapping).astype(int)

# Guardamos los nombres de las clases para usarlos en los gráficos y reportes.
target_names = [f"{c} cyl" for c in clases] # ["4 cyl", "6 cyl", "8 cyl"]
# Guardamos el número de clases, que es un parámetro necesario para XGBoost en modo multiclase.
num_class = len(clases)

print("Predictoras (características) usadas:", features, "\n")
print(f"Mapeo de clases: {mapping}")

# =============================================================================
# 3) Partición: SOLO train / test (30% test, estratificado)
# =============================================================================
# Dividimos los datos en un conjunto de entrenamiento (train) y uno de prueba (test).
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,   # El 30% de los datos se usará para el conjunto de prueba.
    stratify=y,       # 'stratify=y' asegura que la proporción de cada clase (4, 6, 8 cilindros)
                      # sea la misma en train y test. Es crucial en clasificación.
    random_state= 42  # Fija una "semilla" para que la división sea siempre la misma.
)

# =============================================================================
# 4) DMatrix y parámetros XGBoost (configuración multiclase)
# =============================================================================
# XGBoost tiene una estructura de datos interna llamada DMatrix para máxima eficiencia.
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test)

# Definimos los hiperparámetros del modelo.
params = {
    # OBJETIVO Y MÉTRICA
    "objective": "multi:softprob", # Tarea de clasificación multiclase. La salida será un vector de probabilidades.
    "num_class": num_class,        # ¡MUY IMPORTANTE! Debemos decirle a XGBoost cuántas clases hay.
    "eval_metric": ["mlogloss", "merror"], # Métricas a monitorear: pérdida logarítmica y tasa de error multiclase.

    # PARÁMETROS DE CONTROL DEL MODELO (Regularización)
    "eta": 0.1,                     # Tasa de aprendizaje (learning rate).
    "max_depth": 4,                 # Profundidad máxima de cada árbol. Más bajo para evitar sobreajuste.
    "subsample": 0.9,               # Porcentaje de filas a usar por cada árbol.
    "colsample_bytree": 0.9,        # Porcentaje de columnas (features) a usar por cada árbol.

    # PARÁMETROS TÉCNICOS
    "tree_method": "hist",          # Algoritmo eficiente para construir los árboles.
    "seed": 42,                     # Semilla para reproducibilidad.
    "verbosity": 0                  # Silencia los mensajes durante el entrenamiento.
}

# =============================================================================
# 5) Entrenamiento del modelo
# =============================================================================
# La función xgb.train() construye el modelo con los parámetros y datos definidos.
booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=300  # Número de árboles que se construirán secuencialmente.
)

# =============================================================================
# 6) Predicción en el conjunto de TEST
# =============================================================================
# Usamos el modelo entrenado ('booster') para predecir sobre los datos de test ('dtest').
# ¡IMPORTANTE! Para "multi:softprob", la salida es una matriz donde cada fila contiene
# las probabilidades para cada clase. Por ejemplo, para 3 clases: [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], ...]
proba_test = booster.predict(dtest)

# Para obtener la clase final, buscamos el índice de la probabilidad más alta en cada fila.
# El método .argmax(axis=1) hace exactamente eso. El índice (0, 1 o 2) corresponde a nuestra clase predicha.
y_pred = proba_test.argmax(axis=1)

# =============================================================================
# 7) Evaluación: Métricas y Reporte de clasificación
# =============================================================================
# El 'accuracy' es una métrica simple: (aciertos totales) / (total de predicciones).
acc = accuracy_score(y_test, y_pred)
print("Accuracy (TEST):", f"{acc:.4f}\n")

# El reporte de clasificación nos da métricas detalladas (precisión, recall, f1-score) para cada clase.
print("Reporte de Clasificación (TEST):")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# =============================================================================
# 8) Evaluación: Matriz de confusión (gráfica)
# =============================================================================
# La matriz de confusión nos dice cuántas veces el modelo acertó y falló para cada clase.
# Las filas son los valores reales, las columnas son los predichos.
cm = confusion_matrix(y_test, y_pred, labels=list(range(num_class)))

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


# =============================================================================
# 9) Gráfico de Árboles de Decisión Específicos
# =============================================================================
# Índices de los árboles que queremos visualizar (recuerda que empiezan en 0).
# Hemos entrenado 300 árboles (num_boost_round=300).
indices_arboles = [0, 10, 99, 299] # Corresponden al 1ro, 11vo, 100vo y 300vo.

# Iteramos sobre cada índice para crear y guardar un gráfico por cada árbol.
for i in indices_arboles:
    try:
        # Creamos una figura y un eje para el gráfico con un tamaño grande para mayor detalle.
        fig, ax = plt.subplots(figsize=(30, 15))
        
        # Graficamos el árbol específico usando la función plot_tree de XGBoost.
        xgb.plot_tree(booster, num_trees=i, ax=ax)
        
        # Añadimos un título claro y grande para identificar cada gráfico.
        titulo = f"Visualización del Árbol de XGBoost número {i+1} (mtcars)"
        ax.set_title(titulo, fontsize=20)
        
        # Guardamos la figura en un archivo PNG de alta calidad (300 DPI).
        # Esto permite hacer zoom para ver los detalles de las divisiones en cada nodo.
        # La imagen se guardará en la carpeta 'images' que está al mismo nivel que 'src'.
        import os
        output_dir = "../images"
        os.makedirs(output_dir, exist_ok=True)
        nombre_archivo = os.path.join(output_dir, f"arbol_xgboost_mtcars_{i+1}.png")
        print(f"Guardando el gráfico del árbol {i+1} en '{nombre_archivo}'...")
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        
        # Mostramos una vista previa en pantalla (suele verse pequeña aquí).
        # Lo importante es el archivo guardado.
        plt.show()
        
        # Cerramos la figura para liberar memoria antes de crear la siguiente.
        plt.close(fig)

    except Exception as e:
        # Capturamos posibles errores, por ejemplo, si pedimos un árbol que no existe.
        print(f"No se pudo graficar el árbol {i+1}. Posible error: {e}")

print("\nProceso de visualización de árboles finalizado.")
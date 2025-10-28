# Entrenamiento Final — Modelo Óptimo

**Autor:** Humberto Silva Baltazar · **Curso:** Econometría II + Probabilidad y Estadística  
**Script:** [`train_best.py`](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/src/modeling/train_best.py)  
**Entrada:** `diamonds_features_20251012_155956.parquet` + receta óptima

## 1. El Modelo: `HistGradientBoostingRegressor`
Antes de detallar el proceso de entrenamiento, es fundamental entender el motor del modelo seleccionado: el `HistGradientBoostingRegressor` de Scikit-learn.

### Concepto Clave
Es un modelo de ensamble basado en **árboles de decisión** que pertenece a la familia de *Gradient Boosting*. Su objetivo es construir un predictor fuerte a partir de la combinación secuencial de muchos predictores débiles (árboles pequeños). La palabra clave "Hist" (de histograma) se refiere a su principal ventaja técnica: antes de entrenar, **discretiza las variables continuas en un número fijo de *bins*** (por ejemplo, 255). Esto acelera drásticamente el proceso de búsqueda de los mejores puntos de corte en los árboles, resultando en un entrenamiento mucho más rápido y con menor consumo de memoria que su contraparte clásica, `GradientBoostingRegressor`. Esta técnica está inspirada en implementaciones de alto rendimiento como LightGBM.

### ¿Cómo Funciona?
El proceso de entrenamiento sigue una lógica iterativa para minimizar el error progresivamente:
1.  **Discretización (Binning):** Convierte cada variable numérica en un índice entero que representa su *bin* o intervalo. Esto simplifica y acelera los cálculos posteriores.
2.  **Modelo Inicial:** Comienza con una predicción simple, usualmente el promedio del valor objetivo (el precio de los diamantes).
3.  **Cálculo de Errores (Gradientes):** Calcula el error (residual) de cada predicción del modelo actual.
4.  **Entrenamiento Secuencial:** Entrena un nuevo árbol de decisión, no para predecir el precio directamente, sino para **predecir los errores** del paso anterior. El objetivo de este nuevo árbol es corregir lo que el ensamble actual no pudo modelar bien.
5.  **Actualización Ponderada:** Añade el nuevo árbol al ensamble, pero su contribución es ponderada por un factor llamado `learning_rate`. Un `learning_rate` bajo hace que el modelo aprenda más lentamente, lo que a menudo mejora su capacidad de generalización.
6.  **Repetición y Parada Temprana:** Repite los pasos 3 a 5 hasta alcanzar un número máximo de árboles (`n_estimators`) o hasta que el error en un conjunto de validación deja de mejorar, gracias al mecanismo de `early_stopping`.

El resultado es un modelo final robusto, no lineal, capaz de capturar interacciones complejas entre variables y efectos de umbral sin necesidad de escalar los datos previamente.

### Ejemplo Conceptual en Código
El siguiente fragmento ilustra cómo se instancia y entrena este modelo en Scikit-learn, reflejando los principios de parada temprana y la definición de hiperparámetros clave.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Dividimos los datos para tener un conjunto de validación implícito para early stopping
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciamos el modelo con hiperparámetros optimizados
hgb = HistGradientBoostingRegressor(
    loss="squared_error",      # Objetivo: minimizar el error cuadrático medio
    learning_rate=0.1,         # Tasa de aprendizaje para las actualizaciones
    max_leaf_nodes=63,         # Controla la complejidad de cada árbol
    min_samples_leaf=20,       # Suaviza las predicciones, previene el sobreajuste
    l2_regularization=0.1,     # Penalización para evitar pesos extremos en las hojas
    max_bins=255,              # Número de bins para la discretización
    n_estimators=500,          # Número máximo de árboles (en la práctica, early_stopping decide)
    early_stopping=True,       # Activa la parada temprana para evitar sobreajuste
    validation_fraction=0.1,   # Proporción de datos de entrenamiento para validar en cada iteración
    random_state=42
)

# El modelo se entrena en los datos de entrenamiento
hgb.fit(X_train, y_train)

# Se evalúa su rendimiento en datos no vistos
pred = hgb.predict(X_valid)
print(f"Rendimiento del modelo (R²): {hgb.score(X_valid, y_valid):.4f}")
```

## 2. Objetivo y Pipeline
El objetivo de esta ejecución es entrenar el modelo final `HistGradientBoostingRegressor` utilizando la configuración de hiperparámetros óptima, previamente identificada mediante experimentación y búsqueda en malla (GridSearch). El proceso abarca la carga de datos procesados, una selección de variables definitiva, el entrenamiento del modelo, su evaluación rigurosa sobre un conjunto de prueba no visto (*out-of-sample*) y, finalmente, la persistencia de los artefactos generados (pipeline serializado, métricas, gráficas de diagnóstico y un resumen textual) para facilitar su uso en inferencia y garantizar la trazabilidad del experimento.

## 3. Configuración, Receta y Variables
El modelo se entrenó sobre un conjunto de datos preprocesado, del cual se excluyeron 20 registros con dimensiones inválidas y se eliminaron variables con potencial de fuga de información (*leakage*) o alta redundancia. El conjunto final se dividió en 80% para entrenamiento (43.136 muestras) y 20% para prueba (10.784 muestras), utilizando una estratificación por cuantiles de precio para asegurar una distribución representativa en ambos subconjuntos.

### Variables Seleccionadas para el Modelo
Se seleccionaron un total de 20 variables, de las cuales 16 son numéricas y 4 son categóricas. Esta selección busca maximizar el poder predictivo evitando la multicolinealidad.

*   **Variables Fundamentales (4Cs):**
    *   `carat`: Peso del diamante, el factor más influyente.
    *   `fe_cut_ord`, `fe_color_ord`, `fe_clarity_ord`: Calidad del corte, color y claridad, transformadas a una escala numérica ordinal.
*   **Variables de Ingeniería de Atributos (Feature Engineering):**
    *   `fe_carat_x_quality`: Interacción entre los quilates y un puntaje de calidad global. Es crucial para capturar cómo el impacto del tamaño varía según la calidad.
    *   `fe_log_carat`: Transformación logarítmica de los quilates para modelar relaciones no lineales.
    *   `fe_quality_score`: Un puntaje agregado que combina corte, color y claridad.
    *   `fe_spread_per_carat`, `fe_area_per_carat`: Ratios que describen qué tan "grande" se ve el diamante en relación con su peso.
    *   `fe_symmetry_dev_pct`, `fe_z_to_spread_ratio`: Métricas que capturan la simetría y la proporción de la geometría del diamante.
    *   `fe_carat_bin`: Variable categórica que agrupa los diamantes por rangos de quilates.
*   **Variables Geométricas y de Consistencia:**
    *   `depth`, `table`: Porcentaje de profundidad y ancho de la tabla superior.
    *   `fe_depth_pct_is_consistent`, `fe_is_square`: Variables binarias que indican si las proporciones son consistentes o si la forma es cuadrada.

### Hiperparámetros
El modelo `HistGradientBoostingRegressor` se instanció con los siguientes hiperparámetros, que demostraron ser los de mejor rendimiento en fases anteriores.

**Tabla J — Hiperparámetros finales**
| Parámetro | Valor |
|---|---|
| `learning_rate` | 0.1 |
| `max_depth` | 10 |
| `max_leaf_nodes` | 63 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 0.1 |
| `max_bins` | 255 |
| `early_stopping` | True |

## 4. Métricas en Conjunto de Prueba
La evaluación sobre el conjunto de prueba, que el modelo no observó durante el entrenamiento, arroja un desempeño sobresaliente. Con un **R² de 0.9818**, el modelo explica casi el 98.2% de la variabilidad en el precio de los diamantes. El Error Absoluto Medio (MAE) indica que, en promedio, las predicciones se desvían en $278,19 del precio real, mientras que el Error Porcentual Absoluto Medio (MAPE) de 8.03% confirma una alta precisión relativa.

**Tabla K — Desempeño en test**
| Métrica | Valor |
|---|---|
| MAE | 278,19 |
| RMSE | 538,92 |
| R² | 0,9818 |
| MAPE | 8,03% |
| Correlación (Obs vs Pred) | 0,9909 |

## 5. Figuras de Evaluación e Interpretación
![Observado vs Predicho (best)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/observado_vs_predicho_best_20251012_202645.png)  
**Interpretación:** La gráfica de dispersión de precios observados frente a predichos muestra una concentración muy alta de los puntos a lo largo de la línea de identidad (y=x). La línea de tendencia OLS se superpone casi perfectamente, lo que visualmente confirma la altísima correlación (0.99) y el excelente ajuste del modelo. Las predicciones son consistentes a lo largo de todo el rango de precios, con muy pocos valores atípicos que se desvían significativamente, validando la capacidad de generalización del modelo.

![Histograma de residuales (best)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/residuales_hist_best_20251012_202645.png)  
**Interpretación:** El histograma de los residuales (errores de predicción) se centra casi perfectamente en cero (media = 0.40), cumpliendo con un supuesto clave de un modelo bien calibrado. La distribución se asemeja a una campana de Gauss, aunque con una curtosis más alta (leptocúrtica), lo que indica que la mayoría de los errores son muy pequeños y se agrupan cerca de la media, con algunas desviaciones más grandes en las colas. Este comportamiento es ideal, pues refleja un modelo que es preciso para la gran mayoría de los casos.

![Residuales vs Predicho (best)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/residuales_vs_predicho_best_20251012_202645.png)  
**Interpretación:** Este gráfico analiza si el error del modelo varía sistemáticamente con el nivel del precio predicho. Idealmente, los puntos deberían formar una nube aleatoria sin patrones alrededor de la línea horizontal en cero. Si bien la dispersión es mayoritariamente aleatoria, la "tendencia binned" (línea verde) sugiere un patrón muy sutil en forma de "sonrisa", indicando una leve tendencia a subestimar los precios en los extremos (muy bajos y muy altos) y a sobreestimarlos ligeramente en el rango medio. No obstante, la magnitud de esta desviación es muy pequeña y no compromete la utilidad del modelo.

![Permutation Importance (Top-20)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/permutation_importance_best_top20_20251012_202645.png)  
**Interpretación:** La importancia por permutación revela qué variables son más críticas para la precisión del modelo. De manera contundente, la variable de ingeniería `fe_carat_x_quality` (interacción entre quilates y un score de calidad) es la más influyente, seguida de cerca por `carat`. Esto subraya el éxito del feature engineering. Las variables ordinales que representan claridad (`fe_clarity_ord`), corte (`fe_cut_ord`) y color (`fe_color_ord`) siguen en importancia.

## 6. Importancia de Variables (Top-10)
La técnica de *Permutation Importance* mide la caída en el rendimiento del modelo cuando se permuta aleatoriamente una variable, aislando su contribución predictiva. Los resultados confirman el dominio del peso y la calidad.

**Tabla L — Importancias (Top-10)**
| Variable | Importancia (Disminución media de R²) |
|---|---:|
| fe_carat_x_quality | 0,6369 |
| carat | 0,5263 |
| fe_clarity_ord | 0,0685 |
| fe_cut_ord | 0,0367 |
| fe_color_ord | 0,0101 |
| fe_spread_per_carat | 0,0060 |
| fe_carat_bin | 0,0050 |
| table | 0,0007 |
| fe_area_per_carat | 0,0006 |
| fe_quality_score | 0,0006 |

## 7. Robustez y Sensibilidades
El modelo demuestra una alta robustez. Las métricas obtenidas en el conjunto de prueba son consistentes y sólidas, lo que sugiere que el modelo generaliza bien a datos no vistos. La inclusión de regularización L2 (`l2_regularization=0.1`) y el uso de `early_stopping` son mecanismos efectivos implementados para mitigar el sobreajuste. La leve heterocedasticidad observada en el gráfico de residuales vs. predichos es un punto a monitorear, pero dada la magnitud del R² y el bajo MAPE, su impacto práctico es mínimo. El modelo es sensible principalmente a los hiperparámetros que controlan la complejidad del árbol (`max_depth`, `max_leaf_nodes`), cuya calibración fue crucial en la fase de optimización.

## 8. Artefactos y Trazabilidad
Para asegurar la reproducibilidad y el despliegue, se generaron y guardaron los siguientes artefactos:
-   **Pipeline del modelo:** `output/model/best_hgbr_pipeline_20251012_202645.joblib`. Este archivo contiene el pipeline completo, incluyendo el preprocesador y el regresor entrenado, listo para ser cargado y utilizado para hacer predicciones.
-   **Figuras de evaluación:** Guardadas en `plots/modeling/`, incluyendo `observado_vs_predicho_best_...`, `residuales_hist_best_...`, `residuales_vs_predicho_best_...`, y `permutation_importance_best_...`.
-   **Logs y Métricas:** Un log detallado, un JSON con las métricas finales y un resumen en TXT fueron almacenados en `output/logs/modeling/`.
-   **Semilla aleatoria:** Se utilizó una semilla fija (`SEED = 42`) en el split de datos y en el modelo para garantizar la reproducibilidad exacta de los resultados. La trazabilidad de las versiones de librerías (ej. scikit-learn, pandas) sería un paso adicional para un entorno de producción.

## 9. Evidencia (extracto del log)
```
2025-10-12 20:26:44 | INFO     | train_best | Total columnas seleccionadas para X: 20
2025-10-12 20:26:44 | INFO     | train_best | Numéricas (16): ['carat', 'fe_log_carat', 'fe_cut_ord', 'fe_color_ord', 'fe_clarity_ord', 'fe_quality_score', 'fe_carat_x_quality', 'depth', 'table', 'fe_depth_pct_is_consistent', 'fe_invalid_dims', 'fe_symmetry_dev_pct', 'fe_z_to_spread_ratio', 'fe_spread_per_carat', 'fe_area_per_carat', 'fe_is_square']
2025-10-12 20:26:44 | INFO     | train_best | Categóricas (4): ['cut', 'color', 'clarity', 'fe_carat_bin']
2025-10-12 20:26:44 | INFO     | train_best | Split realizado. Train: 43.136 filas; Test: 10.784 filas.
2025-10-12 20:26:44 | INFO     | train_best | Entrenando HistGradientBoostingRegressor con hiperparámetros óptimos...
2025-10-12 20:26:45 | INFO     | train_best | Entrenamiento finalizado.
2025-10-12 20:26:45 | INFO     | train_best | =========================================
2025-10-12 20:26:45 | INFO     | train_best |         MÉTRICAS EN CONJUNTO TEST        
2025-10-12 20:26:45 | INFO     | train_best | =========================================
2025-10-12 20:26:45 | INFO     | train_best | MSE  : 290.437,0350
2025-10-12 20:26:45 | INFO     | train_best | RMSE : 538,9221
2025-10-12 20:26:45 | INFO     | train_best | MAE  : 278,1900
2025-10-12 20:26:45 | INFO     | train_best | R^2  : 0.981846
2025-10-12 20:26:45 | INFO     | train_best | MAPE : 8.032%
2025-10-12 20:26:45 | INFO     | train_best | Corr : 0.990886
2025-10-12 20:26:45 | INFO     | train_best | Métricas guardadas en JSON: ...\metrics_best_20251012_202645.json
2025-10-12 20:26:56 | INFO     | train_best | Pipeline entrenado guardado en: ...\best_hgbr_pipeline_20251012_202645.joblib
2025-10-12 20:26:56 | INFO     | train_best |    Ejecución finalizada sin errores.     
```
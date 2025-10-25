# Entrenamiento Final — Modelo Óptimo

**Autor:** Humberto Silva Baltazar · **Curso:** Econometría II + Probabilidad y Estadística  
**Script:** [`train_best.py`](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/src/modeling/train_best.py)  
**Entrada:** `diamonds_features_20251012_155956.parquet` + receta óptima 

## 1. Objetivo y Pipeline
El objetivo de esta ejecución es entrenar el modelo final `HistGradientBoostingRegressor` utilizando la configuración de hiperparámetros óptima, previamente identificada mediante experimentación y búsqueda en malla (GridSearch). El proceso abarca la carga de datos procesados, una selección de variables definitiva, el entrenamiento del modelo, su evaluación rigurosa sobre un conjunto de prueba no visto (*out-of-sample*) y, finalmente, la persistencia de los artefactos generados (pipeline serializado, métricas, gráficas de diagnóstico y un resumen textual) para facilitar su uso en inferencia y garantizar la trazabilidad del experimento.

## 2. Configuración y Receta
El modelo se entrenó sobre un conjunto de datos preprocesado, del cual se excluyeron 20 registros con dimensiones inválidas y se eliminaron variables con potencial de fuga de información (*leakage*) o alta redundancia. El conjunto final se dividió en 80% para entrenamiento (43.136 muestras) y 20% para prueba (10.784 muestras), utilizando una estratificación por cuantiles de precio para asegurar una distribución representativa en ambos subconjuntos.

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

## 3. Métricas en Conjunto de Prueba
La evaluación sobre el conjunto de prueba, que el modelo no observó durante el entrenamiento, arroja un desempeño sobresaliente. Con un **R² de 0.9818**, el modelo explica casi el 98.2% de la variabilidad en el precio de los diamantes. El Error Absoluto Medio (MAE) indica que, en promedio, las predicciones se desvían en $278,19 del precio real, mientras que el Error Porcentual Absoluto Medio (MAPE) de 8.03% confirma una alta precisión relativa.

**Tabla K — Desempeño en test**
| Métrica | Valor |
|---|---|
| MAE | 278,19 |
| RMSE | 538,92 |
| R² | 0,9818 |
| MAPE | 8,03% |
| Correlación (Obs vs Pred) | 0,9909 |

## 4. Figuras de Evaluación e Interpretación
![Observado vs Predicho (best)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/observado_vs_predicho_best_20251012_202645.png)  
**Interpretación:** La gráfica de dispersión de precios observados frente a predichos muestra una concentración muy alta de los puntos a lo largo de la línea de identidad (y=x). La línea de tendencia OLS se superpone casi perfectamente, lo que visualmente confirma la altísima correlación (0.99) y el excelente ajuste del modelo. Las predicciones son consistentes a lo largo de todo el rango de precios, con muy pocos valores atípicos que se desvían significativamente, validando la capacidad de generalización del modelo.

![Histograma de residuales (best)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/residuales_hist_best_20251012_202645.png)  
**Interpretación:** El histograma de los residuales (errores de predicción) se centra casi perfectamente en cero (media = 0.40), cumpliendo con un supuesto clave de un modelo bien calibrado. La distribución se asemeja a una campana de Gauss, aunque con una curtosis más alta (leptocúrtica), lo que indica que la mayoría de los errores son muy pequeños y se agrupan cerca de la media, con algunas desviaciones más grandes en las colas. Este comportamiento es ideal, pues refleja un modelo que es preciso para la gran mayoría de los casos.

![Residuales vs Predicho (best)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/residuales_vs_predicho_best_20251012_202645.png)  
**Interpretación:** Este gráfico analiza si el error del modelo varía sistemáticamente con el nivel del precio predicho. Idealmente, los puntos deberían formar una nube aleatoria sin patrones alrededor de la línea horizontal en cero. Si bien la dispersión es mayoritariamente aleatoria, la "tendencia binned" (línea verde) sugiere un patrón muy sutil en forma de "sonrisa", indicando una leve tendencia a subestimar los precios en los extremos (muy bajos y muy altos) y a sobreestimarlos ligeramente en el rango medio. No obstante, la magnitud de esta desviación es muy pequeña y no compromete la utilidad del modelo.

![Permutation Importance (Top-20)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/permutation_importance_best_top20_20251012_202645.png)  
**Interpretación:** La importancia por permutación revela qué variables son más críticas para la precisión del modelo. De manera contundente, la variable de ingeniería `fe_carat_x_quality` (interacción entre quilates y un score de calidad) es la más influyente, seguida de cerca por `carat`. Esto subraya el éxito del feature engineering. Las variables ordinales que representan claridad (`fe_clarity_ord`), corte (`fe_cut_ord`) y color (`fe_color_ord`) siguen en importancia.

## 5. Importancia de Variables (Top-10)
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

## 6. Robustez y Sensibilidades
El modelo demuestra una alta robustez. Las métricas obtenidas en el conjunto de prueba son consistentes y sólidas, lo que sugiere que el modelo generaliza bien a datos no vistos. La inclusión de regularización L2 (`l2_regularization=0.1`) y el uso de `early_stopping` son mecanismos efectivos implementados para mitigar el sobreajuste. La leve heterocedasticidad observada en el gráfico de residuales vs. predichos es un punto a monitorear, pero dada la magnitud del R² y el bajo MAPE, su impacto práctico es mínimo. El modelo es sensible principalmente a los hiperparámetros que controlan la complejidad del árbol (`max_depth`, `max_leaf_nodes`), cuya calibración fue crucial en la fase de optimización.

## 7. Artefactos y Trazabilidad
Para asegurar la reproducibilidad y el despliegue, se generaron y guardaron los siguientes artefactos:
-   **Pipeline del modelo:** `output/model/best_hgbr_pipeline_20251012_202645.joblib`. Este archivo contiene el pipeline completo, incluyendo el preprocesador y el regresor entrenado, listo para ser cargado y utilizado para hacer predicciones.
-   **Figuras de evaluación:** Guardadas en `plots/modeling/`, incluyendo `observado_vs_predicho_best_...`, `residuales_hist_best_...`, `residuales_vs_predicho_best_...`, y `permutation_importance_best_...`.
-   **Logs y Métricas:** Un log detallado, un JSON con las métricas finales y un resumen en TXT fueron almacenados en `output/logs/modeling/`.
-   **Semilla aleatoria:** Se utilizó una semilla fija (`SEED = 42`) en el split de datos y en el modelo para garantizar la reproducibilidad exacta de los resultados. La trazabilidad de las versiones de librerías (ej. scikit-learn, pandas) sería un paso adicional para un entorno de producción.

## 8. Evidencia (extracto del log)
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

## 9. Implicaciones prácticas
El modelo entrenado constituye una herramienta de valoración de diamantes altamente precisa y fiable. Su RMSE de aproximadamente $539 proporciona un umbral claro de confianza para sus estimaciones. Podría ser desplegado en una aplicación web o API para que joyeros o consumidores obtengan tasaciones instantáneas.

Para su mantenimiento en un entorno productivo, se recomienda establecer un sistema de monitoreo que rastree la distribución de las variables de entrada y la precisión de las predicciones a lo largo del tiempo. Si se detecta una degradación del rendimiento (*model drift*), por ejemplo, debido a cambios en las tendencias del mercado o en las prácticas de certificación de diamantes, el modelo debería ser re-entrenado con datos recientes. Un ciclo de re-entrenamiento semestral o anual podría ser un punto de partida razonable, ajustado según los resultados del monitoreo.
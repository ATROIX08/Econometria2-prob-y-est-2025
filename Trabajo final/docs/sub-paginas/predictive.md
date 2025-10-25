# Modelo Predictivo — HistGradientBoostingRegressor (Búsqueda)

**Autor:** Humberto Silva Baltazar · **Curso:** Econometría II + Probabilidad y Estadística  
**Script:** [`predictive.py`](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/src/modeling/predictive.py)  
**Entrada:** `diamonds_features_20251012_155956.parquet` 

## 1. Objetivo y Estrategia

El propósito de esta fase del modelado es identificar la combinación óptima de hiperparámetros para un `HistGradientBoostingRegressor`. Esta búsqueda exhaustiva tiene como fin maximizar la capacidad predictiva del modelo para estimar el precio de los diamantes, minimizando el error de generalización. Para lograrlo, se implementa una estrategia de validación cruzada (`GridSearchCV`) sobre un espacio de búsqueda predefinido, evaluando sistemáticamente el rendimiento de cada configuración.

La principal innovación de esta etapa es la paralelización del proceso de búsqueda mediante el framework Dask y su librería `dask-ml`. Esta aproximación permite distribuir el costo computacional de entrenar y validar cientos de modelos a través de múltiples hilos de ejecución, reduciendo drásticamente el tiempo total del experimento sin sacrificar el rigor metodológico. La métrica de optimización seleccionada es el error cuadrático medio negativo de la raíz (`neg_root_mean_squared_error`), que orienta la búsqueda hacia modelos con el menor RMSE promedio en los pliegues de validación.

## 2. Espacio de Búsqueda (Grid/Params)

Se definió un espacio de búsqueda multidimensional para explorar los hiperparámetros clave que controlan el equilibrio entre sesgo y varianza del modelo. Se incluyeron parámetros que regulan la estructura del árbol (complejidad), la tasa de aprendizaje (velocidad de convergencia) y la regularización (control del sobreajuste).

**Tabla H — Hiperparámetros explorados**
| Parámetro | Valores evaluados | Descripción |
|---|---|---|
| `learning_rate` | `[0.01, 0.05, 0.1]` | Tasa a la que se corrigen los errores de los árboles anteriores. |
| `max_depth` | `[None, 6, 10, 16]` | Profundidad máxima de cada árbol. `None` implica sin límite. |
| `max_leaf_nodes`| `[31, 63, 127]` | Número máximo de nodos hoja, controla la complejidad. |
| `min_samples_leaf`| `[20, 50, 100]` | Mínimo de muestras requeridas para formar un nodo hoja. |
| `l2_regularization`| `[0.0, 0.1, 1.0]` | Parámetro de regularización L2 para reducir el sobreajuste. |

## 3. Configuración de Validación y Paralelización

La robustez de los resultados se garantiza mediante un riguroso esquema de validación y una configuración de paralelización eficiente:

- **Esquema de CV:** Se empleó `KFold` con **5 pliegues (`cv=5`)**. Los datos de entrenamiento se dividieron en cinco subconjuntos. En cada iteración, uno se usó para validación y los cuatro restantes para entrenamiento. Para asegurar la representatividad de la variable objetivo (`price`) en cada pliegue, el *split* inicial entre entrenamiento (80%) y prueba (20%) se realizó de forma estratificada sobre cuantiles del precio.
- **Configuración Dask:** La búsqueda se ejecutó sobre un clúster local (`LocalCluster`) configurado con **6 workers basados en hilos (`processes=False`)**. Esta configuración es ideal para tareas limitadas por el GIL de Python, como las que involucran operaciones de NumPy y Scikit-Learn, ya que evita la sobrecarga de la creación de procesos. El uso de Dask permitió evaluar las 324 combinaciones de hiperparámetros en aproximadamente 2 horas, un tiempo significativamente menor al que requeriría una ejecución secuencial.

## 4. Resultados de la Búsqueda (Top-k)

La búsqueda exhaustiva identificó varias configuraciones de alto rendimiento. La tabla siguiente resume las tres mejores combinaciones encontradas, ordenadas por su RMSE promedio durante la validación cruzada.

**Tabla I — Mejores combinaciones y métricas (CV)**
| Rank | Parámetros | Métrica RMSE (media±std) |
|---:|---|---|
| 1 | `{'l2_reg': 0.1, 'lr': 0.1, 'max_depth': 10, 'max_leaf': 63, 'min_samples': 20}` | **537.7307 ± 19.0202** |
| 2 | `{'l2_reg': 1.0, 'lr': 0.1, 'max_depth': 10, 'max_leaf': 63, 'min_samples': 20}` | 537.8046 ± 19.4709 |
| 3 | `{'l2_reg': 0.1, 'lr': 0.1, 'max_depth': None, 'max_leaf': 63, 'min_samples': 20}`| 538.8978 ± 17.5862 |

*Nota: `lr` es `learning_rate`, `l2_reg` es `l2_regularization` y `max_leaf` es `max_leaf_nodes`.*

## 5. Discusión

El análisis de las mejores combinaciones revela patrones consistentes y permite extraer conclusiones sobre la sensibilidad del modelo a ciertos hiperparámetros:

- **Parámetros Dominantes:** Los hiperparámetros `learning_rate=0.1`, `max_leaf_nodes=63` y `min_samples_leaf=20` aparecen de forma constante en las configuraciones de mayor rendimiento. Esto sugiere que una tasa de aprendizaje moderadamente alta, combinada con una complejidad de árbol controlada (63 nodos hoja) y un requisito mínimo de 20 muestras por hoja, constituye un punto de partida robusto para este problema.

- **Efecto de la Complejidad y Regularización:** El valor óptimo para `max_depth` fue 10. Curiosamente, la tercera mejor configuración usó `max_depth=None`, pero los resultados fueron muy similares. Esto indica que la complejidad ya está efectivamente limitada por `max_leaf_nodes=63`, haciendo que `max_depth` sea un factor secundario una vez que es suficientemente grande. La regularización L2, aunque presente en el mejor modelo (`l2_regularization=0.1`), no parece ser un factor crítico, ya que valores de 0.0 y 1.0 también produjeron resultados competitivos. Su inclusión, sin embargo, es una buena práctica para prevenir el sobreajuste.

- **Estabilidad del Rendimiento:** La desviación estándar de los scores de CV (entre $17 y $19) es baja en relación con la magnitud del RMSE promedio (~$538). Esto indica que el rendimiento del modelo es estable y consistente a través de los diferentes pliegues de datos, lo que aumenta la confianza en que los resultados de la validación cruzada se generalizarán bien a datos no vistos. La cercanía entre los scores del Top-5 sugiere que el óptimo se encuentra en una región relativamente plana del espacio de hiperparámetros, donde pequeñas variaciones no degradan drásticamente el rendimiento.

## 6. Evidencia (extracto del log)
```
2025-10-12 17:16:08 | INFO     | predictive | Iniciando GridSearchCV (dask-ml) sobre el cluster Dask...
2025-10-12 19:16:42 | INFO     | predictive | GridSearchCV finalizado en 7,234.6 segundos.
2025-10-12 19:16:42 | INFO     | predictive | ==========================================
2025-10-12 19:16:42 | INFO     | predictive |       MEJOR MODELO (GridSearchCV)        
2025-10-12 19:16:42 | INFO     | predictive | ==========================================
2025-10-12 19:16:42 | INFO     | predictive | Mejor RMSE (CV): 537.7307
2025-10-12 19:16:42 | INFO     | predictive | Mejores hiperparámetros: {
  "regressor__early_stopping": true,
  "regressor__l2_regularization": 0.1,
  "regressor__learning_rate": 0.1,
  "regressor__max_bins": 255,
  "regressor__max_depth": 10,
  "regressor__max_leaf_nodes": 63,
  "regressor__min_samples_leaf": 20
}
2025-10-12 19:16:42 | INFO     | predictive | Top-5 combinaciones (por RMSE CV):
2025-10-12 19:16:42 | INFO     | predictive |   #1: RMSE=537.7307 ± 19.0202 | params={'regressor__early_stopping': True, 'regressor__l2_regularization': 0.1, 'regressor__learning_rate': 0.1, 'regressor__max_bins': 255, 'regressor__max_depth': 10, 'regressor__max_leaf_nodes': 63, 'regressor__min_samples_leaf': 20}
2025-10-12 19:16:42 | INFO     | predictive |   #2: RMSE=537.8046 ± 19.4709 | params={'regressor__early_stopping': True, 'regressor__l2_regularization': 1.0, 'regressor__learning_rate': 0.1, 'regressor__max_bins': 255, 'regressor__max_depth': 10, 'regressor__max_leaf_nodes': 63, 'regressor__min_samples_leaf': 20}
2025-10-12 19:16:42 | INFO     | predictive | ==========================================
2025-10-12 19:16:42 | INFO     | predictive |         MÉTRICAS EN CONJUNTO TEST         
2025-10-12 19:16:42 | INFO     | predictive | ==========================================
2025-10-12 19:16:42 | INFO     | predictive | MSE  : 290,437.0350
2025-10-12 19:16:42 | INFO     | predictive | RMSE : 538.9221
2025-10-12 19:16:42 | INFO     | predictive | MAE  : 278.1900
2025-10-12 19:16:42 | INFO     | predictive | R^2  : 0.981846
2025-10-12 19:16:42 | INFO     | predictive | MAPE : 8.032%
```

## 7. Implicaciones para Entrenamiento Final

La búsqueda sistemática y paralelizada ha culminado con la identificación de una receta de hiperparámetros robusta y de alto rendimiento. El modelo óptimo seleccionado, con un RMSE de validación cruzada de $537.73, demostró un excelente desempeño en el conjunto de prueba, obteniendo un R² de 0.9818 y un MAPE de 8.03%.

Esta configuración se considera validada y será la base para el siguiente paso: el entrenamiento del modelo final. El script `train_best.py` utilizará estos hiperparámetros para entrenar un único `HistGradientBoostingRegressor` sobre la totalidad de los datos de entrenamiento, con el objetivo de consolidar un modelo final listo para producción o inferencia.
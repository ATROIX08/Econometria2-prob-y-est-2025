# ETL

**Autor:** Humberto Silva Baltazar · **Curso:** Econometría II + Probabilidad y Estadística  
**Script:** [`etl_diamonds.py`](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/src/etl_diamonds.py)  
**Entrada:** `diamonds.csv` · **Salida:** `diamonds.parquet`  

## 1. Objetivo y Alcance
El objetivo principal de este proceso de Extracción, Transformación y Carga (ETL) es preparar el conjunto de datos `diamonds.csv` para análisis posteriores, incluyendo el Análisis Exploratorio de Datos (EDA) y el desarrollo de modelos econométricos. El alcance del script abarca varias etapas clave para asegurar la calidad y utilidad de los datos.

Primero, se busca realizar una carga robusta de los datos, seguida de una normalización de esquema que incluye la estandarización de los nombres de las columnas (a formato `snake_case`) y la tipificación correcta de cada variable según su naturaleza (numérica, categórica). Segundo, se implementa un sistema de validación para identificar anomalías como valores nulos, filas duplicadas y datos que violan restricciones lógicas o de dominio (e.g., dimensiones físicas no positivas). Finalmente, el proceso se enfoca en enriquecer el dataset mediante ingeniería de características, creando nuevas variables que capturan relaciones geométricas, económicas y de calidad, con el fin de facilitar la construcción de modelos más robustos y con mayor poder explicativo.

## 2. Insumos, Esquema y Trazabilidad
El proceso utiliza como único insumo el archivo `diamonds.csv`. Las características del dataset de entrada se resumen a continuación.

**Tabla A — Resumen del dataset de entrada**
| Métrica | Valor |
|---|---|
| Registros (filas) | 53,940 |
| Columnas | 11 |
| Memoria aprox. | 4,013,204 bytes |
| Origen | `diamonds.csv` |

**Variables clave de entrada:** `price`, `carat`, `cut`, `color`, `clarity`, `x`, `y`, `z`, `depth`, `table`.

El script está diseñado para mantener una trazabilidad completa. La columna `index` del archivo original se conserva para permitir el rastreo de cualquier registro a su origen. Cada nueva característica generada se prefija con `fe_` para distinguirla claramente de las columnas originales.

## 3. Reglas de Limpieza y Validación
Se aplicó un conjunto de reglas para garantizar la integridad y consistencia de los datos. El proceso no elimina filas, sino que identifica y marca los problemas para que puedan ser gestionados en etapas posteriores.

- **Normalización y Tipificación:** Convertir nombres de columnas a minúsculas y `snake_case`. Forzar el tipo de dato de cada columna a su definición esperada (e.g., `price` a entero, `carat` a flotante, `cut` a string).
- **Tratamiento de Nulos y Duplicados:** Identificar valores nulos (interpretando cadenas como `NA`, `None`, etc.) y filas completamente duplicadas. En esta ejecución, no se encontraron registros de este tipo.
- **Validación de Dominio:** Verificar que los valores de las variables categóricas (`cut`, `color`, `clarity`) pertenezcan a sus dominios predefinidos. No se encontraron violaciones.
- **Validación de Rango Físico:** Identificar y marcar registros con dimensiones no factibles, específicamente aquellos donde `x`, `y`, o `z` son menores o iguales a cero, ya que representan medidas físicas que deben ser positivas.

**Criterios de exclusión/marcaje:** El script no excluye registros. Las filas con dimensiones no positivas (`x`, `y`, `z ≤ 0`) son marcadas a través de la característica booleana `fe_invalid_dims`, permitiendo su posterior análisis o filtrado.

**Tabla B — Controles de calidad aplicados**
| Chequeo | Resultado / Conteo |
|---|---|
| Duplicados (filas) | 0 |
| Nulos totales | 0 |
| `x≤0` / `y≤0` / `z≤0` | 8 / 7 / 20 |
| Rango `price` (≤0) | 0 |
| Rango `carat` (≤0) | 0 |

## 4. Ingeniería de Características (objetivo y definición)
Para aumentar el valor analítico del dataset, se generó un extenso conjunto de nuevas características. El objetivo detrás de estas transformaciones es exponer relaciones no lineales, normalizar distribuciones y combinar información para crear predictores más potentes.

- **Variables Geométricas y de Proporción:**
    - Calcular el **volumen** aproximado del diamante como `fe_volume_mm3 = x * y * z`. Esta variable es fundamental para entender la densidad y puede estar altamente correlacionada con el precio.
    - Crear ratios como `fe_aspect_ratio` (`x/y`) y `fe_symmetry_dev_pct` para cuantificar la forma y simetría del diamante, que son factores de calidad.
    - Recalcular el porcentaje de profundidad (`fe_depth_pct_recalc`) a partir de las dimensiones `x, y, z` para contrastarlo con el valor reportado en `depth` y así evaluar la consistencia de los datos.

- **Variables Económicas y de Valor:**
    - Calcular el **precio por quilate** (`fe_price_per_carat = price / carat`) como una medida de valor normalizada por el peso, permitiendo comparaciones más justas entre diamantes de diferentes tamaños.
    - Aplicar **transformaciones logarítmicas** (`fe_log_price`, `fe_log_carat`) para estabilizar la varianza y linealizar relaciones que a menudo son exponenciales, una práctica común en modelado econométrico para cumplir con los supuestos de los modelos lineales.

- **Índices y Puntuaciones Compuestas:**
    - Construir un **índice de calidad** (`fe_quality_score`) para resumir las tres "C" categóricas (`cut`, `color`, `clarity`) en una única métrica numérica. Se basa en una codificación ordinal de cada categoría y una ponderación heurística que asigna mayor peso al corte. La definición es: `fe_quality_score = (0.5 * fe_cut_ord) + (0.3 * fe_color_ord) + (0.2 * fe_clarity_ord)`.

## 5. Resultado del ETL y Esquema de Salida
El proceso concluye con la generación de un único archivo en formato Apache Parquet, el cual incluye todas las columnas originales y las características de ingeniería. Este formato es eficiente en almacenamiento y rápido en lectura, ideal para flujos de trabajo de ciencia de datos.

- **Archivo de Salida:** `diamonds_features_20251012_155956.parquet`.
- **Columnas nuevas creadas:** `fe_cut_ord`, `fe_color_ord`, `fe_clarity_ord`, `fe_volume_mm3`, `fe_area_mm2`, `fe_spread_mm`, `fe_aspect_ratio`, `fe_invalid_dims`, `fe_symmetry_dev_pct`, `fe_depth_pct_recalc`, `fe_depth_pct_diff`, `fe_depth_pct_is_consistent`, `fe_z_to_spread_ratio`, `fe_price_per_carat`, `fe_log_price`, `fe_log_carat`, `fe_log_price_per_carat`, `fe_depth_dev`, `fe_table_dev`, `fe_table_to_depth_ratio`, `fe_spread_per_carat`, `fe_area_per_carat`, `fe_quality_score`, `fe_carat_bin`, `fe_carat_x_quality`, `fe_ppc_z_by_cqc`, `fe_is_square`.


## 6. Evidencia (extracto del log)
```log
2025-10-12 15:59:52,123 | INFO | Leyendo CSV desde: C:\...\data\diamonds.csv
2025-10-12 15:59:52,456 | INFO | Leído OK: 53,940 filas x 11 columnas
2025-10-12 15:59:52,458 | INFO | Tipificando columnas clave y normalizando nombres…
2025-10-12 15:59:53,812 | INFO | Aplicando ingeniería de características…
2025-10-12 15:59:54,987 | INFO | fe_invalid_dims (True) = 20
2025-10-12 15:59:55,011 | INFO | Escribiendo Parquet: C:\...\output\parquets\diamonds_features_20251012_155956.parquet
2025-10-12 15:59:55,234 | INFO | Parquet escrito con éxito.
2025-10-12 15:59:56,345 | INFO | ETL finalizado con éxito.
```

## 7    . Implicaciones para pasos siguientes
El dataset enriquecido generado por este ETL es la base fundamental para las siguientes etapas del proyecto.

- **Análisis Exploratorio de Datos (EDA):** Las nuevas variables, como `fe_price_per_carat` y `fe_quality_score`, permitirán un análisis más profundo de las relaciones entre las características del diamante y su valor. Las transformaciones logarítmicas facilitarán la visualización de tendencias lineales que de otro modo estarían ocultas.
- **Modelado:** El conjunto de datos está listo para ser utilizado en modelos de regresión para predecir el precio. Las variables de ingeniería pueden mejorar significativamente el rendimiento de los modelos al capturar interacciones y efectos no lineales. La variable `fe_invalid_dims` permite filtrar fácilmente los datos anómalos antes del entrenamiento, asegurando que el modelo aprenda de datos físicamente coherentes.
- **Eficiencia:** Al pre-calcular estas características y guardarlas en un formato eficiente como Parquet, se acelera el ciclo de experimentación en las fases de modelado, ya que no será necesario repetir estos cálculos en cada ejecución.
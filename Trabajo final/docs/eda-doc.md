# Análisis Exploratorio de Datos (EDA) del Dataset de Diamantes

## Introducción

El presente documento detalla los hallazgos del Análisis Exploratorio de Datos (EDA) realizado sobre el conjunto de datos de diamantes, previamente procesado y enriquecido mediante un script ETL. El análisis se efectúa sobre el archivo `diamonds_features_{RUN_TS}.parquet`, el cual contiene tanto las variables originales como un extenso conjunto de características ingenierizadas.

El objetivo de este EDA es **profundizar en la comprensión de los datos** para:
1.  **Validar** la calidad y consistencia de los datos y de las nuevas características.
2.  **Descubrir** patrones, distribuciones y relaciones entre las variables.
3.  **Identificar** anomalías, valores atípicos (outliers) y posibles problemas de multicolinealidad.
4.  **Generar** hipótesis que puedan guiar el futuro proceso de modelado predictivo del precio de los diamantes.

## 1. Resumen General y KPIs del Dataset

El análisis comienza con una visión general de la estructura y composición del dataset.

| KPI | Valor |
| :--- | :--- |
| **Registros** | 53,940 |
| **Columnas** | 38 |
| **Duplicados (filas)** | 0 |
| **Nulos totales** | 5 |
| **Memoria aprox.** | 25.00 MiB |

El dataset es considerablemente ancho, con 38 columnas, lo que refleja la gran cantidad de características generadas en el ETL. Es notable la ausencia de filas duplicadas y un número casi insignificante de valores nulos (5 en total), lo que indica una alta calidad de datos a nivel estructural.

## 2. Análisis Univariado: Distribución de Variables

Para entender el comportamiento de cada variable de forma individual, se generaron histogramas para las variables numéricas y gráficos de barras para las categóricas.

![Distribuciones por columna](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_per_column_distribution.png)

### Hallazgos Clave:
*   **Variables Numéricas Sesgadas**: Variables fundamentales como `price`, `carat`, y las dimensiones `x`, `y`, `z` presentan una clara **distribución sesgada hacia la derecha**. Esto significa que la mayoría de los diamantes se concentran en valores bajos (pequeños y económicos), con una larga cola de diamantes muy grandes y caros. Este sesgo es confirmado por el reporte de texto:
    *   `price`: skew=1.62
    *   `carat`: skew=1.12
    *   `y` y `z`: skew=2.43 y 1.52 respectivamente.
    Las transformaciones logarítmicas aplicadas (`fe_log_price`, `fe_log_carat`) muestran distribuciones mucho más simétricas (cercanas a una normal), lo que las hace más adecuadas para modelos lineales.

*   **Variables Categóricas**:
    *   `cut`: La calidad de corte más común es `Ideal` (casi el 40%), seguida de `Premium` y `Very Good`. `Fair`, el corte de menor calidad, es el menos frecuente.
    *   `color`: El color `G` es el más prevalente, mientras que los colores de mayor calidad (`D`, `E`) y menor calidad (`J`) son menos comunes.
    *   `clarity`: Las claridades más frecuentes son `SI1` y `VS2`, que se encuentran en el rango intermedio de la escala.

*   **Características Ingenierizadas**:
    *   `fe_invalid_dims` y `fe_depth_pct_is_consistent`: Estas variables booleanas actúan como banderas de calidad. Se observa que la gran mayoría de los registros tienen dimensiones válidas y una profundidad consistente.
    *   `fe_carat_bin`: La distribución en los rangos de quilates es relativamente balanceada para los segmentos `<0.5`, `0.5–1.0` y `1.0–1.5`, con menos diamantes en los rangos superiores.

## 3. Análisis de Calidad de Datos y Valores Atípicos (Outliers)

### 3.1. Valores Nulos
El análisis revela un total de 5 valores nulos, todos localizados en la columna `fe_ppc_z_by_cqc`.

![Valores nulos por columna](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_missing_values.png)

La presencia de nulos en `fe_ppc_z_by_cqc` (Z-score del precio por quilate por grupo) es esperable. Ocurre cuando un grupo definido por (`cut`, `color`, `clarity`) tiene muy pocos miembros (ej. uno solo), lo que hace que la desviación estándar sea cero y la división para calcular el Z-score sea indefinida. Estos 5 casos son marginales y no representan un problema de calidad.

### 3.2. Porcentaje de Outliers
Se utilizó el método del Rango Intercuartílico (IQR) para identificar el porcentaje de outliers en cada variable numérica.

![Porcentaje de outliers por columna](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_outlier_rates.png)

*   `price` es la variable con la mayor proporción de outliers (más del 6%). Esto es consistente con su distribución sesgada, donde los diamantes de lujo tienen precios que se desvían significativamente de la norma.
*   Variables de proporción como `fe_z_to_spread_ratio` y `fe_depth_pct_recalc` también muestran un alto porcentaje de outliers. Esto puede deberse a diamantes con formas inusuales o posibles errores de medición en `x`, `y`, `z`.
*   Las variables transformadas logarítmicamente (`fe_log_price`, `fe_log_carat`) tienen una tasa de outliers casi nula, demostrando su efectividad para normalizar las distribuciones.

### 3.3. Anomalías en Dimensiones Físicas
El reporte del ETL ya había identificado registros con dimensiones `x`, `y` o `z` iguales a cero. El gráfico 3D permite visualizar estas anomalías y la estructura general de los datos.

![Dimensiones físicas en 3D](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_3d_xyz.png)

La gran mayoría de los diamantes se agrupan en una "nube" densa y bien definida. Sin embargo, se pueden observar puntos aislados que se alejan drásticamente de este cúmulo. Estos corresponden a los outliers identificados:
*   Puntos en el origen (0,0,0), que son físicamente imposibles.
*   Un punto con un valor de `y` cercano a 60 mm y otro con un valor de `z` superior a 30 mm, ambos extremadamente anómalos en comparación con el resto del dataset.
Estos registros (20 en total según el reporte) deben ser tratados o eliminados antes del modelado, ya que representan errores de datos.

## 4. Análisis Bivariado: Relaciones entre Variables

### 4.1. Correlación entre Variables Numéricas
La matriz de correlación revela la fuerza y dirección de las relaciones lineales entre las variables numéricas.

![Matriz de correlación](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_correlation_matrix.png)

**Hallazgos Principales:**
*   **Fuerte Correlación Positiva**:
    *   Existe una correlación casi perfecta entre `price` y `carat` (0.922). De igual forma, `price` está fuertemente correlacionado con las dimensiones `x, y, z` y las características derivadas como `fe_volume_mm3` y `fe_area_mm2`. Esto confirma la intuición de que el tamaño es el principal impulsor del precio.
    *   Las variables `carat`, `x`, `y`, `z`, `fe_volume_mm3` están extremadamente correlacionadas entre sí, indicando que son medidas redundantes del tamaño del diamante.
*   **Relaciones con el Precio por Quilate**:
    *   `fe_price_per_carat` tiene una correlación positiva fuerte con `carat` (0.770). Esto es un hallazgo clave: no solo los diamantes más grandes son más caros en términos absolutos, sino que también son más caros *por unidad de peso*.
    *   La correlación entre `fe_price_per_carat` y el `fe_quality_score` es muy baja (0.013), sugiriendo que la relación entre calidad y precio no es lineal simple y puede depender de otros factores.

### 4.2. Matriz de Dispersión (Scatter Matrix)
Esta visualización permite examinar las relaciones dos a dos con mayor detalle, incluyendo la forma de las distribuciones en la diagonal.

![Matriz de dispersión](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_scatter_matrix.png)

*   **Relación No Lineal `price` vs. `carat`**: El gráfico de dispersión entre `price` y `carat` muestra una curva exponencial clara. El precio aumenta a una tasa creciente a medida que aumenta el peso.
*   **Efecto de la Transformación Logarítmica**: Al observar `fe_log_price` vs. `fe_log_carat`, la relación se vuelve mucho más lineal. Esto valida la utilidad de las características logarítmicas para simplificar esta relación y adecuarla a modelos lineales.
*   **Relaciones entre Dimensiones**: Los gráficos entre `x`, `y` y `z` muestran relaciones lineales muy fuertes, como era de esperar.

### 4.3. Análisis Detallado de Precio y Carat

![Precio vs. Carat](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_price_vs_carat.png)
![Precio por quilate vs. Carat](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_ppc_vs_carat.png)

*   El gráfico `Precio vs. Carat` confirma visualmente la relación exponencial.
*   El gráfico `Precio por quilate vs. Carat` es particularmente revelador. Muestra que el precio por quilate aumenta con el peso, pero con una estructura compleja. Se observan "bandas" o "clusters" de precios, y una alta variabilidad, especialmente para diamantes de alrededor de 1 a 2 quilates. Esto sugiere que otros factores (como calidad) influyen fuertemente en el precio por quilate.

## 5. Impacto de las Características Categóricas

Se utilizaron diagramas de caja para analizar cómo varía el `fe_price_per_carat` según las categorías de `cut`, `color`, `clarity` y `fe_carat_bin`.

| Gráfico de Cajas por `cut` | Gráfico de Cajas por `color` |
| :---: | :---: |
| ![PPC por Cut](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_ppc_by_cut.png) | ![PPC por Color](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_ppc_by_color.png) |

| Gráfico de Cajas por `clarity` | Gráfico de Cajas por `carat_bin` |
| :---: | :---: |
| ![PPC por Clarity](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_ppc_by_clarity.png) | ![PPC por Carat Bin](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_ppc_by_fe_carat_bin.png) |

**Hallazgos Clave:**
*   **Impacto de `cut`**: Sorprendentemente, la mediana del precio por quilate no sigue un orden estrictamente creciente con la calidad del corte. `Premium` y `Good` tienen medianas más altas que `Ideal`. Esto podría ser una paradoja causada por variables de confusión: es posible que los diamantes de menor calidad de corte tiendan a tener mejor color o claridad, o ser más grandes, inflando su precio por quilate.
*   **Impacto de `color` y `clarity`**: Aquí la tendencia es más clara. A medida que la calidad del color mejora (de `J` a `D`) y la claridad aumenta (de `I1` a `IF`), la mediana y la dispersión del precio por quilate tienden a subir.
*   **Impacto de `carat_bin`**: Confirma de manera contundente que el precio por quilate aumenta sistemáticamente con el tamaño del diamante. El salto de precio es especialmente pronunciado al pasar de un rango de quilates al siguiente.

## 6. Diagnóstico de Multicolinealidad (VIF)

El Factor de Inflación de la Varianza (VIF) se utiliza para medir qué tan bien una variable puede ser explicada por las otras. Un VIF alto (generalmente > 10) indica alta multicolinealidad.

| Variable | VIF |
| :--- | :--- |
| price | 81.009 |
| carat | 103.969 |
| fe_price_per_carat | 78.722 |
| fe_log_price | inf |
| fe_log_carat | inf |
| fe_log_price_per_carat | inf |
| x | 163.994 |
| y | 20.596 |
| z | 23.596 |

**Interpretación**:
Se detecta un **nivel extremo de multicolinealidad**.
*   Las variables relacionadas con el tamaño (`carat`, `x`, `y`, `z`) y el precio (`price`, `fe_price_per_carat`) tienen VIFs masivos. Esto es esperado, ya que son medidas altamente interdependientes.
*   Las variables transformadas logarítmicamente tienen VIF infinito, indicando que son combinaciones lineales casi perfectas de otras variables en el análisis (probablemente de sus contrapartes no transformadas).
*   **Implicación para el modelado**: Para construir un modelo de regresión interpretable (como una regresión lineal), es **imperativo** seleccionar un subconjunto de estas variables y eliminar las redundantes. Por ejemplo, se podría usar `carat` como la principal variable de tamaño y descartar `x`, `y`, `z`, y `fe_volume_mm3`.

## 7. Conclusiones y Próximos Pasos

Este Análisis Exploratorio de Datos ha revelado insights fundamentales sobre el dataset de diamantes:

1.  **Calidad de Datos**: El dataset es de alta calidad, con la excepción de un pequeño número de registros con dimensiones físicas imposibles (cero) que deben ser eliminados.
2.  **Importancia del Tamaño**: El `carat` es, con diferencia, el predictor más importante del precio, pero la relación no es lineal, sino exponencial. La transformación logarítmica es una herramienta eficaz para linealizar esta relación.
3.  **Valor Relativo**: El precio por quilate (`fe_price_per_carat`) no solo depende del tamaño, sino también de una interacción compleja entre `cut`, `color` y `clarity`.
4.  **Redundancia de Información**: Existe una severa multicolinealidad entre las variables de tamaño y precio. La selección de características será un paso crítico antes de entrenar modelos de regresión.
5.  **Potencial de Features Ingenierizadas**: Características como `fe_ppc_z_by_cqc` han demostrado ser útiles para identificar diamantes con precios atípicos en relación a sus pares, lo que podría ser una variable predictiva valiosa.

### Próximos Pasos Recomendados:
1.  **Limpieza Final**: Filtrar y eliminar los 20 registros con dimensiones `x`, `y`, o `z` iguales a cero.
2.  **Selección de Características**: Con base en el análisis VIF y la matriz de correlación, seleccionar un conjunto de predictores no redundantes. Por ejemplo: `carat` (o `fe_log_carat`), `cut`, `color`, `clarity`, y algunas de las interacciones o ratios más informativos.
3.  **Modelado**: Proceder con la construcción de modelos predictivos (ej. regresión lineal con las variables transformadas, o modelos más complejos como Gradient Boosting que pueden manejar interacciones automáticamente).
4.  **Interpretación**: Utilizar los modelos para cuantificar el impacto de cada característica en el precio y entender los impulsores de valor en el mercado de diamantes.
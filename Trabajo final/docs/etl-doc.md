¡Excelente! Aquí tienes una documentación exhaustiva en formato Markdown, diseñada para un `README.md` en GitHub. Sigue todas las directrices solicitadas: es extensa, se centra en la ingeniería de características, documenta el código y el output, y utiliza un lenguaje formal en tercera persona con verbos en infinitivo.

---

# Documentación del Proceso ETL para `diamonds.csv`

## 1. Resumen del Proceso

El presente documento tiene como objetivo describir en detalle el script de Python `etl_diamonds.py`, cuyo propósito es ejecutar un proceso de **Extracción, Transformación y Carga (ETL)** sobre el conjunto de datos `diamonds.csv`.

El flujo de trabajo se ha diseñado para ser robusto, reproducible y transparente. Las principales acciones a realizar son:

*   **Extraer**: Leer los datos desde un archivo CSV de origen.
*   **Transformar**:
    *   **Tipificar y limpiar**: Asegurar que cada columna tenga el tipo de dato correcto y normalizar los nombres de las columnas.
    *   **Validar**: Ejecutar una serie de chequeos de calidad de datos para identificar anomalías como valores nulos, duplicados, violaciones de dominio y valores fuera de rango.
    *   **Enriquecer**: Aplicar una fase extensiva de **ingeniería de características** para crear nuevas variables que puedan mejorar el rendimiento de modelos de machine learning o facilitar análisis más profundos.
*   **Cargar**:
    *   **Generar reportes**: Producir un informe tabular detallado en la terminal, resumiendo el estado del dataset antes y después de las transformaciones.
    *   **Persistir datos**: Guardar el conjunto de datos completo (datos originales + nuevas características) en un archivo formato **Parquet** optimizado para análisis.
    *   **Generar metadatos y logs**: Crear archivos de registro (log), un volcado del esquema en formato JSON y una copia del reporte de terminal para garantizar la trazabilidad y auditabilidad del proceso.

## 2. Estructura del Proyecto

Para la correcta ejecución del script, se debe mantener la siguiente estructura de directorios:

```
.
├── data/
│   └── diamonds.csv              # Archivo de entrada
├── output/
│   ├── logs/                     # Directorio para logs, reportes y schema JSON
│   └── parquets/                 # Directorio para el archivo Parquet de salida
└── src/
    └── etl_diamonds.py           # El script principal del ETL
```

## 3. Descripción Detallada del Código Fuente (`etl_diamonds.py`)

El script se organiza en secciones modulares para facilitar su comprensión y mantenimiento.

### 3.1. Configuración y Logging

*   **Configuración de Rutas**: Se definen las rutas de entrada y salida utilizando `pathlib.Path` para asegurar la compatibilidad entre sistemas operativos. Se crean automáticamente los directorios de salida si no existen.
*   **Sistema de Logging**: Se implementa un sistema de registro dual utilizando el módulo `logging`.
    *   Un `FileHandler` se encarga de escribir un registro detallado de cada paso del proceso en un archivo `.log` con timestamp. Esto es crucial para **depurar** y **auditar** ejecuciones pasadas.
    *   Un `StreamHandler` se encarga de imprimir los mismos mensajes de log en la consola (terminal), permitiendo **monitorear** el progreso en tiempo real.

### 3.2. Clases y Funciones de Utilidad

Se definen varias funciones auxiliares para mantener el código limpio y reutilizable.

*   **`ReportBuffer`**: Una clase diseñada para capturar todo el texto que se imprime en la consola. Su propósito es **almacenar** el reporte completo para luego guardarlo en un archivo `.txt`.
*   **Funciones de Formato**: Un conjunto de funciones (`sep_line`, `center_text`, `human_int`, `pct`, etc.) para **formatear** la salida tabular de manera legible y consistente.

### 3.3. Diccionario de Variables y Mapeos

*   **`variable_dictionary()`**: Una función que retorna un diccionario describiendo cada variable original del dataset. Su objetivo es **centralizar** el conocimiento del dominio y **servir** como base para la generación de reportes.
*   **Mapeos Ordinales**: Se definen diccionarios (`MAP_CUT_ORD`, `MAP_COLOR_ORD`, `MAP_CLARITY_ORD`) para **convertir** las variables categóricas ordinales (`cut`, `color`, `clarity`) en representaciones numéricas. Este paso es fundamental para muchos algoritmos de modelado que requieren entradas numéricas.

### 3.4. Fases del Proceso ETL

#### a. Carga y Tipificación (`load_csv_polars`, `coerce_schema`)

El proceso comienza con la carga de datos.
1.  **Leer CSV**: Se utiliza la librería `polars` por su alto rendimiento en el manejo de grandes volúmenes de datos. La función `pl.read_csv` se configura para inferir el esquema, manejar valores nulos comunes y especificar la codificación.
2.  **Normalizar Nombres**: La función `coerce_schema` se encarga de **limpiar** los nombres de las columnas (eliminar espacios, convertir a minúsculas) para facilitar su manipulación.
3.  **Tipificar Columnas**: Se realiza una conversión explícita (`cast`) de las columnas a los tipos de datos esperados (ej. `price` a `Int64`, `carat` a `Float64`). Esto previene errores en cálculos posteriores y optimiza el uso de memoria.

#### b. Validaciones y Calidad de Datos (`data_quality_checks`)

Esta función es clave para **diagnosticar** la salud del dataset. Se encarga de **calcular** un conjunto de métricas de calidad:
*   **Duplicados**: Contar el número de filas completamente idénticas.
*   **Nulos**: Contar valores nulos por cada columna y en total.
*   **Cardinalidad**: Contar el número de valores únicos por columna.
*   **Violaciones de Dominio**: Verificar si los valores de las variables categóricas (`cut`, `color`, `clarity`) se encuentran dentro de los dominios esperados predefinidos.
*   **Valores Fuera de Rango**: Identificar registros con valores numéricos que son lógicamente inválidos o sospechosos (ej. `price <= 0`, dimensiones `x, y, z <= 0`, o valores extremos en `depth` y `table`).

#### c. Ingeniería de Características (`feature_engineering`)

Esta es la sección más crítica del proceso de transformación. Su objetivo es **crear** un conjunto de nuevas variables a partir de las existentes para **extraer** información latente y **potenciar** la capacidad predictiva de futuros modelos. El prefijo `fe_` se utiliza para identificar claramente estas nuevas características.

**1. Codificación Ordinal:**
*   `fe_cut_ord`, `fe_color_ord`, `fe_clarity_ord`: Se utilizan los mapeos predefinidos para **transformar** las categorías de calidad, color y claridad en valores numéricos que respetan su orden intrínseco. Por ejemplo, `Ideal` (mejor corte) recibe un valor numérico mayor que `Fair` (peor corte).

**2. Características Geométricas y de Proporción:**
*   `fe_volume_mm3`: Se calcula como `x * y * z`. Su propósito es **estimar** el volumen del diamante, una proxy de su tamaño físico.
*   `fe_area_mm2`: Calculada como `x * y`, busca **representar** el área de la superficie superior del diamante.
*   `fe_spread_mm`: Calculada como el promedio de `x` e `y`. Intenta **medir** el "diámetro" promedio o la apariencia de tamaño del diamante visto desde arriba.
*   `fe_aspect_ratio`: El ratio `x / y`. Permite **cuantificar** cuán "redondo" o "alargado" es el diamante. Un valor cercano a 1 indica una forma más cercana a un círculo.
*   `fe_invalid_dims`: Una bandera booleana (`True`/`False`) para **identificar** filas donde alguna de las dimensiones (`x`, `y`, `z`) es cero o negativa, lo cual es físicamente imposible.
*   `fe_depth_pct_recalc`: Se recalcula el porcentaje de profundidad usando la fórmula `100 * z / ((x + y) / 2)`. El objetivo es **verificar** la consistencia con la columna `depth` original.
*   `fe_depth_pct_diff`, `fe_depth_pct_is_consistent`: Se calculan la diferencia y una bandera de consistencia entre el `depth` reportado y el recalculado. Permite **detectar** posibles errores de medición o registro.

**3. Características de Precio y Densidad:**
*   `fe_price_per_carat`: Calculado como `price / carat`. Esta es una de las métricas más importantes, ya que permite **normalizar** el precio por unidad de peso y comparar el valor relativo de diferentes diamantes.
*   `fe_log_price`, `fe_log_carat`, `fe_log_price_per_carat`: Se aplica una transformación logarítmica a `price`, `carat` y `price_per_carat`. El propósito es **manejar** la distribución sesgada (asimétrica) de estas variables, lo cual es común en variables de precios y tamaños, y a menudo mejora el rendimiento de modelos lineales.

**4. Desviaciones y Ratios Adicionales:**
*   `fe_depth_dev`, `fe_table_dev`: Calculan la desviación de `depth` y `table` respecto a valores considerados "ideales" o típicos (61.5 y 57.0 respectivamente). Permiten **cuantificar** qué tan lejos está un diamante de las proporciones estándar.
*   `fe_table_to_depth_ratio`: El ratio entre `table` y `depth`. Busca **capturar** otra faceta de la proporción del diamante.

**5. Score de Calidad Compuesto:**
*   `fe_quality_score`: Un índice sintético calculado como una suma ponderada de las características ordinales: `0.5*fe_cut_ord + 0.3*fe_color_ord + 0.2*fe_clarity_ord`. El objetivo es **crear** una única métrica que resuma la calidad general del diamante basada en sus atributos categóricos. Las ponderaciones son heurísticas.

**6. Bins o Agrupaciones:**
*   `fe_carat_bin`: Se agrupa la variable `carat` en rangos discretos. Esto puede **ayudar** a modelos a capturar relaciones no lineales y es útil para análisis de negocio (ej., analizar diamantes de "menos de 1 quilate" vs. "más de 1 quilate").

**7. Interacciones entre Características:**
*   `fe_carat_x_quality`: Se multiplica `carat` por `fe_quality_score`. La intención es **modelar** la idea de que el efecto del peso sobre el precio puede depender de la calidad del diamante (y viceversa).

**8. Normalización por Grupo (Z-score):**
*   `fe_ppc_z_by_cqc`: Se calcula el Z-score de `fe_price_per_carat` dentro de cada grupo definido por `cut`, `color` y `clarity`. El objetivo es **identificar** diamantes que son inusualmente caros o baratos para su categoría específica de calidad. Un Z-score alto indica un "premium" de precio, mientras que uno bajo indica un "descuento".

## 4. Análisis del Output del Proceso ETL

A continuación, se interpreta el reporte generado en la terminal durante la ejecución del script.

### 4.1. KPIs Globales
> ```
> ------------------------------------------------------------------------------------------------------------------------
>                                                KPIs Globales del Dataset
> ------------------------------------------------------------------------------------------------------------------------
> KPI                       Valor
> ------------------------------------------------------------------------------------------------------------------------
> Registros                 53,940
> Columnas                  11
> Duplicados (filas)        0
> Nulos totales             0
> Memoria aprox.            4,013,204 bytes
> ```
*   **Interpretación**: El dataset inicial es de buena calidad. Contiene 53,940 registros y 11 columnas. Es destacable la ausencia de filas duplicadas y de valores nulos, lo que simplifica la fase de limpieza.

### 4.2. Resumen por Columna
> ```
> ------------------------------------------------------------------------------------------------------------------------
>                                                   Resumen por Columna
> ------------------------------------------------------------------------------------------------------------------------
> Columna                Tipo                  Nulos   Únicos Top-1                          Top-2                          Top-3
> ------------------------------------------------------------------------------------------------------------------------
> index                  Int64                     0    53940 24405 (0.00%)                  51671 (0.00%)                  13261 (0.00%)
> carat                  Float64                   0      273 0.3 (4.83%)                    0.31 (4.17%)                   1.01 (4.16%)
> cut                    String                    0        5 Ideal (39.95%)                 Premium (25.57%)               Very Good (22.40%)
> color                  String                    0        7 G (20.93%)                     E (18.16%)                     F (17.69%)
> clarity                String                    0        8 SI1 (24.22%)                   VS2 (22.73%)                   SI2 (17.04%)
> ...
> ```
*   **Interpretación**: Esta tabla permite **observar** la distribución y tipo de cada columna.
    *   La columna `index` (proveniente de una columna sin nombre en el CSV) parece ser un identificador único.
    *   Las variables categóricas (`cut`, `color`, `clarity`) tienen una baja cardinalidad (pocos valores únicos) y presentan una distribución concentrada en ciertos valores (`Ideal`, `G`, `SI1` son los más frecuentes).
    *   Las variables numéricas como `carat` y `price` tienen una alta cardinalidad.

### 4.3. Estadísticos de Variables Numéricas
> ```
> ------------------------------------------------------------------------------------------------------------------------
>                                           Estadísticos de Variables Numéricas
> ------------------------------------------------------------------------------------------------------------------------
> Columna                   min          p25          p50          p75          max         mean          std
> ------------------------------------------------------------------------------------------------------------------------
> price                  326.00       950.00     2,401.00     5,324.00    18,823.00     3,932.80     3,989.44
> x                        0.00         4.71         5.70         6.54        10.74         5.73         1.12
> y                        0.00         4.72         5.71         6.54        58.90         5.73         1.14
> z                        0.00         2.91         3.53         4.04        31.80         3.54         0.71
> ```
*   **Interpretación**: Esta sección revela información clave sobre la distribución de las variables numéricas.
    *   `price`: Muestra un rango muy amplio y una gran desviación estándar. La media (`3,932`) es significativamente mayor que la mediana (`2,401`), lo que sugiere una distribución con sesgo a la derecha (cola de valores altos).
    *   `x`, `y`, `z`: Se observa un valor mínimo de `0.00` para las tres dimensiones, lo cual es físicamente imposible para un diamante y representa un problema de calidad de datos. Además, la variable `y` tiene un `max` de `58.90`, que parece un outlier extremo comparado con su media y el `max` de `x`.

### 4.4. Resumen de Calidad de Datos
> ```
> ------------------------------------------------------------------------------------------------------------------------
>                                                Calidad de Datos (Resumen)
> ------------------------------------------------------------------------------------------------------------------------
> Chequeo                             Detalle/Conteo
> ------------------------------------------------------------------------------------------------------------------------
> cut_invalid_values                  OK
> color_invalid_values                OK
> clarity_invalid_values              OK
> price_out_of_range                  0
> carat_out_of_range                  0
> x_nonpositive                       8
> y_nonpositive                       7
> z_nonpositive                       20
> ...
> ```
*   **Interpretación**: Este resumen confirma los hallazgos anteriores.
    *   Las variables categóricas no contienen valores fuera de su dominio esperado (`OK`).
    *   No hay precios o pesos no positivos.
    *   Se confirma la existencia de **20 registros** con al menos una dimensión (`x`, `y`, o `z`) igual a cero. Estos registros son problemáticos y fueron marcados por la característica `fe_invalid_dims`.

## 5. Descripción de los Archivos de Salida

Al finalizar la ejecución, el script genera un conjunto de artefactos diseñados para garantizar la reproducibilidad y facilitar el análisis posterior.

### 5.1. Archivo Parquet (`diamonds_features_{RUN_TS}.parquet`)

*   **Descripción**: Este es el principal entregable del proceso. Es un archivo binario en formato **Apache Parquet**, que ofrece una compresión eficiente y un almacenamiento columnar optimizado para cargas de trabajo analíticas.
*   **Contenido**: El archivo contiene todas las **53,940 filas** del dataset original, sin aplicar ningún filtro. Incluye las **11 columnas originales** (ya limpiadas y tipificadas) más todas las **nuevas características** generadas durante la fase de `feature_engineering`. El resultado es una tabla ancha, lista para ser consumida por herramientas de visualización, análisis estadístico o entrenamiento de modelos de machine learning.
*   **Ventajas**: Almacenar en Parquet en lugar de CSV permite **conservar** los tipos de datos de forma precisa, **reducir** el espacio en disco y **acelerar** significativamente los tiempos de lectura en análisis posteriores.

### 5.2. Archivos de Log y Metadatos

*   **`etl_diamonds_{RUN_TS}.log`**: Un archivo de texto con un registro cronológico de cada paso del ETL. Su propósito es **servir** para la depuración y para tener un historial detallado de la ejecución.
*   **`etl_diamonds_reporte_{RUN_TS}.txt`**: Una copia exacta del informe tabular que se muestra en la consola. Permite **revisar** los KPIs y resúmenes del dataset sin necesidad de volver a ejecutar el script.
*   **`etl_diamonds_schema_{RUN_TS}.json`**: Un archivo JSON que contiene metadatos clave sobre la ejecución:
    *   El timestamp de la ejecución.
    *   Las rutas de los archivos de entrada y salida.
    *   El número total de filas y columnas en el Parquet resultante.
    *   Una lista detallada de todas las columnas finales con sus respectivos tipos de datos.
    *   Una copia de los resultados de los chequeos de calidad de datos.
    *   El propósito de este archivo es **proporcionar** un resumen programático del artefacto de datos generado, facilitando su integración en sistemas automatizados.
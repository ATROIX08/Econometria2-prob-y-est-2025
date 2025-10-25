# Proyecto Diamantes — Overview (Modelos Explicativo y Predictivo)

**Autor:** Humberto Silva Baltazar  
**Curso:** Econometría II + Probabilidad y Estadística  
**Repositorio:** [Econometria2-prob-y-est-2025](https://github.com/ATROIX08/Econometria2-prob-y-est-2025)  
**Ubicación del archivo:** `C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Econometria2-prob-y-est-2025\Trabajo final\trabajo-final.md`  
**Fecha:** 24 de octubre de 2025

## 1. Propósito y Estructura del Trabajo Final
El presente documento tiene como finalidad integrar y sintetizar los hallazgos de un proyecto integral de ciencia de datos sobre el dataset de diamantes. El trabajo se estructura en una secuencia lógica que abarca desde la preparación de los datos hasta el desarrollo y la evaluación de dos tipos de modelos con objetivos distintos: uno **explicativo**, enfocado en la inferencia y la interpretabilidad de las relaciones causales, y otro **predictivo**, orientado a maximizar la precisión en la estimación del precio. Se documenta el flujo de trabajo completo, detallando las decisiones metodológicas tomadas en cada fase: (1) Extracción, Transformación y Carga (ETL); (2) Análisis Exploratorio de Datos (EDA); (3) Modelado Econométrico con Mínimos Cuadrados Ordinarios (OLS); (4) Búsqueda de Hiperparámetros para un modelo de Gradient Boosting; y (5) Entrenamiento y Evaluación del modelo predictivo final. El propósito es contrastar estas dos filosofías de modelado, demostrando cómo cada una aporta un valor único al análisis de un mismo problema de negocio.

## 2. Flujo de Datos y Preparación
La base de cualquier modelo robusto reside en la calidad de los datos que lo alimentan. El proceso ETL fue diseñado no solo para limpiar, sino para enriquecer sistemáticamente el dataset original. Las decisiones clave en esta fase habilitaron directamente el éxito de los modelos posteriores. Se implementó una normalización de esquema (nombres de columna en `snake_case`) y una tipificación estricta de variables. Más allá de la limpieza estándar (identificación de nulos o duplicados, que no se encontraron), se crearon variables de validación, como `fe_invalid_dims`, para marcar registros con dimensiones físicas imposibles (e.g., `z=0`) sin eliminarlos prematuramente, permitiendo una gestión flexible en la etapa de modelado.

El pilar del ETL fue la **ingeniería de características**. Se generaron más de 25 nuevas variables con objetivos específicos:
-   **Linealización y estabilización de varianza:** Se aplicaron transformaciones logarítmicas a `price` y `carat`. Esta decisión fue fundamental para cumplir con los supuestos del modelo OLS, convirtiendo una relación exponencial en una lineal y mitigando la heterocedasticidad.
-   **Creación de predictores compuestos:** Se construyó un `fe_quality_score` para resumir las tres "C" de calidad (corte, color, claridad) en una métrica ordinal ponderada.
-   **Captura de interacciones:** Se generó la variable `fe_carat_x_quality`, una interacción entre el tamaño y la calidad, que posteriormente demostró ser el predictor más potente en el modelo de machine learning.
-   **Normalización económica:** La creación de `fe_price_per_carat` permitió analizar el valor relativo del diamante, aislando el efecto del tamaño.

Este pre-procesamiento exhaustivo, almacenado eficientemente en formato Parquet, constituyó el cimiento sobre el cual se construyeron los análisis subsecuentes, acelerando la experimentación y garantizando la consistencia.

##EDA (patrones y riesgos)
El Análisis Exploratorio de Datos (EDA) fue crucial para diagnosticar las complejidades del dataset y guiar la estrategia de modelado. El análisis reveló dos desafíos principales: una fuerte asimetría en las variables clave (`price`, `carat`) y una severa multicolinealidad entre los predictores relacionados con el tamaño físico.

![EDA 3D XYZ](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_3d_xyz.png)  
**Interpretación:** Este gráfico tridimensional de las dimensiones físicas (`x`, `y`, `z`) ofrece una confirmación visual inmediata del diagnóstico de multicolinealidad. Los puntos no forman una nube esférica y dispersa, sino una línea densa y alargada que parte del origen. Esto demuestra que las tres variables se mueven casi en perfecta sincronía; conocer dos de ellas permite predecir la tercera con alta precisión. Incluir `x`, `y`, `z` y `carat` simultáneamente en un modelo de regresión lineal inflaría artificialmente la varianza de los coeficientes, haciéndolos inestables e ininterpretables. El EDA, por tanto, impuso una decisión de diseño crítica: seleccionar un único representante del tamaño (se eligió `carat` y su transformación `fe_log_carat`) para preservar la integridad del modelo econométrico.

## 4. Modelo Explicativo (OLS): lo que explica y cómo validarlo
Para el objetivo de **explicar** las variaciones del precio, se optó por un modelo de regresión por Mínimos Cuadrados Ordinarios (OLS) debido a su inigualable interpretabilidad. La especificación del modelo fue semi-logarítmica (`log(price)` como variable dependiente), lo que permite interpretar los coeficientes como cambios porcentuales. El modelo logró un R² ajustado de 0.983, indicando un altísimo poder explicativo. El coeficiente de `fe_log_carat` (1.88) se interpretó como una elasticidad: un aumento del 1% en el peso del diamante se asocia con un aumento del 1.88% en su precio, *ceteris paribus*. De manera similar, se cuantificaron los descuentos o primas asociadas a cada categoría de corte, color y claridad.

Sin embargo, un modelo explicativo no es válido solo por su R². Su fiabilidad depende del cumplimiento de los supuestos clásicos, lo cual se evaluó mediante diagnósticos de residuos.

![OLS Residuos vs Ajustados](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/explicativo/ols_resid_vs_fitted_20251024_195908.png)  
**Interpretación:** El gráfico de residuos contra valores ajustados revela una debilidad fundamental del modelo OLS clásico en este contexto. El patrón de "embudo" o "cono", donde la dispersión de los errores aumenta a medida que aumenta el valor predicho, es una evidencia inequívoca de **heterocedasticidad**. Esto significa que el modelo es mucho menos preciso para los diamantes más caros. La implicación econométrica es grave: aunque los coeficientes estimados siguen siendo insesgados, sus errores estándar son incorrectos. Por lo tanto, todas las inferencias estadísticas (p-valores, intervalos de confianza) reportadas por el modelo clásico no son fiables.

## 5. Modelo Predictivo (HGBR + Entrenamiento final): potencia y límites
Con el objetivo de **predecir** el precio con la máxima precisión posible, se eligió un enfoque de machine learning, específicamente el `HistGradientBoostingRegressor` (HGBR), un algoritmo basado en árboles de decisión que captura de forma nativa relaciones no lineales e interacciones complejas. El proceso se dividió en dos fases: primero, una búsqueda exhaustiva de hiperparámetros (`GridSearchCV`) paralelizada con Dask para encontrar la "receta" óptima del modelo; y segundo, el entrenamiento del modelo final con esta receta sobre el conjunto completo de datos de entrenamiento.

El modelo final demostró un rendimiento predictivo excepcional en el conjunto de prueba (datos no vistos).

![Observado vs Predicho (best)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/observado_vs_predicho_best_20251012_202645.png)  
**Lectura crítica:** Esta gráfica valida de manera contundente la potencia del modelo predictivo. La casi perfecta alineación de los puntos a lo largo de la línea de identidad (y=x) indica un ajuste extraordinario, corroborado por un R² en el conjunto de prueba de 0.9818 y un Error Porcentual Absoluto Medio (MAPE) de solo 8.03%. A diferencia del modelo OLS, que mostraba una creciente dispersión de errores, el HGBR mantiene una alta precisión en todo el rango de precios, incluyendo los diamantes de mayor valor. El análisis de importancia de variables reveló que la característica más influyente fue `fe_carat_x_quality`, una interacción creada en el ETL, superando incluso a `carat` por sí solo. Esto subraya que los modelos de machine learning no solo son potentes por su algoritmo, sino también por su capacidad para explotar una ingeniería de características inteligente.

## 6. Comparativa explícita: Explicar vs Predecir
La elección entre un modelo explicativo y uno predictivo depende del objetivo final. El OLS busca entender las relaciones subyacentes y probar hipótesis, mientras que el HGBR busca la predicción más precisa posible, a menudo a costa de la interpretabilidad.

**Tabla M — Contraste de enfoques**
| Criterio | OLS (explicativo) | HGBR (predictivo) |
|---|---|---|
| Objetivo | Interpretar relaciones y cuantificar efectos marginales (*ceteris paribus*). | Minimizar el error de predicción en datos no vistos (generalización). |
| Métricas foco | R², p-valores, pruebas F, diagnósticos de residuos (AIC, BIC). | MAE/MAPE/RMSE y R² en un conjunto de prueba (*hold-out*). |
| Supuestos | Requiere el cumplimiento de supuestos estrictos (linealidad, homocedasticidad, normalidad de errores) para una inferencia válida. | Es no paramétrico y robusto a la no linealidad, interacciones y distribuciones no normales. No tiene supuestos estadísticos en el sentido clásico. |
| Producto | Un informe con una ecuación interpretable y pruebas de hipótesis sobre la significancia de los coeficientes. | Una "receta" de hiperparámetros óptima y un objeto serializado (`.joblib`) que encapsula el pipeline entrenado, listo para hacer predicciones. |

## 7. Limitaciones, Riesgos y Recomendaciones
A pesar del éxito de los modelos, es crucial reconocer sus limitaciones y los riesgos inherentes al proceso.
-   **Calidad de Datos:** Se identificaron y marcaron registros con dimensiones inválidas, pero la precisión de las mediciones originales es una suposición. Errores de medición podrían introducir ruido.
-   **Colinealidad y Leakage:** La colinealidad fue gestionada, pero el riesgo de fuga de datos (`data leakage`) es persistente. Se tuvo especial cuidado en excluir cualquier variable derivada del precio (`fe_price_per_carat`) del conjunto de predictores para evitar que el modelo "hiciera trampa".
-   **Estabilidad Temporal:** Los modelos se entrenaron con datos de corte transversal. Las relaciones de precios en el mercado de diamantes pueden cambiar con el tiempo (*model drift*). Un modelo desplegado en producción requeriría un monitoreo continuo de su rendimiento.

## 8. Navegación del Reporte
Para un análisis detallado de cada una de las fases del proyecto, se puede consultar la documentación específica a través de los siguientes enlaces:

-   [ETL — Preparador de Datos](docs/sub-paginas/etl-diamonds.md)  
-   [EDA — Explorador de Datos](docs/sub-paginas/eda-diamonds.md)  
-   [Modelo Explicativo (OLS)](docs/sub-paginas/explicative.md)  
-   [Modelo Predictivo (Búsqueda)](docs/sub-paginas/predictive.md)  
-   [Entrenamiento Final (best)](docs/sub-paginas/train_best.md)
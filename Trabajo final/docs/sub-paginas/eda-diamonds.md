# EDA

**Autor:** Humberto Silva Baltazar · **Curso:** Econometría II + Probabilidad y Estadística  
**Script:** [`eda_diamonds.py`](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/src/eda_diamonds.py)  
**Entrada:** `diamonds.parquet` 

## 1. Objetivo y Enfoque
El presente Análisis Exploratorio de Datos (EDA) tiene como objetivo fundamental caracterizar el dataset de diamantes para informar el diseño de un modelo econométrico robusto que explique y prediga la variable `price`. El enfoque metodológico se centra en tres pilares: (1) el análisis de las distribuciones univariadas para identificar asimetrías y la presencia de valores atípicos; (2) la evaluación de relaciones bivariadas y multivariadas para cuantificar la asociación entre predictores y la variable dependiente; y (3) el diagnóstico de problemas de calidad de datos, con especial atención en la multicolinealidad, que puede comprometer la estabilidad y la interpretabilidad de los coeficientes de un modelo de regresión. Para ello, se emplean tanto técnicas de visualización (histogramas, diagramas de dispersión, matrices de correlación) como métodos estadísticos cuantitativos (estadísticos descriptivos, Factor de Inflación de la Varianza o VIF).

## 2. Descriptivos Esenciales
El conjunto de datos procesado contiene 53,940 registros y 38 columnas, incluyendo las variables originales y un conjunto de características de ingeniería (`features`). El análisis descriptivo revela información crucial para el modelado: la variable objetivo, `price`, presenta una media de 3,932.80 USD y una mediana de 2,401.00 USD, con una desviación estándar elevada de 3,989.44 USD. Esta divergencia entre media y mediana, junto a un sesgo positivo de 1.62, confirma una distribución fuertemente asimétrica hacia la derecha, donde una minoría de diamantes de alto valor influye desproporcionadamente en la media. Un comportamiento similar se observa en la variable `carat` (quilates), con una media de 0.80 y una mediana de 0.70 (sesgo de 1.12), lo que es esperable dado que es el principal determinante del precio.

Se identificaron hallazgos notables que requieren atención. Primero, la pronunciada asimetría en las distribuciones de `price` y `carat` sugiere que la aplicación de transformaciones logarítmicas podría ser indispensable para estabilizar la varianza de los errores y linealizar la relación, cumpliendo así con los supuestos del modelo de Mínimos Cuadrados Ordinarios (OLS). Segundo, el chequeo de calidad de datos reveló que 20 registros (0.04% del total) presentan al menos una dimensión (`x`, `y`, o `z`) con valor cero o negativo, lo cual es físicamente imposible. Aunque su proporción es mínima, estos puntos anómalos pueden ejercer una influencia indebida en la estimación de los parámetros del modelo y deben ser gestionados, ya sea mediante su eliminación o imputación.

## 3. Correlaciones y Riesgo de Multicolinealidad
El análisis de correlación y multicolinealidad es vital para seleccionar un conjunto de predictores linealmente independientes. Las correlaciones de Pearson más relevantes y los resultados del VIF se resumen a continuación.

**Tabla C — Indicadores de correlación (resumen)**
| Par de variables | ρ (Pearson) |
|---|---|
| `price`–`carat` | 0.922 |
| `fe_price_per_carat`–`carat` | 0.770 |

**Tabla D — VIF (variables numéricas)**
| Variable | VIF |
|---|---|
| `x` | 163.994 |
| `carat` | 103.969 |

**Nota:** El análisis VIF confirma un riesgo severo de multicolinealidad. Valores muy superiores al umbral crítico (generalmente establecido en 5 o 10) se observan en todas las variables relacionadas con el tamaño: `carat`, las dimensiones físicas (`x`, `y`, `z`) y, por extensión, el precio. Las variables logarítmicas (`fe_log_price`, `fe_log_carat`) reportan un VIF infinito, lo cual es matemáticamente esperado al incluirlas junto a sus contrapartes no transformadas, indicando redundancia perfecta. Esto obliga a una selección de variables muy cuidadosa; por ejemplo, se debe utilizar `carat` como el único representante del tamaño físico, excluyendo `x`, `y`, `z` y el volumen (`fe_volume_mm3`) del modelo final para evitar la inflación de la varianza de los estimadores.

## 4. Figuras EDA e Interpretación Analítica
![3D x-y-z](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_3d_xyz.png)  
**Interpretación:** El gráfico de dispersión 3D de las dimensiones físicas (`x`, `y`, `z`) ilustra la fuerte relación co-lineal que existe entre ellas. Los puntos forman una nube densa y alargada que se extiende desde el origen, lo que es coherente con la forma en que las dimensiones de un diamante escalan conjuntamente. Más importante aún, esta visualización permite identificar de forma inmediata valores atípicos extremos que podrían ser errores de medición o de entrada de datos.

![Precio por claridad](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_ppc_by_clarity.png)  
**Interpretación:** Este diagrama de cajas revela una relación no monotónica y compleja entre el precio por quilate (`fe_price_per_carat`) y la claridad del diamante. Contrario a la intuición de que una mayor claridad siempre resulta en un mayor precio relativo, las claridades de gama media-alta (VS1, VS2) presentan medianas y rangos intercuartílicos comparables e incluso superiores a los de las categorías más altas (VVS1, IF). Este fenómeno sugiere la existencia de variables de confusión; es posible que los diamantes de mayor tamaño (y por tanto, más caros en términos absolutos) sean más propensos a tener inclusiones, lo que concentra el `carat` en las categorías de claridad intermedias. Esta no linealidad implica que tratar `clarity` como una simple variable ordinal en el modelo podría ser una simplificación excesiva.

![Matriz de Correlación](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_correlation_matrix.png)  
**Interpretación:** El mapa de calor de correlaciones ofrece una visión panorámica de las interdependencias lineales. El bloque de color amarillo intenso en la esquina superior izquierda confirma visualmente la severa multicolinealidad detectada por el VIF entre `price`, `carat`, las dimensiones físicas y las características de volumen/área. Esta redundancia de información es la principal amenaza para la estabilidad del modelo. Adicionalmente, se observa que las características de calidad codificadas ordinalmente (`fe_cut_ord`, `fe_color_ord`, `fe_clarity_ord`) tienen una correlación positiva pero moderada con el precio, validando su inclusión como predictores relevantes pero no dominantes.

![Tasa de Outliers](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/diamonds_eda_20251012_163209_outlier_rates.png)  
**Interpretación:** El gráfico de barras que cuantifica el porcentaje de valores atípicos por variable (usando el criterio del rango intercuartílico) resalta que `price` es la variable con la mayor proporción de outliers (superior al 6%). Esto es consistente con su distribución asimétrica. Varias de las características de ingeniería, especialmente las que se basan en ratios, también presentan una tasa de atípicos significativa. 

## 5. Implicaciones para Modelado
- **Seleccionar variables candidatas (post-EDA):** El conjunto inicial de predictores para un modelo de regresión debería incluir `log_carat` como la variable principal de tamaño (transformada para corregir asimetría y mitigar colinealidad). A esta se deben sumar las métricas de calidad (`fe_cut_ord`, `fe_color_ord`, `fe_clarity_ord`) y potencialmente `table` y `depth`. Variables como `x`, `y`, `z`, `fe_volume_mm3` y `price` deben ser excluidas de los predictores para evitar la multicolinealidad y la endogeneidad, si `log_price` es la variable dependiente.
- **Recomendar transformaciones y codificaciones:** Es imperativo aplicar una transformación logarítmica a la variable dependiente (`price` → `log_price`) y al predictor principal (`carat` → `log_carat`). Las variables categóricas (`cut`, `color`, `clarity`) ya están codificadas ordinalmente, lo que constituye un punto de partida válido. Sin embargo, se debería experimentar con una codificación *one-hot* (variables dummy) para capturar los efectos no lineales observados, especialmente en el caso de `clarity`.

## 6. Riesgos, Sesgos y Decisiones de Diseño
- **Señalar riesgos de outliers, colinealidad remanente, leakages potenciales:** Los riesgos primordiales identificados son: (1) **Multicolinealidad severa** entre todas las variables asociadas al tamaño, lo que exige la exclusión de predictores redundantes. (2) **Outliers influyentes** en `price` y otras variables clave que pueden sesgar las estimaciones de OLS. (3) **Fuga de datos (data leakage)** si se utilizan por error características derivadas del precio (como `fe_price_per_carat` o `fe_ppc_z_by_cqc`) para predecir el propio precio.
- **Justificar decisiones de exclusión/transformación:** La decisión de utilizar exclusivamente `log_carat` como proxy del tamaño, excluyendo `x`, `y`, `z`, está directamente justificada por los resultados del VIF, con el fin de obtener estimadores de coeficientes estables. La transformación logarítmica de `price` y `carat` es una respuesta directa a la fuerte asimetría detectada en sus distribuciones. Finalmente, los registros con `fe_invalid_dims = True` deben ser eliminados del conjunto de entrenamiento por representar datos corruptos.

## 7. Evidencia (extracto del log)
```
------------------------------------------------------------------------------------------------------------------------
                                         Diagnóstico de Multicolinealidad (VIF)                                         
------------------------------------------------------------------------------------------------------------------------
Variable                        VIF
------------------------------------------------------------------------------------------------------------------------
price                        81.009
carat                       103.969
fe_price_per_carat           78.722
fe_quality_score              2.513
fe_log_price                    inf
fe_log_carat                    inf
fe_log_price_per_carat          inf
x                           163.994
y                            20.596
z                            23.596
depth                         2.034
table                         1.422
```

## 8. Conclusiones operativas
El EDA ha demostrado que un modelo lineal simple sobre las variables en su escala original sería ingenuo e probablemente ineficaz, sesgado y con estimadores inestables. El análisis prescribe una ruta clara para la especificación del modelo: es necesario transformar las variables clave para cumplir con los supuestos de linealidad y homocedasticidad, y seleccionar cuidadosamente un conjunto de predictores no colineales para garantizar la interpretabilidad y fiabilidad de los coeficientes. Este análisis exploratorio no solo orienta la construcción del modelo OLS inicial, sino que también sienta las bases para futuras iteraciones, como la inclusión de interacciones o el uso de algoritmos de aprendizaje automático más complejos y robustos.
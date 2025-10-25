# Modelo Explicativo — OLS

**Autor:** Humberto Silva Baltazar · **Curso:** Econometría II + Probabilidad y Estadística  
**Script:** [`explicative.py`](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/src/modeling/explicative.py)  
**Entrada:** `diamonds_features_20251012_155956.parquet` 

## 1. Objetivo y Diseño
El objetivo central de este análisis es desarrollar un modelo econométrico para explicar las variaciones en el precio de los diamantes. Se utiliza el método de Mínimos Cuadrados Ordinarios (OLS) por su alta interpretabilidad, que permite cuantificar el impacto marginal de características clave como el peso (quilates), la calidad del corte, el color y la claridad.

El diseño del estudio se enfoca en la validación de los supuestos del modelo lineal clásico para asegurar la fiabilidad de las inferencias estadísticas. Para ello, la variable dependiente se transforma logarítmicamente —**log(price)**—. Esta transformación tiene un doble propósito: primero, linealizar la relación teóricamente exponencial entre el peso y el precio; y segundo, mitigar la heterocedasticidad, estabilizando la varianza del término de error, un problema común en datos de corte transversal con variables monetarias.

## 2. Especificación y Muestras
El modelo se especifica como una regresión semi-logarítmica (log-lin), donde el logaritmo natural del precio es la variable dependiente. Las variables explicativas incluyen el logaritmo del peso (`fe_log_carat`), desviaciones de las proporciones óptimas (`fe_depth_dev`, `fe_table_dev`) y variables categóricas para el corte, color y claridad, las cuales son tratadas como variables dummy con categorías de referencia explícitas (`Ideal`, `G`, `SI1`).

- **Fórmula:** `log(price) ~ fe_log_carat + fe_depth_dev + fe_table_dev + C(cut, Treatment(reference='Ideal')) + C(color, Treatment(reference='G')) + C(clarity, Treatment(reference='SI1'))`.  
- **Muestra:** El conjunto de datos, tras la limpieza de 20 registros inválidos, se dividió en muestras de entrenamiento y prueba. El modelo se ajustó sobre la muestra de entrenamiento.
    - **Train:** 43,136 observaciones.
    - **Test:** 10,784 observaciones.
- **Criterios de partición:** Se realizó una partición aleatoria del 80% para entrenamiento y 20% para prueba, utilizando una semilla fija (`seed=42`) para garantizar la reproducibilidad de los resultados.

## 3. Resultados Principales (métricas globales)
El modelo OLS clásico demuestra un poder explicativo excepcionalmente alto. Un R² de 0.983 indica que el 98.3% de la variabilidad en el logaritmo del precio es explicada por las variables del modelo. La significancia global, confirmada por un p-valor del estadístico F cercano a cero, valida la relevancia conjunta de los regresores. Las métricas de error en la escala original del precio (USD) ofrecen una visión práctica de su rendimiento predictivo.

**Tabla E — Métricas OLS**
| Métrica | Valor | Interpretación |
|---|---|---|
| R² | 0.983 | Alto poder explicativo en la escala logarítmica. |
| R² ajustado | 0.983 | Penaliza por predictores, confirmando el buen ajuste. |
| F-stat / p-value | 1.214e+05 / <0.001 | El modelo en su conjunto es estadísticamente significativo. |
| RMSE (en Test, USD) | 783.76 | El error de predicción típico del modelo es de $783.76 USD. |
| **MAPE (en Test)** | **10.43%** | En promedio, las predicciones del modelo se desvían un 10.43% del precio real. |

## 4. Coeficientes Clave e Interpretación
La interpretación de los coeficientes en este modelo permite cuantificar efectos porcentuales sobre el precio. A continuación, se detallan los hallazgos para las variables más importantes y se construye la ecuación del modelo.

**Tabla F — Coeficientes (selección)**
| Variable | Signo esperado | Estimación | p-valor | Lectura económica |
|---|---|---:|---:|---|
| `fe_log_carat` | + | 1.8843 | <0.001 | Ceteris paribus, un aumento del 1% en el peso (quilates) se asocia con un aumento del 1.88% en el precio. |
| `C(cut)[T.Fair]` | - | -0.1558 | <0.001 | Un corte 'Fair' se asocia con un precio ~14.4% menor que un corte 'Ideal', manteniendo lo demás constante. |

### Construcción de la Ecuación del Modelo

La ecuación estimada a partir de los resultados de OLS es la siguiente:

```math
\begin{array}{rl}
\widehat{\log(\text{price})} = & 8.4508 \\
& - 0.1558\,\mathrm{I}(\text{cut}=\text{Fair}) \\
& - 0.0799\,\mathrm{I}(\text{cut}=\text{Good}) \\
& - 0.0222\,\mathrm{I}(\text{cut}=\text{Premium}) \\
& - 0.0424\,\mathrm{I}(\text{cut}=\text{Very Good}) \\
& + 0.1607\,\mathrm{I}(\text{color}=\text{D}) \\
& + 0.1065\,\mathrm{I}(\text{color}=\text{E}) \\
& + 0.0664\,\mathrm{I}(\text{color}=\text{F}) \\
& - 0.0905\,\mathrm{I}(\text{color}=\text{H}) \\
& - 0.2128\,\mathrm{I}(\text{color}=\text{I}) \\
& - 0.3494\,\mathrm{I}(\text{color}=\text{J}) \\
& - 0.5939\,\mathrm{I}(\text{clarity}=\text{I1}) \\
& + 0.5203\,\mathrm{I}(\text{clarity}=\text{IF}) \\
& - 0.1655\,\mathrm{I}(\text{clarity}=\text{SI2}) \\
& + 0.2190\,\mathrm{I}(\text{clarity}=\text{VS1}) \\
& + 0.1488\,\mathrm{I}(\text{clarity}=\text{VS2}) \\
& + 0.4259\,\mathrm{I}(\text{clarity}=\text{VVS1}) \\
& + 0.3545\,\mathrm{I}(\text{clarity}=\text{VVS2}) \\
& + 1.8843\,\text{fe\_log\_carat} \\
& - 0.0012\,\text{fe\_depth\_dev} \\
& - 0.0004\,\text{fe\_table\_dev}
\end{array}
```
Donde `I(...)` representa las variables indicadoras (dummy) que toman el valor de 1 si la condición es verdadera y 0 en caso contrario.

### Interpretación de la Ecuación

1.  **Intercepto (8.4508):** El intercepto representa el valor predicho del logaritmo del precio para un **diamante base de referencia**. Este diamante tiene todas las variables continuas en cero (`fe_log_carat=0` que implica 1 quilate, `fe_depth_dev=0`, etc.) y todas las variables categóricas en su nivel de referencia (`cut='Ideal'`, `color='G'`, `clarity='SI1'`). Por lo tanto, el precio predicho para este diamante de referencia es $e^{8.4508} \approx \$4,678.50$ USD.

2.  **Variables Continuas (`fe_log_carat`):** Al ser un modelo log-log para esta variable, el coeficiente `1.8843` es una **elasticidad**. Indica que por cada 1% de aumento en el peso del diamante, se espera que el precio aumente en un 1.88%, ceteris paribus.

3.  **Variables Categóricas (Dummies):** El coeficiente de una variable dummy, como `C(cut)[T.Fair] = -0.1558`, representa la diferencia en el log-precio en comparación con la categoría de referencia (`cut='Ideal'`). Para interpretarlo como un cambio porcentual, se utiliza la fórmula $(e^\beta - 1) \times 100\%$.
    - **Ejemplo de Corte:** Un diamante con corte 'Fair', comparado con uno idéntico pero de corte 'Ideal', tiene un precio esperado que es $(e^{-0.1558} - 1) \times 100\% \approx -14.4\%$ más bajo.
    - **Ejemplo de Color:** Un diamante de color 'D' (el mejor), comparado con uno idéntico pero de color 'G' (referencia), tiene un precio esperado que es $(e^{0.1607} - 1) \times 100\% \approx +17.4\%$ más alto.

## 5. Diagnósticos Gráficos e Interpretación
El análisis gráfico de los residuos es fundamental para validar los supuestos del modelo OLS.

![Residuos vs Ajustados](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/explicativo/ols_resid_vs_fitted_20251024_195908.png)  
**Interpretación:** El gráfico de residuos contra valores ajustados revela un patrón de "embudo" (cono), donde la dispersión de los residuos aumenta a medida que el log-precio predicho se incrementa. Este es un síntoma inequívoco de **heterocedasticidad**. Implica que el modelo es menos preciso y tiene mayor incertidumbre en sus predicciones para los diamantes más caros. Esta violación del supuesto de homocedasticidad invalida los errores estándar y, por ende, las pruebas de hipótesis del OLS clásico.

![QQ-plot de Residuales](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/explicativo/ols_qqplot_20251024_195908.png)  
**Interpretación:** El QQ-plot compara los cuantiles de los residuos con los de una distribución normal teórica. Los puntos se desvían sistemáticamente de la línea de 45 grados, especialmente en las colas. Esto indica que la distribución de los residuos tiene **colas más pesadas (leptokurtosis)** que la normal. En la práctica, significa que el modelo genera errores extremos (tanto sobreestimaciones como subestimaciones muy grandes) con más frecuencia de lo que se esperaría bajo normalidad. Esto viola el supuesto de normalidad de los residuos.

### Gráfico de Predicciones vs. Valores Reales (Test)
![Predicho vs Real (Test)](https://github.com/ATROIX08/Econometria2-prob-y-est-2025/blob/main/Trabajo%20final/plots/modeling/explicativo/pred_vs_real_test_ols_20251024_195908.png?raw=1)  
**Interpretación:** Este gráfico compara los precios predichos con los reales en el conjunto de prueba, ambos en su escala original (USD). La línea de 45 grados (discontinua) representa una predicción perfecta (Predicho = Real). La alta concentración de puntos alrededor de esta línea confirma visualmente el excelente poder predictivo general del modelo, alineado con el alto R² y el MAPE relativamente bajo.
- **Visualización de Heterocedasticidad:** El gráfico ofrece una visión intuitiva de la heterocedasticidad. Para precios bajos (< $5,000), los puntos están muy ajustados a la línea. A medida que el precio real aumenta, la nube de puntos se ensancha verticalmente, indicando que los errores de predicción en dólares se vuelven mucho más grandes. Esto confirma que la fiabilidad de las predicciones puntuales disminuye para los diamantes de mayor valor.

## 6. Supuestos y Validez
La validez del modelo OLS clásico depende del cumplimiento de varios supuestos clave.

**Tabla G — Verificación de supuestos**
| Supuesto | Evidencia | Conclusión |
|---|---|---|
| Linealidad en los parámetros | Gráfico de residuos vs. ajustados no muestra un patrón curvo sistemático. | **Cumplido (tentativamente)**. La forma funcional lineal parece adecuada. |
| Normalidad de residuos | Prueba de Jarque-Bera (p < 0.001) y QQ-plot con colas pesadas. | **Claramente violado**. La no normalidad afecta la validez de las pruebas t y F, aunque este efecto se mitiga en muestras grandes. |
| Homocedasticidad | Pruebas de Breusch-Pagan y White (ambas p < 0.001) y patrón de embudo en gráfico de residuos. | **Claramente violado**. La presencia de heterocedasticidad es la principal debilidad del modelo clásico. Invalida las inferencias basadas en errores estándar no robustos. |

## 8. Limitaciones y Sensibilidades
- **Limitaciones y Sesgos:** La principal limitación del modelo OLS clásico presentado es la severa violación de los supuestos de **homocedasticidad y normalidad**. Esto implica que los errores estándar, los p-valores y los intervalos de confianza reportados en el resumen estándar **no son confiables** para la inferencia estadística. Aunque los coeficientes estimados por OLS son insesgados y consistentes incluso con heterocedasticidad, las pruebas de hipótesis sobre ellos son inválidas. El alto R² puede ser engañoso si el objetivo es la inferencia.

## 9. Evidencia (extracto del log)
```
2025-10-24 19:59:08 | INFO | ============================================================================================================
2025-10-24 19:59:08 | INFO | INICIO — MODELO EXPLICATIVO (Diamantes) — versión simple (solo OLS clásico)
2025-10-24 19:59:08 | INFO | ============================================================================================================
2025-10-24 19:59:08 | INFO | RUN_TS: 20251024_195908
2025-10-24 19:59:08 | INFO | Python: 3.12.7, pandas: 2.3.1, numpy: 1.26.4, statsmodels: 0.14.2
2025-10-24 19:59:08 | INFO | Fórmula usada (simple, sin interacciones):
2025-10-24 19:59:08 | INFO | fe_log_price ~ fe_log_carat + fe_depth_dev + fe_table_dev + C(cut, Treatment(reference='Ideal')) + C(color, Treatment(reference='G')) + C(clarity, Treatment(reference='SI1'))
2025-10-24 19:59:08 | INFO | Split train/test: train=43,136 | test=10,784
2025-10-24 19:59:08 | INFO | Ajustando OLS (clásico)...
2025-10-24 19:59:09 | INFO | [OLS] R² train=0.982550 | R² test=0.982888
2025-10-24 19:59:09 | INFO | [OLS] AIC=-50857.934 | BIC=-50675.820
2025-10-24 19:59:09 | INFO | [OLS] Métricas TEST (escala precio): MAE=397.24 | RMSE=783.76 | MAPE=10.43%
2025-10-24 19:59:10 | INFO | [OLS] Jarque–Bera: stat=17037.4782, p=0, skew=0.2542, kurt=6.0366
2025-10-24 19:59:10 | INFO | [OLS] Breusch–Pagan: stat=1084.3205, p=3.958e-217 | White: stat=4857.3355, p=0
2025-10-24 19:59:11 | INFO | TOP 10 influyentes (Cook’s D) [pos=posición en train, idx=índice original]:
2025-10-24 19:59:11 | INFO | 
     pos    idx   cooks_d  leverage  std_resid
0  10230  49773  0.033792  0.003376  14.472620
1  35572  46476  0.011477  0.002062  10.801583
2  10384  38153  0.011213  0.002380   9.935409
...
2025-10-24 19:59:12 | INFO | Resumen OLS (clásico):
2025-10-24 19:59:12 | INFO | 
                            OLS Regression Results                            
==============================================================================
Dep. Variable:           fe_log_price   R-squared:                       0.983
Model:                            OLS   Adj. R-squared:                  0.983
Method:                 Least Squares   F-statistic:                 1.214e+05
Date:                Fri, 24 Oct 2025   Prob (F-statistic):               0.00
Time:                        19:59:12   Log-Likelihood:                 25450.
No. Observations:               43136   AIC:                        -5.086e+04
Df Residuals:                   43115   BIC:                        -5.068e+04
Df Model:                          20                                         
Covariance Type:            nonrobust                                         
=====================================================================================================================
                                                        coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------------------------------------
Intercept                                             8.4508      0.002   3896.737      0.000       8.447       8.455
C(cut, Treatment(reference='Ideal'))[T.Fair]         -0.1558      0.004    -35.073      0.000      -0.165      -0.147
...
fe_log_carat                                          1.8843      0.001   1481.204      0.000       1.882       1.887
...
==============================================================================
Omnibus:                     3526.693   Durbin-Watson:                   2.012
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            17037.478
Skew:                           0.254   Prob(JB):                         0.00
Kurtosis:                       6.037   Cond. No.                         21.6
==============================================================================
...
2025-10-24 19:59:12 | INFO | FIN — MODELO EXPLICATIVO (simple, OLS clásico)
2025-10-24 19:59:12 | INFO | ============================================================================================================
```

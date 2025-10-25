# Trabajo Final - Análisis y Modelado de Diamantes

Este proyecto contiene un flujo de trabajo completo para el análisis de un conjunto de datos de diamantes, desde la limpieza y preparación de los datos hasta el modelado predictivo y explicativo.

## Estructura del Directorio

- **data/**: Contiene el conjunto de datos original `diamonds.csv`.
- **docs/**: Documentación adicional del proyecto.
- **output/**: Almacena los resultados generados por los scripts, como archivos Parquet, logs y modelos entrenados.
- **plots/**: Guarda las visualizaciones y gráficos generados durante el análisis exploratorio y el modelado.
- **src/**: Contiene el código fuente del proyecto, organizado en subcarpetas.
  - **modeling/**: Scripts para el modelado de datos.
    - `explicative.py`: Modelo explicativo utilizando OLS.
    - `predictive.py`: Modelo predictivo con `HistGradientBoostingRegressor` y `GridSearchCV`.
    - `train_best.py`: Entrenamiento del mejor modelo con los hiperparámetros óptimos.
  - `eda_diamonds.py`: Script para el análisis exploratorio de datos (EDA).
  - `etl_diamonds.py`: Script para la extracción, transformación y carga de datos (ETL).

## Flujo de Trabajo

1. **ETL**: Ejecutar `etl_diamonds.py` para limpiar y transformar los datos.
2. **EDA**: Ejecutar `eda_diamonds.py` para generar visualizaciones y entender los datos.
3. **Modelado**: Ejecutar los scripts en la carpeta `modeling` para crear y evaluar modelos.

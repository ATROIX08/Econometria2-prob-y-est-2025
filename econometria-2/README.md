# Econometría II

Esta carpeta contiene scripts y proyectos relacionados con la materia de Econometría II.

## Contenido

### `red-neuronal-mtcars.py`

Este script implementa un **Perceptrón Multicapa (MLP)**, un tipo de red neuronal, para resolver un problema de clasificación.

- **Objetivo**: Clasificar el número de cilindros (`cyl`) de los automóviles del famoso dataset `mtcars`.
- **Librerías**: Utiliza `scikit-learn` para el modelo, `pandas` para la manipulación de datos y `matplotlib`/`seaborn` para las visualizaciones.

El código está extensamente comentado para servir como una guía de aprendizaje sobre los siguientes temas:
- La importancia de escalar los datos de entrada.
- Cómo dividir los datos en conjuntos de entrenamiento y prueba sin "fuga de información" (`data leakage`).
- El uso de `Pipeline` en `scikit-learn` para encapsular el preprocesamiento y el modelo.
- Los conceptos básicos de un MLP: capas, neuronas y funciones de activación.
- El significado de los hiperparámetros clave del modelo.
- Cómo evaluar el rendimiento del clasificador a través de métricas como *accuracy*, la matriz de confusión y la curva de pérdida.

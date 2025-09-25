# -*- coding: utf-8 -*-
# =============================================================================
# MLP minimalista (pero bien explicado) para clasificar 'cyl' (cilindros) en el
# dataset mtcars (32 filas).
#
# ¿Qué aprenderás leyendo SOLO los COMENTARIOS de este archivo?
# - Por qué escalamos los inputs antes de entrenar redes neuronales.
# - Cómo separar entrenamiento/prueba sin "hacer trampa" (fuga de información).
# - Qué es un Pipeline en scikit-learn y por qué es la forma correcta de juntar
#   preprocesamiento + modelo sin fugas.
# - Qué es un MLP (perceptrón multicapa), cómo se "arma" con capas, neuronas y
#   funciones de activación.
# - Qué significan y por qué se eligieron los hiperparámetros principales:
#   hidden_layer_sizes, activation, solver, alpha (regularización L2),
#   learning_rate, learning_rate_init, max_iter y early_stopping.
# - Cómo entrena internamente (muy resumido) un MLPClassifier (backprop + optimizador).
# - Cómo evaluar el modelo: accuracy, reporte de clasificación, matriz de confusión,
#   curva de pérdida y validación cruzada estratificada.
#
# NOTA: Dejamos al final dos visualizaciones opcionales (dibujo de red y heatmap
# de pesos). No son necesarias para entender la red; están como extra. Si no las
# quieres, puedes comentar esas celdas sin afectar lo esencial.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as mcm  # <- usamos 'mcm' para evitar colisiones con variables llamadas 'cm'

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- 1) Carga y preparación de datos ----------
# Usamos el dataset "mtcars" (clásico de R) desde una URL pública.
# Contiene varias variables numéricas de autos (mpg, hp, wt, etc.) y, entre ellas,
# 'cyl' (número de cilindros) que será nuestra variable objetivo (clasificación multiclase).
url = "https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"

# Leemos el CSV y renombramos la columna 'Unnamed: 0' a 'model' (nombre del auto).
df = pd.read_csv(url).rename(columns={"Unnamed: 0": "model"})

# Definimos X (features) y y (target):
# - X: TODAS las variables numéricas excepto 'model' (texto/etiqueta) y 'cyl' (target).
# - y: la columna 'cyl' convertida a int (por si acaso).
X = df.drop(columns=["model", "cyl"])
y = df["cyl"].astype(int)

# ---------- 2) Split estratificado ----------
# Dividimos en entrenamiento y prueba con estratificación:
# - 'test_size=0.3' -> 30% de los datos a prueba.
# - 'stratify=y' asegura que la proporción de clases (cilindros 4, 6, 8) se mantenga
#   similar en train y test. Esto es crucial en datasets pequeños para que test sea representativo.
# - 'random_state=42' fija la semilla para reproducibilidad.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ---------- 3) Pipeline MLP (núcleo de la red neuronal) ----------
# Un PIPELINE encadena pasos: primero el escalado (StandardScaler), luego el MLP.
# ¿Por qué escalar?
# - El MLP entrena mediante gradiente; si las features tienen escalas muy distintas (hp ~ 100-300,
#   wt ~ 2-5, mpg ~ 10-35, etc.), los gradientes se "desbalancean" y el entrenamiento es ineficiente
#   o inestable. StandardScaler centra las variables (media 0) y las escala a desviación estándar 1.
#
# ¿Por qué un Pipeline y no escalar "a mano"?
# - Para evitar "data leakage" (fuga de información). El scaler se ajusta SOLO con X_train y luego
#   se aplica a X_train y X_test. El Pipeline lo garantiza automáticamente y además se integra
#   perfectamente con la validación cruzada (cada fold re-ajusta su scaler dentro del fold).
#
# ¿Qué es un MLPClassifier?
# - Es una red neuronal feed-forward básica con:
#   * Capa de entrada: tantas neuronas como features (se infiere de X).
#   * Capas ocultas: definidas por 'hidden_layer_sizes'.
#   * Capa de salida: tantas neuronas como clases (en mtcars, típicamente 3: 4, 6 y 8 cilindros).
# - Para clasificación multiclase, la capa de salida usa softmax internamente y entrena minimizando
#   la pérdida de entropía cruzada (log-loss).
#
# Explicación de hiperparámetros que usamos:
# - hidden_layer_sizes=(32,):
#     Una sola capa oculta con 32 neuronas. Más neuronas = más capacidad para modelar relaciones
#     no lineales; pero demasiadas neuronas en datasets pequeños pueden sobreajustar.
# - activation="relu":
#     Función de activación ReLU en las capas ocultas. Ayuda con gradientes más estables
#     (evita parte del "vanishing gradient" típico de sigmoide/tanh) y suele converger más rápido.
# - solver="adam":
#     Optimizador Adam (estocástico) que adapta la tasa de aprendizaje por parámetro usando momentos
#     del gradiente. Es robusto y funciona bien "out of the box". Alternativas:
#       * "lbfgs": optimizador cuasi-Newton, suele ir muy bien en datasets MUY pequeños,
#         pero no produce 'loss_curve_' ni soporta 'early_stopping'.
#       * "sgd": descenso por gradiente estocástico puro (necesita más tuning).
# - alpha=1e-3:
#     Regularización L2 (también llamada "weight decay"). Penaliza pesos grandes para reducir
#     sobreajuste. Si ves overfitting, sube alpha; si el modelo se queda corto, bájalo un poco.
# - learning_rate="adaptive" + learning_rate_init=1e-3:
#     Comenzamos con 0.001 y, si el entrenamiento se estanca, Adam puede adaptar internamente
#     la tasa efectiva. Con 'adaptive', MLPClassifier reduce la tasa si no mejora el loss en
#     un número de iteraciones consecutivas (cuando no usamos lbfgs).
# - max_iter=2000:
#     Límite máximo de épocas/iteraciones de entrenamiento. Con datos pequeños, a veces el
#     criterio de parada tarda; elevar este límite permite converger. Si ves "ConvergenceWarning",
#     sube max_iter o ajusta otros hiperparámetros.
# - early_stopping=False:
#     Si fuera True, el MLP separa internamente un "validation set" del propio entrenamiento y se
#     detiene cuando deja de mejorar ahí. En datasets MUY pequeños, esto puede quitar demasiados
#     datos al entrenamiento; por eso aquí lo ponemos en False. (Si tuvieras más datos, probar True
#     suele ser buena idea).
# - random_state=42:
#     Fija inicializaciones (pesos aleatorios, mezcla de datos) para reproducibilidad.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32,),    # 1 capa oculta con 32 neuronas (capacidad moderada)
        activation="relu",           # no linealidad en capas ocultas
        solver="adam",               # optimizador robusto y práctico en general
        alpha=1e-3,                  # regularización L2 (weight decay) para evitar sobreajuste
        learning_rate="adaptive",    # permite reducir la tasa si no hay mejora
        learning_rate_init=1e-3,     # tasa de aprendizaje inicial
        max_iter=2000,               # iteraciones máximas (suficiente para converger en datasets chicos)
        early_stopping=False,        # en datasets pequeños, mejor no restar muestras de train
        random_state=42              # reproducibilidad (pesos iniciales, orden de muestras)
    ))
])

# ---------- 4) Entrenar ----------
# Aquí ocurre la "magia": fit() hace dos cosas porque estamos en un Pipeline:
# 1) Ajusta el StandardScaler con X_train (calcula medias y desviaciones) y transforma X_train.
# 2) Con esos datos escalados, entrena la red MLP:
#    - Inicializa pesos de forma aleatoria (controlado por random_state).
#    - Hace forward pass (propagación hacia adelante) para obtener predicciones.
#    - Calcula la pérdida (log-loss) comparando con y_train (multiclase).
#    - Hace backpropagation (retropropagación del error) para calcular gradientes.
#    - Actualiza pesos con Adam (usa promedios móviles de gradientes y gradientes al cuadrado).
#    - Repite (épocas) hasta converger o alcanzar max_iter.
pipe.fit(X_train, y_train)

# ---------- 5) Evaluar en test ----------
# Usamos el Pipeline para predecir en X_test. Importante:
# - El scaler dentro del Pipeline usa LOS PARÁMETROS AJUSTADOS EN TRAIN (no se re-ajusta con test).
# - Luego, la red produce la clase con mayor probabilidad (softmax) para cada fila de test.
y_pred = pipe.predict(X_test)

# Accuracy: proporción de aciertos (rápido de interpretar, pero ojo con clases desbalanceadas).
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy (test): {acc:.3f}\n")

# Reporte de clasificación: precisión, recall, F1 por clase y promedios.
# - precision: de lo que predije como "clase k", ¿qué % fue correcto?
# - recall (sensibilidad): de los "verdaderos clase k", ¿qué % detecté?
# - F1: media armónica de precision y recall (balancea ambos).
print("Classification report (test):\n",
      classification_report(y_test, y_pred, zero_division=0,
                            target_names=[f"{c} cylinders" for c in sorted(y.unique())]))

# ---------- 6) Matriz de confusión ----------
# La matriz de confusión muestra conteos de (clase real vs predicha).
# Diagonal alta = buen desempeño. Fuera de diagonal = confusiones.
labels = sorted(y.unique())
conf_mat = confusion_matrix(y_test, y_pred, labels=labels)  # <- evitamos usar el nombre 'cm' por claridad
plt.figure(figsize=(5, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Matriz de confusión (conteos)")
plt.xlabel("Predicción"); plt.ylabel("Real")
plt.tight_layout()
plt.show()

# ---------- 7) Curva de pérdida (cómo baja el error durante el entrenamiento) ----------
# El MLP almacena 'loss_curve_' (pérdida por iteración) solo con solvers que iteran (adam/sgd).
# Esto ayuda a diagnosticar si:
# - La pérdida baja y se estabiliza (bien).
# - La pérdida oscila o no baja (tal vez LR es grande o faltan épocas).
# - La pérdida baja muy lento (quizá LR muy chico).
mlp_trained = pipe.named_steps["mlp"]
if hasattr(mlp_trained, "loss_curve_"):
    plt.figure(figsize=(6, 4))
    plt.plot(mlp_trained.loss_curve_)
    plt.xlabel("Iteraciones / Épocas")
    plt.ylabel("Loss (log-loss)")
    plt.title("Curva de pérdida del MLP")
    plt.tight_layout()
    plt.show()

# ---------- 8) (Opcional) Validación cruzada estratificada 5-fold ----------
# Para tener una métrica más "estable" con tan pocos datos, usamos CV:
# - StratifiedKFold preserva proporciones de clase en cada fold.
# - cross_val_score re-ajusta el Pipeline en cada fold (¡sin fugas!), devolviendo accuracy por fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
print(f"Accuracy CV 5-fold (media ± sd): {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

# =============================================================================
# ======================   PLOT DE LA RED (OPCIONAL)   ========================
# Las siguientes dos secciones son VISUALIZACIONES OPCIONALES para:
# (A) Dibujar la arquitectura de la red con conexiones coloreadas por peso.
# (B) Ver un heatmap de los pesos de entrada → capa oculta.
# Puedes comentarlas si solo te interesa el entrenamiento/evaluación.
# =============================================================================


def plot_mlp_network(pipe, feature_names=None, class_names=None,
                     edge_maxwidth=4.0, weight_threshold=0.05, figsize=(11, 6), dpi=200):
    """
    Dibuja la arquitectura y pesos del MLP entrenado dentro de un Pipeline.

    ¿Qué muestra?
    - Nodos por capa (entrada, ocultas, salida).
    - Aristas (conexiones) coloreadas por signo del peso y con grosor proporcional
      a su magnitud. Así se intuye qué features/caminos pesan más.

    Parámetros:
    - edge_maxwidth: grosor máximo de arista (se escala por |peso|).
    - weight_threshold: porcentaje del peso máximo por capa para filtrar conexiones
      MUY débiles (reduce ruido visual).
    - figsize: tamaño del gráfico.
    - dpi: resolución para una visualización más nítida.

    Nota: Esto usa los atributos 'coefs_' del MLP (lista de matrices de pesos W^(l)).
    Cada W tiene forma [n_l, n_{l+1}], es decir, pesos de capa L a capa L+1.
    """
    # Extraemos el MLP desde el Pipeline y sus matrices de pesos.
    mlp = pipe.named_steps["mlp"]
    coefs = mlp.coefs_
    # intercepts = mlp.intercepts_  # existen, pero aquí no se grafican.

    # Calculamos tamaños de cada capa:
    # - La capa de entrada tiene tantas neuronas como columnas de X.
    # - Luego, por cada matriz de pesos W, su dimensión .shape[1] es el número de neuronas en la capa siguiente.
    n_layers = len(coefs) + 1
    layer_sizes = [coefs[0].shape[0]] + [W.shape[1] for W in coefs]  # [n_in, ..., n_out]

    # Nombres default si no se pasan:
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(layer_sizes[0])]
    if class_names is None:
        class_names = [str(c) for c in mlp.classes_]

    # Posiciones (x, y) de nodos por capa para dibujar:
    xs = np.linspace(0, n_layers - 1, n_layers)
    positions = []
    for L, size in enumerate(layer_sizes):
        # Distribución vertical uniforme de nodos dentro de [0, 1] para esa capa:
        ys = np.linspace(0, 1, size)
        positions.append(list(zip([xs[L]] * size, ys)))

    # Preparar normalización de color en torno a 0 (negativos a un color, positivos a otro).
    allW = np.concatenate([W.ravel() for W in coefs]) if len(coefs) else np.array([0.0])
    vmax = np.max(np.abs(allW)) if np.max(np.abs(allW)) > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = mcm.get_cmap("coolwarm")

    # 🔧 Cambiamos la creación de la figura para poder controlar DPI y márgenes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title("Red neuronal (MLP): nodos y pesos aprendidos")
    ax.axis("off")

    # Deja un pequeño margen a la derecha para que las etiquetas de salida no queden pegadas.
    # (Además, como la barra de color irá en un eje separado, este margen ayuda a que nada se corte).
    ax.set_xlim(-0.2, (n_layers - 1) + 0.25)

    # Dibujar aristas (conexiones) capa a capa:
    for L, W in enumerate(coefs):
        absW = np.abs(W)
        thr = weight_threshold * absW.max() if absW.size else 0.0
        for i in range(W.shape[0]):       # neurona i en capa L
            x0, y0 = positions[L][i]
            for j in range(W.shape[1]):   # neurona j en capa L+1
                w = W[i, j]
                if np.abs(w) < thr:
                    continue  # filtra pesos muy pequeños para evitar "spaghetti"
                x1, y1 = positions[L + 1][j]
                # Grosor proporcional a |w| y un mínimo para que se vea:
                lw = max(0.3, edge_maxwidth * (np.abs(w) / vmax))
                ax.plot([x0, x1], [y0, y1],
                        color=cmap(norm(w)), linewidth=lw, alpha=0.8)

    # Dibujar nodos de cada capa:
    for L, coords in enumerate(positions):
        xsL, ysL = zip(*coords)
        ax.scatter(xsL, ysL, s=250, c="#f0f0f0", edgecolors="#444", zorder=3)
        # Etiquetas para capa de entrada (features):
        if L == 0:
            for idx, (x, y_) in enumerate(coords):
                ax.text(x - 0.05, y_, feature_names[idx], ha="right", va="center", fontsize=8)
        # Etiquetas para capa de salida (clases):
        if L == n_layers - 1:
            for idx, (x, y_) in enumerate(coords):
                ax.text(x + 0.05, y_, class_names[idx], ha="left", va="center",
                        fontsize=9, fontweight="bold")

     # -------------------------------------------------------------------------
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    # 'size' controla el ancho de la barra; 'pad' la separación respecto al eje principal.
    cax = divider.append_axes("right", size="3%", pad=0.5)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Peso (signo y magnitud)")

    # Ajuste final para que nada se corte (respeta el eje extra del colorbar).
    fig.tight_layout()
    plt.show()

# ---- Llamada del diagrama de la red (opcional) ----
feature_names = X.columns.tolist()
class_names = [f"{int(c)} cylinders" for c in sorted(y.unique())]
plot_mlp_network(pipe, feature_names=feature_names, class_names=class_names,
                 edge_maxwidth=4.0, weight_threshold=0.05, figsize=(11.5, 6.5), dpi=200)

# ---- Heatmap de pesos entrada → capa oculta (opcional) ----
# Visualización complementaria: muestra, en una matriz, qué tan fuerte (y con qué signo)
# conecta cada feature con cada neurona de la capa oculta. Útil para interpretar tendencias:
# - Pesos positivos grandes: esa feature "empuja" la activación de esa neurona hacia arriba.
# - Pesos negativos grandes: la "frenan".
W_in = pipe.named_steps["mlp"].coefs_[0]  # matriz [n_features, n_hidden]
dfW = pd.DataFrame(W_in, index=feature_names,
                   columns=[f"h{j+1}" for j in range(W_in.shape[1])])

plt.figure(figsize=(min(12, 1.2 * W_in.shape[1]), 6))
sns.heatmap(dfW, cmap="coolwarm", center=0, robust=True)
plt.title("Pesos entrada → capa oculta (signo y magnitud)")
plt.xlabel("Neuronas ocultas"); plt.ylabel("Features")
plt.tight_layout(); plt.show()

# =============================================================================
# SUGERENCIAS PRÁCTICAS (para que sepas "qué tocar" si algo no converge o no rinde):
#
# - Si ves "ConvergenceWarning" (no converge):
#     * Sube max_iter (p. ej., 5000) o
#     * Baja learning_rate_init (p. ej., 5e-4) o
#     * Cambia solver a "lbfgs" (en datasets muy chicos puede converger más rápido).
#
# - Si rinde bien en train pero mal en test (overfitting):
#     * Sube alpha (p. ej., 1e-2 o 1e-1).
#     * Reduce hidden_layer_sizes (menos neuronas o menos capas).
#     * Usa early_stopping=True si tienes más datos (separa un validation interno).
#
# - Si rinde mal en train y test (underfitting):
#     * Baja alpha (p. ej., 1e-4).
#     * Aumenta hidden_layer_sizes (más neuronas/capas).
#     * Cambia activation a 'tanh' (a veces captura relaciones distintas).
#
# - Para comparar con otros modelos:
#     * Prueba 'lbfgs' como solver (rápido en datasets pequeños, pero sin loss_curve_).
#     * Prueba modelos no neurales (LogisticRegression/RandomForest) como baseline.
#
# Recuerda: este dataset es muy pequeño (32 filas). Los resultados pueden variar con splits distintos.
# =============================================================================

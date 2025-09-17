# -*- coding: utf-8 -*-
# =========================================================================================
#                               GUÍA COMENTADA PASO A PASO
#                              Por Humberto Silva Baltazar
# =========================================================================================
# Objetivo:
# --------------------
# Este script es una clase práctica sobre dos enfoques de clasificación:
#   1) K-Nearest Neighbors (KNN) con selección de k mediante validación cruzada (CV).
#   2) Una red neuronal multicapa (MLP) entrenada "por épocas" con early stopping manual.
#
# Lo que aprenderás:
# - Qué es un split Train/Test y por qué se usa.
# - Por qué escalamos variables (StandardScaler) y cómo evitar fugas de información
#   con un Pipeline (en KNN) y con un "fit en train / transform en valid/test" (en MLP).
# - Cómo seleccionar hiperparámetros (k de KNN) con CV de manera honesta.
# - Cómo se entrena un MLP de scikit-learn por épocas, aunque el estimador no expone
#   épocas explícitas: usamos max_iter=1 + warm_start=True para "simular" épocas.
# - Qué es el early stopping, por qué es útil (evitar sobreajuste) y cómo implementarlo
#   monitoreando una métrica de validación (aquí: log_loss en validación).
# - Cómo evaluar un clasificador con accuracy, F1 (por clase y agregados), ROC-AUC y PR-AUC.
# - Cómo visualizar la arquitectura del MLP con sus pesos/sesgos sin que se "encimen" etiquetas.
#
# Notas didácticas clave sobre redes neuronales:
# - Una red neuronal multicapa (MLP) compone funciones lineales (multiplicaciones de matrices
#   W * x + b) con funciones no lineales (activaciones como ReLU). Lo esencial es multiplicar
#   matrices y sumar sesgos, capa por capa. Eso mismo, a gran escala, es la base de los LLM:
#   montones de multiplicaciones de matrices optimizadas en GPU/TPU.
# - Entrenamos ajustando W y b para reducir una función de pérdida: aquí usamos Adam como
#   optimizador interno y monitoreamos log_loss de validación para decidir cuándo detenernos.
#
# ------------------
# =========================================================================================

import warnings
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, log_loss
)
from sklearn.exceptions import ConvergenceWarning

from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

# ------------------------ #
# Configuración de gráficos
# ------------------------ #
# Usamos un estilo amable para cuadernos y alta densidad de puntos por figura.
sns.set(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# Silenciamos advertencias de convergencia de MLP para no distraer.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------------ #
# Carga de datos
# ------------------------ #
# Dataset clásico de cáncer de mama (Breast Cancer Wisconsin), binario:
#   target 0 = malignant (maligno), 1 = benign (benigno).
# Elegimos este dataset porque:
# - Está limpio.
# - Tiene suficientes observaciones para demostrar KNN y MLP.
# - Variables numéricas → escalado tiene sentido.
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target
df["target_name"] = df["target"].map({0: "malignant", 1: "benign"})

print("--- DATASET ---")
print("Primeras 5 filas:")
print(df.head(), "\n")

# ------------------------ #
# Split Train/Test global
# ------------------------ #
# Separar en entrenamiento y prueba es fundamental para medir desempeño "fuera de muestra".
# - test_size=0.3: 30% de datos para test.
# - stratify=y: mantiene proporción de clases en train y test (estratificado).
X = df.drop(columns=["target", "target_name"])
y = df["target"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =============================================================================
# 1) KNN: Selección de k con validación cruzada + evaluación en Test
# =============================================================================
print("\n--- MODELO KNN (con selección de k) ---")

# Por qué un rango de k IMPAR:
# - KNN decide por "voto" de vecinos. Con k impar reducimos empates entre clases.
k_values = list(range(1, 31, 2))

# Validación cruzada estratificada (5 folds), barajando datos.
# Motivo: estimar desempeño de manera robusta en el conjunto de entrenamiento,
# evitando depender de un único split.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []
for k in k_values:
    # Pipeline = forma correcta de encadenar preprocesamiento y modelo:
    # StandardScaler "dentro" de cada fold (fit solo en train del fold; transform en valid del fold)
    # evita fuga de información (data leakage).
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    acc_scores, f1_scores = [], []
    for tr_idx, va_idx in cv.split(X_train_full, y_train_full):
        # Dividimos manualmente en índices de train y valid del fold
        X_tr, X_va = X_train_full.iloc[tr_idx], X_train_full.iloc[va_idx]
        y_tr, y_va = y_train_full.iloc[tr_idx], y_train_full.iloc[va_idx]

        # Ajustamos SOLO con el fold-train; transform + predicción en fold-valid
        pipe.fit(X_tr, y_tr)
        y_va_pred = pipe.predict(X_va)

        # Métricas en validación del fold
        acc_scores.append(accuracy_score(y_va, y_va_pred))
        # Por defecto f1_score toma pos_label=1 (benign).
        # Usamos F1 como criterio principal porque balancea precisión y exhaustividad.
        f1_scores.append(f1_score(y_va, y_va_pred))

    # Guardamos promedios por k para luego elegir el "mejor"
    cv_results.append({
        "k": k,
        "cv_accuracy_mean": np.mean(acc_scores),
        "cv_f1_mean": np.mean(f1_scores)
    })

# Convertimos los resultados a DataFrame para ordenar y visualizar
cv_df = pd.DataFrame(cv_results)

# Criterio de selección:
#   1) Máximo F1 (pos=1).
#   2) En caso de empate, mayor Accuracy.
#   3) Luego, el k más pequeño (simplicidad).
best_row = cv_df.sort_values(
    by=["cv_f1_mean", "cv_accuracy_mean", "k"],
    ascending=[False, False, True]
).iloc[0]
best_k = int(best_row["k"])
print(f"Mejor k (por F1 en CV): {best_k}")
print(cv_df.sort_values(by='k').to_string(index=False))

# Gráfica de desempeño vs k → nos ayuda a ver la tendencia general.
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cv_df["k"], cv_df["cv_accuracy_mean"], marker="o", label="Accuracy (CV)")
ax.plot(cv_df["k"], cv_df["cv_f1_mean"], marker="s", label="F1 (CV, pos=benign)")
ax.axvline(best_k, linestyle="--", alpha=0.8, label=f"Mejor k = {best_k}")
ax.set_xlabel("k (n_neighbors)")
ax.set_ylabel("Métrica media (5-fold CV)")
ax.set_title("KNN: desempeño en validación cruzada vs k")
ax.legend()
plt.tight_layout()
plt.show()

# Re-entrenamos KNN con el mejor k usando TODO el train_full (sin tocar test).
# Este es el modelo final para evaluar en test.
pipe_best_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])
pipe_best_knn.fit(X_train_full, y_train_full)

# Predicción en Test (clase predicha y probabilidades)
y_pred_knn = pipe_best_knn.predict(X_test)
y_proba_knn = pipe_best_knn.predict_proba(X_test)[:, 1]  # prob(clase 1 = benign)

# Métricas para test (lo que realmente nos importa para "generalizar")
acc_knn = accuracy_score(y_test, y_pred_knn)
# F1 explícito por clase para interpretar el desempeño en cada etiqueta
f1_knn_benign = f1_score(y_test, y_pred_knn, pos_label=1)
f1_knn_malign = f1_score(y_test, y_pred_knn, pos_label=0)
f1_knn_macro  = f1_score(y_test, y_pred_knn, average="macro")
f1_knn_weight = f1_score(y_test, y_pred_knn, average="weighted")

# Calculamos métricas de área bajo curvas (sin graficarlas):
# - ROC-AUC mide capacidad de ranking por umbrales.
# - PR-AUC (Average Precision) es útil cuando hay desbalance.
roc_auc_knn = roc_auc_score(y_test, y_proba_knn)
ap_knn      = average_precision_score(y_test, y_proba_knn)

print("\n--- Evaluación en TEST (KNN) ---")
print(f"Accuracy (Test):         {acc_knn:.4f}")
print(f"F1 benign (1, default):  {f1_knn_benign:.4f}")
print(f"F1 malignant (0):        {f1_knn_malign:.4f}")
print(f"F1 macro:                {f1_knn_macro:.4f}")
print(f"F1 weighted:             {f1_knn_weight:.4f}")
print(f"ROC-AUC (Test):          {roc_auc_knn:.4f}")
print(f"PR-AUC  (Test, pos=1):   {ap_knn:.4f}")

# Matriz de confusión de KNN para ver aciertos/errores por clase.
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("\nMatriz de Confusión (KNN):")
print(cm_knn)
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(
    cm_knn, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=["malignant", "benign"],
    yticklabels=["malignant", "benign"]
)
plt.xlabel("Predicción"); plt.ylabel("Real")
plt.title("Matriz de confusión - KNN")
plt.tight_layout(); plt.show()

# Reporte de clasificación con precision/recall/F1 por clase.
print("\nInforme de Clasificación (KNN):\n",
      classification_report(y_test, y_pred_knn, target_names=cancer.target_names))

# =============================================================================
# 2) RED NEURONAL (MLP) con entrenamiento por épocas + early stopping
# =============================================================================
print("\n--- RED NEURONAL (MLP) ---")

# Hacemos un split interno de validación a partir del Train_Full.
# ¿Por qué? Necesitamos un conjunto de "validación" para:
# - monitorear log_loss y decidir el early stopping,
# - ajustar hiperparámetros si quisiéramos (aquí no lo hacemos).
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Escalado para MLP:
# - fit SOLO en train
# - transform en valid y test
# Motivo: el MLP es sensible a la escala; esta práctica evita fuga de información.
scaler_mlp = StandardScaler()
X_train_mlp = scaler_mlp.fit_transform(X_train)
X_valid_mlp = scaler_mlp.transform(X_valid)
X_test_mlp  = scaler_mlp.transform(X_test)

# Configuración de la red:
# - hidden_layer_sizes=(64, 32, 16): tres capas ocultas con 64, 32 y 16 neuronas.
# - activation="relu": no linealidad típica, ayuda a modelar relaciones complejas.
# - solver="adam": optimizador robusto en práctica (descenso estocástico adaptativo).
# - alpha=1e-4: regularización L2, ayuda a controlar sobreajuste penalizando pesos grandes.
# - learning_rate_init=1e-3: paso de aprendizaje inicial.
# - max_iter=1 + warm_start=True: entrenamos "una época" por .fit(...) y luego seguimos.
#   Así podemos controlar manualmente épocas y aplicar early stopping externo.
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-4,             # L2
    batch_size=32,
    learning_rate_init=1e-3,
    max_iter=1,             # una "época" por fit()
    warm_start=True,        # continuar entrenando en la siguiente llamada a fit
    shuffle=True,
    random_state=42,
    verbose=False
)

# Entrenamiento por épocas con early stopping (monitor: val_loss).
# - n_epochs: máximo de épocas a intentar.
# - patience: cuántas épocas seguidas toleramos sin mejora en val_loss.
# - min_delta: mejora mínima requerida para considerar que "sí mejoró".
n_epochs = 200
patience = 20
min_delta = 1e-3            # mejora mínima para actualizar "mejor modelo"
best_val_loss = np.inf
epochs_no_improve = 0
best_mlp = None

# Historial SOLO de accuracy de validación, tal como se solicita (sin pérdida).
history = {
    "epoch": [],
    "val_acc": []
}

for epoch in range(1, n_epochs + 1):
    # Una "época" de entrenamiento. Gracias a warm_start=True y max_iter=1,
    # cada llamada a fit avanza un poco el estado del optimizador interno.
    mlp.fit(X_train_mlp, y_train)

    # Obtenemos probabilidades (salida sigmoide/softmax de la última capa).
    # En binario, [:, 1] es la prob. de la clase positiva (aquí, 1=benign).
    proba_train = mlp.predict_proba(X_train_mlp)[:, 1]
    proba_valid = mlp.predict_proba(X_valid_mlp)[:, 1]

    # Medimos log_loss en train y valid.
    # Por qué log_loss: es una pérdida "suave" para clasificación probabilística,
    # sensible a la calibración de probabilidades.
    loss_tr = log_loss(y_train, proba_train, labels=[0, 1])
    loss_va = log_loss(y_valid, proba_valid, labels=[0, 1])

    # Para la única gráfica de entrenamiento pedida: Accuracy en validación.
    # Nota: NO optimizamos umbral, usamos 0.5 fijo por consigna.
    y_valid_pred = (proba_valid >= 0.5).astype(int)
    acc_va = accuracy_score(y_valid, y_valid_pred)

    history["epoch"].append(epoch)
    history["val_acc"].append(acc_va)

    # Early stopping: si la pérdida de validación no mejora al menos min_delta,
    # contamos una época sin mejora. Si acumulamos "patience" → detenemos.
    improved = loss_va < (best_val_loss - min_delta)
    if improved:
        best_val_loss = loss_va
        best_mlp = copy.deepcopy(mlp)  # guardamos copia "mejor hasta ahora"
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping en época {epoch} (sin mejora por {patience} épocas, min_delta={min_delta}).")
            break

# Usamos el mejor modelo encontrado durante el entrenamiento (si hubo mejoras).
if best_mlp is not None:
    mlp = best_mlp

# Convertimos historial a DataFrame para graficar la curva de accuracy en validación.
hist_df = pd.DataFrame(history)

# ÚNICA curva pedida: Accuracy de validación por época.
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(hist_df["epoch"], hist_df["val_acc"], label="Accuracy (valid)")
ax.set_xlabel("Época"); ax.set_ylabel("Accuracy")
ax.set_title("MLP: Accuracy de validación por época")
ax.legend()
plt.tight_layout(); plt.show()

# Evaluación final en TEST (MLP):
# - Usamos el modelo "mejor" según val_loss.
# - Predicción con umbral fijo 0.5 (sin optimizar umbral para malignant).
proba1_test = mlp.predict_proba(X_test_mlp)[:, 1]  # prob(benign=1)
y_test_pred_mlp = (proba1_test >= 0.5).astype(int)

# Métricas en test para interpretar desempeño real fuera de muestra.
acc_te = accuracy_score(y_test, y_test_pred_mlp)
f1_te_benign = f1_score(y_test, y_test_pred_mlp, pos_label=1)
f1_te_malign = f1_score(y_test, y_test_pred_mlp, pos_label=0)
f1_te_macro  = f1_score(y_test, y_test_pred_mlp, average="macro")
f1_te_weight = f1_score(y_test, y_test_pred_mlp, average="weighted")

# Métricas de área (sin graficar curvas).
roc_auc = roc_auc_score(y_test, proba1_test)            # ROC-AUC
ap     = average_precision_score(y_test, proba1_test)   # PR-AUC (AP)

print("\n--- Evaluación en TEST (MLP) ---")
print(f"Accuracy (Test):         {acc_te:.4f}")
print(f"F1 benign (1, default):  {f1_te_benign:.4f}")
print(f"F1 malignant (0):        {f1_te_malign:.4f}")
print(f"F1 macro:                {f1_te_macro:.4f}")
print(f"F1 weighted:             {f1_te_weight:.4f}")
print(f"ROC-AUC (Test, pos=1):   {roc_auc:.4f}")
print(f"PR-AUC  (Test, pos=1):   {ap:.4f}")

# Matriz de confusión del MLP:
cm_mlp = confusion_matrix(y_test, y_test_pred_mlp)
print("\nMatriz de Confusión (MLP):")
print(cm_mlp)
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(
    cm_mlp, annot=True, fmt="d", cmap="Blues", cbar=False,
    xticklabels=["malignant", "benign"],
    yticklabels=["malignant", "benign"]
)
plt.xlabel("Predicción"); plt.ylabel("Real")
plt.title("Matriz de confusión - MLP")
plt.tight_layout(); plt.show()

# Reporte de clasificación con métricas por clase y agregados.
print("\nInforme de Clasificación (MLP):\n",
      classification_report(y_test, y_test_pred_mlp, target_names=cancer.target_names))

# =============================================================================
# 3) Gráfica de la RED NEURONAL entrenada (diagrama de arquitectura + pesos)
#    -> Ajustada para que NOMBRES/LEYENDAS/INDICADORES NO SE ENCIMEN
# =============================================================================
def plot_mlp_network(mlp_model, feature_names=None, node_size=140,
                     lw_min=0.2, lw_max=3.5, alpha_edges=0.75,
                     cmap_name="coolwarm", show_input_labels=True):
    """
    Diagrama pedagógico de la red MLP:
    ----------------------------------
    Qué hace:
      - Dibuja las capas (entrada, ocultas y salida) como columnas de nodos.
      - Dibuja conexiones entre capas con color por signo del peso (negativo/positivo)
        y grosor proporcional a su magnitud |w|.
      - Colorea nodos según el valor del sesgo b.

    Cómo lo hace:
      1) Extrae coefs_ (pesos) e intercepts_ (sesgos) del MLP ya entrenado.
      2) Posiciona nodos por capa con coordenadas (x,y) equiespaciadas.
      3) Normaliza pesos/sesgos (TwoSlopeNorm) para mapear a un colormap centrado en 0.
      4) Dibuja segmentos entre nodos de capas consecutivas con LineCollection.
      5) Añade colorbars externas para pesos y sesgos y ajusta márgenes para evitar solapes.

    Por qué así:
      - Visualizar W y b ayuda a entender “qué tanto” conecta cada neurona.
      - Separar colorbars y bajar etiquetas evita que leyendas/nombres se encimen.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    # 1) Extraemos pesos (coefs_) y sesgos (intercepts_) del modelo entrenado.
    coefs = mlp_model.coefs_
    intercepts = mlp_model.intercepts_

    # Tamaños de capas: entrada → ocultas → salida
    n_layers = len(coefs) + 1
    n_inputs = coefs[0].shape[0]
    n_outputs = coefs[-1].shape[1]
    hidden_sizes = [w.shape[1] for w in coefs[:-1]]
    layer_sizes = [n_inputs] + hidden_sizes + [n_outputs]

    # 2) Coordenadas de nodos:
    # - x equiespaciado por capa.
    # - y equiespaciado por nodo (de arriba hacia abajo).
    xs = np.linspace(0.10, 0.90, n_layers)
    coords = []
    for L, n_nodes in enumerate(layer_sizes):
        ys = np.array([0.5]) if n_nodes == 1 else np.linspace(0.90, 0.10, n_nodes)
        coords.append(np.column_stack([np.full(n_nodes, xs[L]), ys]))

    # 3) Normalización robusta de pesos y sesgos (percentil 99) para no saturar colormap.
    all_w = np.concatenate([w.ravel() for w in coefs])
    w_abs = np.abs(all_w)
    w_max = np.percentile(w_abs, 99) if np.any(w_abs) else 1.0

    all_b = np.concatenate([b.ravel() for b in intercepts]) if len(intercepts) else np.array([0.0])
    b_abs = np.abs(all_b)
    b_max = np.percentile(b_abs, 99) if np.any(b_abs) else 1.0

    cmap  = plt.get_cmap(cmap_name)
    wnorm = TwoSlopeNorm(vmin=-w_max, vcenter=0.0, vmax=w_max)
    bnorm = TwoSlopeNorm(vmin=-b_max, vcenter=0.0, vmax=b_max)

    # 4) Figura con márgenes holgados; espacio derecho para colorbars (evita solapes).
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.subplots_adjust(left=0.32, right=0.80, top=0.90, bottom=0.08)
    fig.suptitle("Diagrama de la red neuronal (MLP) con pesos", y=0.965, fontsize=13)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # 5) Dibujamos las conexiones capa a capa con color por signo y grosor por magnitud.
    for L in range(len(coefs)):
        W = coefs[L]
        in_xy, out_xy = coords[L], coords[L + 1]
        segments, colors, widths = [], [], []
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = W[i, j]
                segments.append([tuple(in_xy[i]), tuple(out_xy[j])])
                colors.append(cmap(wnorm(w)))
                widths.append(lw_min + (min(abs(w), w_max) / w_max) * (lw_max - lw_min))
        lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=alpha_edges, zorder=1)
        ax.add_collection(lc)

    # 6) Dibujamos nodos y etiquetas de capa (ligeramente más bajas para no tocar el título).
    for L, (xy, n_nodes) in enumerate(zip(coords, layer_sizes)):
        biases = np.zeros(n_nodes) if L == 0 else intercepts[L - 1]
        node_colors = cmap(bnorm(biases))
        ax.scatter(xy[:, 0], xy[:, 1], s=node_size, c=node_colors, edgecolor="k", zorder=2)

        # Etiqueta de cada capa:
        if L == 0:
            label = f"Entrada ({n_nodes})"
        elif L == len(layer_sizes) - 1:
            label = f"Salida ({n_nodes})"
        else:
            label = f"Capa oculta {L} ({n_nodes})"

        ax.text(xy[0, 0], 0.97, label, ha="center", va="top",
                fontsize=11, weight="bold", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95))

        # Etiquetas de variables de entrada con fondo blanco (mejor legibilidad).
        if L == 0 and show_input_labels:
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(n_nodes)]
            if n_nodes <= 30:  # si hay demasiadas, se puede ocultar para no saturar
                for i, name in enumerate(feature_names):
                    ax.text(xy[i, 0] - 0.07, xy[i, 1], name,
                            ha="right", va="center", fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="white", ec="none", alpha=0.95))

    # 7) Colorbars separadas (pesos y sesgos) con gap amplio para que no se encimen.
    cax1 = fig.add_axes([0.84, 0.15, 0.022, 0.70])   # pesos w
    cax2 = fig.add_axes([0.92, 0.15, 0.022, 0.70])   # sesgos b (más separada)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=wnorm, cmap=cmap), cax=cax1)
    cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=bnorm, cmap=cmap), cax=cax2)
    cb1.set_label("Peso w (negativo ↔ positivo)", rotation=90, labelpad=10, fontsize=9)
    cb2.set_label("Sesgo b (negativo ↔ positivo)", rotation=90, labelpad=10, fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    cb2.ax.tick_params(labelsize=8)

    plt.show()


def plot_mlp_weight_heatmaps(mlp_model, vmax_percentile=99, cmap_name="coolwarm"):
    """
    Heatmaps de pesos por capa:
    ---------------------------
    Qué hace:
      - Para cada matriz de pesos W_L (de capa L a L+1), muestra un mapa de calor.
        Filas = neuronas de entrada (capa L), columnas = neuronas de salida (capa L+1).

    Cómo lo hace:
      1) Reúne todos los pesos para calcular un máximo robusto (percentil 99).
      2) Usa TwoSlopeNorm centrado en 0 (colores fríos para negativos, cálidos para positivos).
      3) Muestra un subplot por capa con su barra de color.

    Por qué así:
      - Es una forma compacta de visualizar patrones de conectividad "densos".
      - Centrar en 0 es intuitivo: negativo ↔ azul, positivo ↔ rojo (o similar según cmap).
    """
    coefs = mlp_model.coefs_
    nL = len(coefs)

    all_w = np.concatenate([w.ravel() for w in coefs])
    wmax = np.percentile(np.abs(all_w), vmax_percentile) if np.any(all_w) else 1.0
    norm = TwoSlopeNorm(vmin=-wmax, vcenter=0.0, vmax=wmax)
    cmap = plt.get_cmap(cmap_name)

    fig, axes = plt.subplots(1, nL, figsize=(4.5 * nL, 5), squeeze=False)
    axes = axes[0]

    for L, W in enumerate(coefs):
        ax = axes[L]
        im = ax.imshow(W, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"Pesos W{L}: {W.shape[0]} → {W.shape[1]}")
        ax.set_xlabel("Neuronas capa destino")
        ax.set_ylabel("Neuronas capa origen")
        if W.shape[1] == 1:  # si es matriz columna, quitamos ticks para que no se vea "vacío"
            ax.set_xticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# Llamadas para graficar la red entrenada
# ---------------------------------------
# Recomendación: si ves muchas etiquetas de entrada, considera show_input_labels=False.
plot_mlp_network(
    mlp_model=mlp,
    feature_names=X.columns.tolist(),
    node_size=130,
    lw_min=0.2,
    lw_max=3.0,
    alpha_edges=0.65,
    show_input_labels=True  # ponlo en False si prefieres sin etiquetas de entrada
)

# Heatmaps de pesos por capa para otra perspectiva visual.
plot_mlp_weight_heatmaps(mlp_model=mlp, vmax_percentile=99)

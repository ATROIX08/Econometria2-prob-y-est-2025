# -*- coding: utf-8 -*-
"""
Naive Bayes de Texto (TF-IDF) — Script didáctico y dinámico
-----------------------------------------------------------
Objetivo: mostrar, paso a paso, cómo entrenar y evaluar un clasificador
Multinomial Naive Bayes para análisis de sentimiento (negative/neutral/positive)
usando títulos/noticias en inglés.

Qué hace el script:
1) Carga un CSV sin encabezados (col0 = 'sentiment', col1 = 'text')
2) Limpia datos y filtra clases válidas
3) Divide en train/test de forma estratificada
4) Entrena un pipeline TF-IDF + MultinomialNB
5) Evalúa con accuracy, classification report y matriz de confusión
6) Grafica y GUARDA automáticamente en la carpeta indicada:
   - Distribución de clases
   - Matriz de confusión (conteos y porcentajes)
   - Barras de Precision/Recall/F1 por clase (con etiquetas de valor)
   - Barras de F2 por clase (con etiquetas de valor)
   - Top tokens (palabras más características) por clase según Naive Bayes
7) Muestra ejemplos de predicción con probabilidades para comprensión

Solo cambia la ruta del archivo en DATA_PATH. Las imágenes se guardan en IMG_DIR.

Requisitos:
- Python 3.8+
- pandas, numpy, matplotlib
- scikit-learn
"""

# ==========================
# IMPORTS
# ==========================
import os
import re
import math
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# Opcional: silenciar warnings de matplotlib/font para clase
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================
# CONFIG (EDITA AQUÍ LA RUTA DEL CSV SI LO REQUIERES)
# ==========================
DATA_PATH = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\prob-y-est\data\2 datasets de finanzas.csv"

# Carpeta donde se guardarán TODAS las imágenes
IMG_DIR = Path(r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\prob-y-est\images")

# Clases objetivo (si el dataset trae otras, se filtrarán)
VALID_LABELS = ['negative', 'neutral', 'positive']

# Semilla y split
RANDOM_STATE = 42
TEST_SIZE = 0.20  # 20% test

# Vectorizador (TF-IDF) — parámetros didácticos y fáciles de ajustar
NGRAM_RANGE = (1, 2)       # unigrams y bigrams ayudan a capturar frases cortas
MIN_DF = 2                 # ignora términos que aparecen en menos de 2 documentos
MAX_DF = 0.95              # ignora términos demasiado frecuentes
USE_IDF = True
SUBLINEAR_TF = True
LOWERCASE = True
STOP_WORDS = 'english'     # el texto está en inglés
STRIP_ACCENTS = 'unicode'  # normaliza acentos

# Visualización
TOP_N_TOKENS = 15          # cuántas palabras top por clase mostrar
FIG_DPI = 110


# ==========================
# UTILIDADES
# ==========================
def ensure_img_dir() -> None:
    """
    Crea la carpeta de imágenes si no existe.
    """
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    """
    Convierte un texto en un nombre de archivo seguro.
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\-.]+", "_", text)
    return text


def save_fig(fig: plt.Figure, filename: str) -> None:
    """
    Guarda la figura en IMG_DIR con el nombre dado.
    """
    ensure_img_dir()
    filepath = IMG_DIR / filename
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches="tight")
    print(f"[Guardado] {filepath}")


def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga el CSV sin encabezados. La 1a col = 'sentiment', 2a col = 'text'.
    Incluye tolerancia de encoding para evitar errores comunes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo en la ruta:\n{path}")

    encodings_to_try = ['latin1', 'utf-8', 'utf-8-sig']
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc, header=None, names=['sentiment', 'text'])
            return df
        except Exception as e:
            last_err = e
            continue
    # Si llegamos aquí, fallaron todos los intentos
    raise RuntimeError(
        f"No se pudo leer el CSV con encodings {encodings_to_try}. "
        f"Último error: {last_err}"
    )


def clean_and_filter(df: pd.DataFrame, valid_labels: List[str]) -> pd.DataFrame:
    """
    - Elimina filas con NaN en 'sentiment' o 'text'
    - Filtra por etiquetas válidas
    - Limpia espacios
    """
    df = df.copy()
    df.dropna(subset=['sentiment', 'text'], inplace=True)
    df['sentiment'] = df['sentiment'].astype(str).str.strip()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['sentiment'].isin(valid_labels)]

    if len(df) == 0:
        raise ValueError(
            "No hay datos con las clases válidas. "
            f"Se esperaban etiquetas en {valid_labels}."
        )
    return df


def plot_class_distribution(df: pd.DataFrame, label_col: str = 'sentiment') -> None:
    """
    Grafica la distribución (conteo) de clases para contexto en clase.
    """
    counts = df[label_col].value_counts().reindex(VALID_LABELS, fill_value=0)
    fig = plt.figure(figsize=(6, 4), dpi=FIG_DPI)
    bars = plt.bar(counts.index, counts.values)
    plt.title("Distribución de Clases (conteos)")
    plt.xlabel("Clase")
    plt.ylabel("Número de ejemplos")

    # Etiquetas de valor encima de cada barra
    for i, bar in enumerate(bars):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 0.01*max(1, counts.max()),
                 str(int(h)), ha='center', va='bottom')
    plt.tight_layout()

    # Guardar y mostrar
    save_fig(fig, "01_distribucion_clases.png")
    plt.show()
    plt.close(fig)


def build_pipeline() -> Pipeline:
    """
    Crea el pipeline TF-IDF + MultinomialNB con parámetros de CONFIG.
    """
    vect = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        use_idf=USE_IDF,
        sublinear_tf=SUBLINEAR_TF,
        lowercase=LOWERCASE,
        stop_words=STOP_WORDS,
        strip_accents=STRIP_ACCENTS
    )
    nb = MultinomialNB()
    pipe = Pipeline([
        ('tfidf', vect),
        ('nb', nb)
    ])
    return pipe


def compute_and_plot_confusion(y_true: List[str],
                               y_pred: List[str],
                               labels: List[str]) -> None:
    """
    Muestra matriz de confusión en conteos y porcentajes por fila.
    Guarda ambas figuras.
    """
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_perc = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)

    # Figura 1: Conteos
    fig1 = plt.figure(figsize=(6, 5), dpi=FIG_DPI)
    plt.imshow(cm_counts, interpolation='nearest')
    plt.title("Matriz de Confusión (conteos)")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = cm_counts.max() / 2.0 if cm_counts.max() > 0 else 0.5

    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            val = cm_counts[i, j]
            plt.text(j, i, str(val),
                     ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.ylabel("Etiqueta verdadera")
    plt.xlabel("Etiqueta predicha")
    plt.tight_layout()
    save_fig(fig1, "02_matriz_confusion_conteos.png")
    plt.show()
    plt.close(fig1)

    # Figura 2: Porcentajes por fila
    fig2 = plt.figure(figsize=(6, 5), dpi=FIG_DPI)
    plt.imshow(cm_perc, interpolation='nearest')
    plt.title("Matriz de Confusión (porcentaje por clase verdadera)")
    plt.colorbar()
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh2 = cm_perc.max() / 2.0 if cm_perc.max() > 0 else 0.5

    for i in range(cm_perc.shape[0]):
        for j in range(cm_perc.shape[1]):
            val = cm_perc[i, j]
            plt.text(j, i, f"{val:.2f}",
                     ha="center", va="center",
                     color="white" if val > thresh2 else "black")
    plt.ylabel("Etiqueta verdadera")
    plt.xlabel("Etiqueta predicha")
    plt.tight_layout()
    save_fig(fig2, "03_matriz_confusion_porcentaje.png")
    plt.show()
    plt.close(fig2)


def plot_prf_bars(report_dict: dict, class_order: List[str]) -> None:
    """
    Grafica barras de Precision/Recall/F1 por clase, con etiquetas de valor.
    """
    class_labels = [c for c in class_order if c in report_dict]

    precision_vals = [report_dict[c]['precision'] for c in class_labels]
    recall_vals    = [report_dict[c]['recall']    for c in class_labels]
    f1_vals        = [report_dict[c]['f1-score']  for c in class_labels]

    x = np.arange(len(class_labels))
    bar_width = 0.25

    fig = plt.figure(figsize=(8, 5), dpi=FIG_DPI)
    bars1 = plt.bar(x - bar_width, precision_vals, bar_width, label='Precision')
    bars2 = plt.bar(x,            recall_vals,    bar_width, label='Recall')
    bars3 = plt.bar(x + bar_width, f1_vals,       bar_width, label='F1-Score')

    plt.xlabel('Clases')
    plt.ylabel('Puntuación')
    plt.title('Métricas por Clase — Precision, Recall y F1')
    plt.xticks(x, class_labels)
    plt.ylim(0, 1.05)
    plt.legend()

    # Etiquetas
    for bars, vals in [(bars1, precision_vals), (bars2, recall_vals), (bars3, f1_vals)]:
        for i, b in enumerate(bars):
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2.0, h + 0.02, f"{vals[i]:.2f}",
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_fig(fig, "04_metricas_por_clase_PRF.png")
    plt.show()
    plt.close(fig)


def plot_f2_bars(report_dict: dict, class_order: List[str]) -> None:
    """
    F2 = (5 * P * R) / (4 * P + R). Enfatiza Recall (penaliza más FN).
    """
    class_labels = [c for c in class_order if c in report_dict]
    f2_vals = []
    for c in class_labels:
        p = report_dict[c]['precision']
        r = report_dict[c]['recall']
        f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0.0
        f2_vals.append(f2)

    fig = plt.figure(figsize=(6, 4.5), dpi=FIG_DPI)
    bars = plt.bar(class_labels, f2_vals)
    plt.title("F2 por Clase (más peso al Recall)")
    plt.xlabel("Clase")
    plt.ylabel("F2")
    plt.ylim(0, 1.05)

    for i, b in enumerate(bars):
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2.0, h + 0.02, f"{h:.2f}",
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    save_fig(fig, "05_metricas_por_clase_F2.png")
    plt.show()
    plt.close(fig)


def show_top_tokens_per_class(pipe: Pipeline,
                              class_order: List[str],
                              top_n: int = 15) -> None:
    """
    Extrae y grafica las palabras más características por clase según Naive Bayes.
    - Para MultinomialNB, usamos feature_log_prob_ (log P(token|class)).
    - Muestra y guarda las top-N por clase.
    """
    vect: TfidfVectorizer = pipe.named_steps['tfidf']
    nb: MultinomialNB = pipe.named_steps['nb']

    feature_names = np.array(vect.get_feature_names_out())
    log_probs = nb.feature_log_prob_  # shape: (n_classes, n_features)
    nb_labels = list(nb.classes_)

    for lbl in [c for c in class_order if c in nb_labels]:
        cls_idx = nb_labels.index(lbl)
        top_idx = np.argsort(log_probs[cls_idx])[::-1][:top_n]
        top_features = feature_names[top_idx]
        top_scores = log_probs[cls_idx][top_idx]

        # Imprimir en consola (útil para clase)
        print(f"\nTop {top_n} tokens característicos para la clase '{lbl}':")
        for t, s in zip(top_features, top_scores):
            print(f"  {t:20s}  logP(token|{lbl}) = {s:.3f}")

        # Gráfica horizontal
        fig = plt.figure(figsize=(8, 5), dpi=FIG_DPI)
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_scores)
        plt.yticks(y_pos, top_features)
        plt.gca().invert_yaxis()  # el más alto arriba
        plt.title(f"Top {top_n} tokens para clase '{lbl}'")
        plt.xlabel("log P(token | clase)")
        plt.tight_layout()
        fname = f"06_top_tokens_{slugify(lbl)}.png"
        save_fig(fig, fname)
        plt.show()
        plt.close(fig)


def demo_predictions(pipe: Pipeline, examples: List[str]) -> None:
    """
    Muestra predicciones y probabilidades para ejemplos de texto.
    Didáctico para entender cómo decide Naive Bayes.
    """
    preds = pipe.predict(examples)
    if hasattr(pipe.named_steps['nb'], "predict_proba"):
        probs = pipe.predict_proba(examples)
        labels = pipe.named_steps['nb'].classes_
    else:
        probs = None
        labels = []

    print("\n=== Predicciones de ejemplo ===")
    for i, txt in enumerate(examples):
        print(f"\nTexto {i+1}: {txt}")
        print(f"Predicción: {preds[i]}")
        if probs is not None:
            prob_row = probs[i]
            show = ", ".join([f"P({labels[j]})={prob_row[j]:.3f}" for j in range(len(labels))])
            print(f"Probabilidades: {show}")


# ==========================
# MAIN (flujo didáctico)
# ==========================
def main():
    print("=== Naive Bayes de Texto — Flujo Didáctico ===\n")
    print(f"Ruta de datos:\n{DATA_PATH}")
    print(f"Carpeta de imágenes de salida:\n{IMG_DIR}\n")

    ensure_img_dir()

    # 1) Cargar
    df = load_dataset(DATA_PATH)
    print("\nPrimeras filas del DataFrame:")
    print(df.head())

    # 2) Limpiar & filtrar
    df = clean_and_filter(df, VALID_LABELS)
    print("\nClases presentes tras filtrar:", sorted(df['sentiment'].unique()))
    print(f"Número total de filas tras limpieza: {len(df)}")

    # 3) Distribución de clases (útil para hablar de desbalance)
    plot_class_distribution(df, 'sentiment')

    # 4) Split estratificado
    X = df['text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # importante en clasificación multiclase
    )
    print(f"\nTamaños -> Train: {len(X_train)} | Test: {len(X_test)}")

    # 5) Pipeline TF-IDF + MultinomialNB
    pipe = build_pipeline()

    # 6) Entrenar
    pipe.fit(X_train, y_train)

    # 7) Predecir
    y_pred = pipe.predict(X_test)

    # 8) Métricas globales
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print("\n=== Resultados en Test ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 (macro): {f1_macro:.3f}  |  F1 (weighted): {f1_weighted:.3f}")

    # 9) Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3, labels=VALID_LABELS))

    # 10) Matriz de confusión (conteos y porcentajes)
    desired_labels = [lbl for lbl in VALID_LABELS if lbl in df['sentiment'].unique()]
    compute_and_plot_confusion(y_test, y_pred, desired_labels)

    # 11) Barras de Precision/Recall/F1 por clase
    report_dict = classification_report(y_test, y_pred, output_dict=True, labels=desired_labels)
    plot_prf_bars(report_dict, desired_labels)

    # 12) F2 por clase
    plot_f2_bars(report_dict, desired_labels)

    # 13) Ejemplos de errores (opcionales, didácticos)
    results_df = pd.DataFrame({
        'text': X_test.values,
        'true': y_test.values,
        'pred': y_pred
    })
    errors_df = results_df[results_df['true'] != results_df['pred']].copy()
    if not errors_df.empty:
        print("\n=== Algunos errores de clasificación (primeros 5) ===")
        print(errors_df.head(5).to_string(index=False))
    else:
        print("\nNo hubo errores en el subconjunto de test (inusual, revisa tamaño o dificultad).")

    # 14) Top tokens por clase (qué palabras 'caracterizan' cada clase para NB)
    show_top_tokens_per_class(pipe, desired_labels, top_n=TOP_N_TOKENS)

    # 15) Demostración de predicciones con probabilidades (didáctico)
    demo_examples = [
        "Stocks tumble as inflation concerns rise",
        "Company reports neutral earnings with stable outlook",
        "Markets rally on positive economic data",
        "Oil prices fall after weak demand forecast",
        "Tech sector posts strong gains amid optimism",
    ]
    demo_predictions(pipe, demo_examples)

    print("\n=== Fin del flujo didáctico de Naive Bayes ===")
    print(f"Las imágenes fueron guardadas en: {IMG_DIR.resolve()}")


if __name__ == "__main__":
    main()

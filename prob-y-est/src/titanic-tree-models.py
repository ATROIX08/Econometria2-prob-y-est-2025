# src/titanic-tree-models.py
# -*- coding: utf-8 -*-
"""
Evaluación comparativa en Titanic (archivo: data/titanic/Titanic-Dataset.csv)
Modelos: Árbol de Decisión, Random Forest y XGBoost (obligatorio)
Preprocesamiento: imputación + One-Hot
Gráficas: matriz de confusión con métricas, ROC, PR, e importancias
Visualización de árboles: DT completo, un árbol de RF (#0) y árbol #0 de XGB
Salidas de imagen en: images/titanic_plots/

Ejecución (desde la raíz del repo):
    python src/titanic-tree-models.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Backend no interactivo para generar PNGs sin abrir ventanas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb


# ============================ Configuración ============================

SEED = 42
TEST_SIZE = 0.20
IMAGES_SUBDIR = "titanic_plots"


# ============================== Rutas =================================

def resolve_paths() -> Tuple[Path, Path, Path]:
    this_file = Path(__file__).resolve()
    root = this_file.parents[1]  # .../prob-y-est
    data_csv = root / "data" / "titanic" / "Titanic-Dataset.csv"
    images_dir = (root / "images" / IMAGES_SUBDIR)
    images_dir.mkdir(parents=True, exist_ok=True)
    return root, data_csv, images_dir


# ============================== Datos =================================

def load_titanic_csv(csv_path: Path) -> pd.DataFrame:
    exists_flag = csv_path.exists()
    if exists_flag is False:
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Columnas mínimas requeridas
    required = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Faltan columnas requeridas: {missing}\nDisponibles: {list(df.columns)}")

    # Y binaria sin nulos (no incluir PassengerId en ningún paso)
    df["Survived"] = pd.to_numeric(df["Survived"], errors="coerce")
    df = df.dropna(subset=["Survived"]).copy()
    df["Survived"] = df["Survived"].astype(int)

    return df


def select_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    # Solo variables útiles; NO incluir PassengerId
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].copy()
    y = df["Survived"].astype(int).values
    return X, y


# ============================ Preproceso ==============================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Numéricos: imputación mediana | Categóricos: moda + One-Hot (denso)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return pre


def fit_transform_preprocessor(pre: ColumnTransformer,
                               X_train: pd.DataFrame,
                               X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pre.fit(X_train)
    X_train_p = pre.transform(X_train)
    X_test_p = pre.transform(X_test)
    return X_train_p, X_test_p


def get_feature_names(pre: ColumnTransformer, X_sample: pd.DataFrame) -> List[str]:
    names: List[str] = []
    # num
    num_cols = pre.transformers_[0][2]
    names.extend(list(num_cols))
    # cat
    cat_cols = pre.transformers_[1][2]
    ohe = pre.named_transformers_["cat"].named_steps["ohe"]
    ohe_names = list(ohe.get_feature_names_out(cat_cols))
    names.extend(ohe_names)
    return names


# ============================== Modelos ===============================

def build_models(seed: int = SEED) -> Dict[str, object]:
    # Parámetros ajustados con intención de mejorar accuracy
    models = {}

    # Árbol de decisión: profundidad y splits moderados (evitar sobreajuste)
    models["DecisionTree"] = DecisionTreeClassifier(
        random_state=seed,
        criterion="gini",
        max_depth=5,
        min_samples_split=3,
        min_samples_leaf=2
    )

    # Random Forest: más árboles, profundidad controlada, sqrt en features
    models["RandomForest"] = RandomForestClassifier(
        random_state=seed,
        n_estimators=900,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1
    )

    # XGBoost: LR baja, más árboles, regularización y subsampling
    models["XGBoost"] = xgb.XGBClassifier(
        random_state=seed,
        n_estimators=800,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.3,
        reg_alpha=0.0,
        min_child_weight=3,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1
    )

    return models


# ========================= Evaluación & Plots =========================

def evaluate_and_plot(model_name: str,
                      clf,
                      X_train_p: np.ndarray, y_train: np.ndarray,
                      X_test_p: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str],
                      outdir: Path) -> Dict:
    # Entrenar
    clf.fit(X_train_p, y_train)

    # Predicciones
    y_pred = clf.predict(X_test_p)
    y_prob = clf.predict_proba(X_test_p)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f} | Balanced Acc.: {bacc:.4f}")
    print(report)

    # Matriz de confusión + métricas
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title(f"{model_name} — Matriz de confusión", fontsize=12)
    txt = f"Accuracy: {acc:.4f}\nBalanced Acc.: {bacc:.4f}\n\n{report}"
    fig.text(1.02, 0.5, txt, va="center", ha="left", fontsize=9, transform=ax.transAxes)
    plt.tight_layout()
    fig.savefig(outdir / f"titanic_{model_name}_matriz_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"{model_name} — Curva ROC")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(outdir / f"titanic_{model_name}_roc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"{model_name} — Curva Precision-Recall")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(outdir / f"titanic_{model_name}_pr.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Importancias
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8.4, max(4.2, 0.36 * len(order))))
    ax.barh(np.array(feature_names)[order][::-1], imp[order][::-1])
    ax.set_title(f"{model_name} — Importancia de variables (Top 20)")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    fig.savefig(outdir / f"titanic_{model_name}_importancias.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "model_name": model_name,
        "estimator": clf,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "report": report
    }


# ======================= Visualización de Árboles =====================

def plot_decision_tree_full(clf_dt: DecisionTreeClassifier,
                            feature_names: List[str],
                            outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(
        clf_dt,
        feature_names=feature_names,
        class_names=["No sobrevive", "Sobrevive"],
        filled=True,
        impurity=True,
        rounded=True,
        proportion=True,
        ax=ax
    )
    ax.set_title("Árbol de Decisión — Completo")
    plt.tight_layout()
    fig.savefig(outdir / "titanic_DecisionTree_arbol.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_random_forest_one_tree(clf_rf: RandomForestClassifier,
                                feature_names: List[str],
                                outdir: Path) -> None:
    tree0 = clf_rf.estimators_[0]
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(
        tree0,
        feature_names=feature_names,
        class_names=["No sobrevive", "Sobrevive"],
        filled=True,
        impurity=True,
        rounded=True,
        proportion=True,
        ax=ax
    )
    ax.set_title("Random Forest — Árbol ejemplo (#0)")
    plt.tight_layout()
    fig.savefig(outdir / "titanic_RandomForest_arbol_0.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_xgb_one_tree(clf_xgb: xgb.XGBClassifier,
                      outdir: Path) -> None:
    booster = clf_xgb.get_booster()
    fig, ax = plt.subplots(figsize=(20, 12))
    xgb.plot_tree(booster, num_trees=0, ax=ax)
    ax.set_title("XGBoost — Árbol (0)")
    plt.tight_layout()
    fig.savefig(outdir / "titanic_XGBoost_arbol_0.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================== Main =================================

def main(seed: int = SEED) -> None:
    # Rutas
    _, data_csv, images_dir = resolve_paths()

    # Datos y variables
    df = load_titanic_csv(data_csv)
    X, y = select_features_and_target(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )

    # Preprocesamiento
    pre = build_preprocessor(X)
    X_train_p, X_test_p = fit_transform_preprocessor(pre, X_train, X_test)
    feature_names = get_feature_names(pre, X_train)

    # Modelos
    models = build_models(seed)

    # Evaluación + gráficas (entrena dentro)
    results: List[Dict] = []
    for name, model in models.items():
        res = evaluate_and_plot(
            name, model,
            X_train_p, y_train, X_test_p, y_test,
            feature_names, images_dir
        )
        results.append(res)

    # Árboles (modelos ya entrenados arriba)
    plot_decision_tree_full(models["DecisionTree"], feature_names, images_dir)
    plot_random_forest_one_tree(models["RandomForest"], feature_names, images_dir)
    plot_xgb_one_tree(models["XGBoost"], images_dir)

    # Resumen
    print("\n=== RESUMEN ===")
    for r in results:
        print(f"{r['model_name']}: acc={r['accuracy']:.4f} | bacc={r['balanced_accuracy']:.4f}")
    print(f"\nImágenes en: {images_dir.resolve()}")


if __name__ == "__main__":
    main()

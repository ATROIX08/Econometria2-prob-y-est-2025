# -*- coding: utf-8 -*-
"""
predictive.py
--------------------------------------------------------------------------------
Modelo predictivo robusto para precio de diamantes con Grid Search
paralelizado con Dask usando dask-ml. Genera métricas, gráficos y logs.

Requisitos (asumidos instalados):
- Python 3.10+
- numpy, pandas, scikit-learn, matplotlib, joblib
- dask, distributed
- dask-ml
- pyarrow (para leer Parquet con pandas)

Ejecución (rutas por defecto de tu proyecto):
    python predictive.py

Opcionalmente puedes sobreescribir rutas:
    python predictive.py --parquet "<ruta.parquet>" \
        --plots_dir "<ruta_plots>" \
        --logs_dir "<ruta_logs>"

Notas:
- Split 80/20 estratificado con bins por cuantiles de la variable objetivo (price).
- Se excluyen TODAS las columnas que contengan 'price' o 'ppc' (price per carat)
  en su nombre para evitar leakage (la Y=price se separa).
- Se eliminan filas con dimensiones inválidas y columnas redundantes de tamaño.
- Modelo: HistGradientBoostingRegressor + GridSearchCV (KFold) con dask-ml.
- Gráficos: observado vs predicho, hist residuales, residuales vs predicho,
  y Permutation Importance (Top-K).
"""

import os
import sys
import json
import time
import math
import argparse
import logging
import warnings
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import GridSearchCV  # <<< dask-ml GridSearchCV


# --------------------------------------------------------------------------------------
# Rutas por defecto (según solicitud)
# --------------------------------------------------------------------------------------

DEFAULT_PARQUET_PATH = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\parquets\diamonds_features_20251012_155956.parquet"
DEFAULT_PLOTS_DIR = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\plots\modeling"
DEFAULT_LOGS_DIR = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\logs\modeling"


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(logs_dir: str) -> Tuple[logging.Logger, str]:
    ensure_dir(logs_dir)
    log_path = os.path.join(logs_dir, f"predictive_{now_ts()}.log")

    logger = logging.getLogger("predictive")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    warnings.simplefilter("always")

    logger.info("==========================================")
    logger.info("   Inicio de ejecución: predictive.py     ")
    logger.info("==========================================")
    logger.info(f"Logs detallados en: {log_path}")

    return logger, log_path


def read_parquet_to_pandas(parquet_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Leyendo Parquet desde: {parquet_path}")
    if not os.path.exists(parquet_path):
        logger.error(f"No se encontró el archivo Parquet en: {parquet_path}")
        raise FileNotFoundError(f"Parquet no encontrado: {parquet_path}")
    pdf = pd.read_parquet(parquet_path)
    logger.info(f"Parquet leído: {pdf.shape[0]:,} filas x {pdf.shape[1]:,} columnas.")
    return pdf


def detect_and_drop_price_related_columns(columns: List[str]) -> List[str]:
    cols_to_drop = []
    for c in columns:
        cl = c.lower()
        if "price" in cl or "ppc" in cl:
            cols_to_drop.append(c)
    if "fe_ppc_z_by_cqc" in columns:
        cols_to_drop.append("fe_ppc_z_by_cqc")
    return sorted(set(cols_to_drop))


def select_feature_columns(
    df: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if "price" not in df.columns:
        raise ValueError("La columna 'price' no está en el dataset.")

    y = df["price"].astype(float)

    if "fe_invalid_dims" in df.columns:
        before = df.shape[0]
        df = df.loc[~df["fe_invalid_dims"].astype(bool)].copy()
        dropped = before - df.shape[0]
        logger.info(f"Filas eliminadas por 'fe_invalid_dims'==True: {dropped:,}")
        y = y.loc[df.index]
    else:
        invalid_mask = None
        for dim in ("x", "y", "z"):
            if dim in df.columns:
                m = (df[dim] <= 0)
                invalid_mask = m if invalid_mask is None else (invalid_mask | m)
        if invalid_mask is not None:
            before = df.shape[0]
            df = df.loc[~invalid_mask].copy()
            dropped = before - df.shape[0]
            logger.info(f"Filas eliminadas por dimensiones no positivas: {dropped:,}")
            y = y.loc[df.index]

    price_related = detect_and_drop_price_related_columns(df.columns.tolist())
    if "price" in price_related:
        price_related.remove("price")
    if len(price_related) > 0:
        logger.info(f"Columnas excluidas por leakage (contienen 'price'/'ppc'): {price_related}")

    redundant_size_cols = [
        "x", "y", "z",
        "fe_volume_mm3", "fe_area_mm2", "fe_spread_mm",
        "fe_aspect_ratio", "fe_depth_pct_recalc", "fe_depth_pct_diff",
        "fe_table_to_depth_ratio", "fe_depth_dev", "fe_table_dev"
    ]
    redundant_existing = [c for c in redundant_size_cols if c in df.columns]
    if len(redundant_existing) > 0:
        logger.info(f"Columnas redundantes (tamaño) eliminadas: {redundant_existing}")

    candidate_cols = [
        c for c in df.columns
        if c not in price_related
        and c not in redundant_existing
        and c != "price"
        and c != "index"
    ]

    guided_keep_priority = [
        "carat", "fe_log_carat", "cut", "color", "clarity",
        "fe_cut_ord", "fe_color_ord", "fe_clarity_ord",
        "fe_carat_bin", "fe_quality_score", "fe_carat_x_quality",
        "depth", "table", "fe_depth_pct_is_consistent"
    ]
    selected_cols = [c for c in guided_keep_priority if c in candidate_cols]
    for c in candidate_cols:
        if c not in selected_cols:
            selected_cols.append(c)

    X = df[selected_cols].copy()

    categorical_features = [c for c in ["cut", "color", "clarity", "fe_carat_bin"] if c in X.columns]
    for c in categorical_features:
        X[c] = X[c].astype("category")

    numeric_features = [c for c in X.columns if c not in categorical_features]

    logger.info(f"Total columnas seleccionadas para X: {len(selected_cols)}")
    logger.info(f"Numéricas ({len(numeric_features)}): {numeric_features}")
    logger.info(f"Categóricas ({len(categorical_features)}): {categorical_features}")

    y = y.loc[X.index].astype(float)

    return X, y, numeric_features, categorical_features


def stratified_train_test_split_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    n_bins: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    strata = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strata
    )
    return X_train, X_test, y_train, y_test


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int
) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Para scikit-learn 1.7+, usar 'sparse_output' (no 'sparse')
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop",
        verbose_feature_names_out=True,
        n_jobs=None  # evitar interferencias; dask-ml ya paraleliza a nivel de CV
    )

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=random_state,
        early_stopping=True
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    return pipe


def get_param_grid() -> Dict[str, List]:
    return {
        "regressor__learning_rate": [0.01, 0.05, 0.1],
        "regressor__max_depth": [None, 6, 10, 16],
        "regressor__max_leaf_nodes": [31, 63, 127],
        "regressor__min_samples_leaf": [20, 50, 100],
        "regressor__l2_regularization": [0.0, 0.1, 1.0],
        "regressor__max_bins": [255],
        "regressor__early_stopping": [True],
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def save_metrics_json(metrics: dict, logs_dir: str, logger: logging.Logger) -> str:
    ensure_dir(logs_dir)
    path = os.path.join(logs_dir, f"metrics_{now_ts()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Métricas guardadas en JSON: {path}")
    return path


def plot_and_save(y_test: np.ndarray, y_pred: np.ndarray, plots_dir: str, logger: logging.Logger) -> dict:
    import matplotlib.pyplot as plt

    ensure_dir(plots_dir)
    ts = now_ts()

    # 1) Observado vs Predicho
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(y_test, y_pred, alpha=0.5)
    minv = float(min(np.min(y_test), np.min(y_pred)))
    maxv = float(max(np.max(y_test), np.max(y_pred)))
    ax1.plot([minv, maxv], [minv, maxv], linestyle="--")
    ax1.set_title("Precio observado vs. precio predicho")
    ax1.set_xlabel("Observado (price)")
    ax1.set_ylabel("Predicho")
    scatter_path = os.path.join(plots_dir, f"observado_vs_predicho_{ts}.png")
    fig1.savefig(scatter_path, dpi=120, bbox_inches="tight")
    plt.close(fig1)

    # 2) Histograma de residuales
    residuals = y_test - y_pred
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.hist(residuals, bins=50)
    ax2.set_title("Histograma de residuales")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frecuencia")
    resid_hist_path = os.path.join(plots_dir, f"residuales_hist_{ts}.png")
    fig2.savefig(resid_hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # 3) Residuales vs Predicho
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.scatter(y_pred, residuals, alpha=0.5)
    ax3.axhline(0.0, linestyle="--")
    ax3.set_title("Residuales vs. Predicho")
    ax3.set_xlabel("Predicho")
    ax3.set_ylabel("Residual")
    resid_vs_pred_path = os.path.join(plots_dir, f"residuales_vs_predicho_{ts}.png")
    fig3.savefig(resid_vs_pred_path, dpi=120, bbox_inches="tight")
    plt.close(fig3)

    logger.info("Gráficos generados:")
    logger.info(f"  - Observado vs Predicho: {scatter_path}")
    logger.info(f"  - Histograma residuales: {resid_hist_path}")
    logger.info(f"  - Residuales vs Predicho: {resid_vs_pred_path}")

    return {
        "observado_vs_predicho": scatter_path,
        "residuales_hist": resid_hist_path,
        "residuales_vs_predicho": resid_vs_pred_path
    }


def plot_permutation_importance(
    best_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_k: int,
    plots_dir: str,
    logger: logging.Logger
) -> Optional[str]:
    import matplotlib.pyplot as plt

    result = permutation_importance(
        best_pipeline, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    pre: ColumnTransformer = best_pipeline.named_steps["preprocessor"]
    feature_names = pre.get_feature_names_out()
    importances = result.importances_mean
    stds = result.importances_std

    order = np.argsort(np.abs(importances))[::-1]
    top = order[:top_k]
    names_top = feature_names[top]
    imp_top = importances[top]
    std_top = stds[top]

    ensure_dir(plots_dir)
    ts = now_ts()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.barh(range(len(top)), imp_top[::-1], xerr=std_top[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(n) for n in names_top[::-1]])
    ax.set_xlabel("Importancia (media de decremento de score)")
    ax.set_title(f"Permutation Importance - Top {top_k}")
    fig.tight_layout()
    out_path = os.path.join(plots_dir, f"permutation_importance_top{top_k}_{ts}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Permutation importance guardado en: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Modelo predictivo de precio de diamantes (HGBR + GridSearch con dask-ml).")
    parser.add_argument("--parquet", type=str, default=DEFAULT_PARQUET_PATH,
                        help="Ruta al Parquet de entrada (features + originales).")
    parser.add_argument("--plots_dir", type=str, default=DEFAULT_PLOTS_DIR,
                        help="Directorio para guardar gráficos del modelado.")
    parser.add_argument("--logs_dir", type=str, default=DEFAULT_LOGS_DIR,
                        help="Directorio para guardar logs y resúmenes del modelado.")
    parser.add_argument("--test_size", type=float, default=0.20, help="Proporción de test (por defecto 0.20).")
    parser.add_argument("--cv_splits", type=int, default=5, help="Número de folds de CV para GridSearch (default=5).")
    parser.add_argument("--random_state", type=int, default=42, help="Semilla de aleatoriedad.")
    parser.add_argument("--top_k_importance", type=int, default=20, help="Top-K features para permutation importance.")
    args = parser.parse_args()

    ensure_dir(args.plots_dir)
    ensure_dir(args.logs_dir)

    logger, log_path = setup_logging(args.logs_dir)

    logger.info("Parámetros de ejecución:")
    logger.info(f"  Parquet  : {args.parquet}")
    logger.info(f"  Plots dir: {args.plots_dir}")
    logger.info(f"  Logs dir : {args.logs_dir}")
    logger.info(f"  test_size: {args.test_size}")
    logger.info(f"  cv_splits: {args.cv_splits}")
    logger.info(f"  seed     : {args.random_state}")
    logger.info(f"  top_k_imp: {args.top_k_importance}")

    # --------------------------------------------------------------------------
    # Dask: cluster local con hilos (sin procesos) para evitar spawn en Windows.
    # 'silence_logs' debe ser un nivel válido -> logging.ERROR
    # --------------------------------------------------------------------------
    n_workers = max(1, (os.cpu_count() or 2) // 2)
    cluster = LocalCluster(
        processes=False,
        n_workers=n_workers,
        threads_per_worker=1,
        silence_logs=logging.ERROR
    )
    client = Client(cluster)
    logger.info(f"Dask cluster inicializado con {n_workers} workers (threads_per_worker=1).")
    logger.info(f"Dashboard: {client.dashboard_link}")

    start_time = time.time()

    # Lectura de datos
    df = read_parquet_to_pandas(args.parquet, logger)

    # Selección de features guiada por EDA y separación de Y
    X, y, numeric_features, categorical_features = select_feature_columns(df, logger)

    # Split estratificado 80/20
    X_train, X_test, y_train, y_test = stratified_train_test_split_regression(
        X, y, test_size=args.test_size, random_state=args.random_state, n_bins=10
    )
    logger.info(f"Split realizado. Train: {X_train.shape[0]:,} filas; Test: {X_test.shape[0]:,} filas.")

    # Pipeline y GridSearch (dask-ml)
    pipeline = build_pipeline(numeric_features, categorical_features, random_state=args.random_state)
    param_grid = get_param_grid()
    cv = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)

    # dask-ml GridSearchCV: paraleliza evaluaciones con Dask
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        return_train_score=True  # dask-ml lo soporta
    )

    logger.info("Iniciando GridSearchCV (dask-ml) sobre el cluster Dask...")
    grid.fit(X_train, y_train)

    elapsed = time.time() - start_time
    logger.info(f"GridSearchCV finalizado en {elapsed:,.1f} segundos.")

    # Resultados del GridSearch
    best_score_neg_rmse = grid.best_score_
    best_rmse_cv = -best_score_neg_rmse
    best_params = grid.best_params_
    logger.info("==========================================")
    logger.info("      MEJOR MODELO (GridSearchCV)        ")
    logger.info("==========================================")
    logger.info(f"Mejor RMSE (CV): {best_rmse_cv:,.4f}")
    logger.info(f"Mejores hiperparámetros: {json.dumps(best_params, indent=2)}")

    # cv_results_ puede variar ligeramente según versión de dask-ml/sklearn
    cv_results = pd.DataFrame(grid.cv_results_)
    if "mean_test_score" in cv_results.columns:
        cv_results["mean_RMSE"] = -cv_results["mean_test_score"]
    else:
        # respaldo por si el nombre cambiara (poco probable)
        score_col = [c for c in cv_results.columns if c.startswith("mean_test")]
        cv_results["mean_RMSE"] = -cv_results[score_col[0]] if score_col else np.nan

    cv_results_sorted = cv_results.sort_values("mean_RMSE", ascending=True).reset_index(drop=True)
    logger.info("Top-5 combinaciones (por RMSE CV):")
    for i in range(min(5, cv_results_sorted.shape[0])):
        row = cv_results_sorted.iloc[i]
        std_col = "std_test_score" if "std_test_score" in cv_results_sorted.columns else None
        std_val = row[std_col] if std_col else np.nan
        logger.info(
            f"  #{i+1}: RMSE={row['mean_RMSE']:.4f}"
            + (f" ± {std_val:.4f}" if std_col else "")
            + f" | params={row['params']}"
        )

    # Evaluación en Test
    best_estimator: Pipeline = grid.best_estimator_
    y_pred = best_estimator.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = rmse(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_mape = mape(y_test, y_pred)

    logger.info("==========================================")
    logger.info("        MÉTRICAS EN CONJUNTO TEST         ")
    logger.info("==========================================")
    logger.info(f"MSE  : {test_mse:,.4f}")
    logger.info(f"RMSE : {test_rmse:,.4f}")
    logger.info(f"MAE  : {test_mae:,.4f}")
    logger.info(f"R^2  : {test_r2:,.6f}")
    logger.info(f"MAPE : {test_mape:,.3f}%")

    metrics_payload = {
        "timestamp": now_ts(),
        "rows_train": int(X_train.shape[0]),
        "rows_test": int(X_test.shape[0]),
        "features_used": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "best_params": best_params,
        "cv_best_rmse": float(best_rmse_cv),
        "test_mse": float(test_mse),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "test_mape_pct": float(test_mape)
    }
    save_metrics_json(metrics_payload, args.logs_dir, logger)

    # Gráficos principales
    plot_paths = plot_and_save(y_test.values, y_pred, args.plots_dir, logger)

    # Permutation Importance
    perm_path = plot_permutation_importance(
        best_estimator, X_test, y_test, args.top_k_importance, args.plots_dir, logger
    )

    # Resumen TXT
    txt_summary_path = os.path.join(args.logs_dir, f"model_summary_{now_ts()}.txt")
    with open(txt_summary_path, "w", encoding="utf-8") as f:
        f.write("RESUMEN DE MODELADO - PREDICCIÓN DE PRECIO DE DIAMANTES\n")
        f.write("=======================================================\n\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write(f"Parquet origen: {args.parquet}\n")
        f.write(f"Tamaño Train/Test: {X_train.shape} / {X_test.shape}\n")
        f.write("\nMejores hiperparámetros (GridSearchCV):\n")
        f.write(json.dumps(best_params, indent=2, ensure_ascii=False))
        f.write("\n\nMétricas en Test:\n")
        f.write(f"- MSE  : {test_mse:,.6f}\n")
        f.write(f"- RMSE : {test_rmse:,.6f}\n")
        f.write(f"- MAE  : {test_mae:,.6f}\n")
        f.write(f"- R^2  : {test_r2:,.6f}\n")
        f.write(f"- MAPE : {test_mape:,.6f}%\n")
        f.write("\nGráficos generados:\n")
        for k, v in plot_paths.items():
            f.write(f"- {k}: {v}\n")
        if perm_path is not None:
            f.write(f"- permutation_importance: {perm_path}\n")
        f.write("\nColumnas utilizadas (X):\n")
        f.write(", ".join(X.columns))
        f.write("\n")
    logger.info(f"Resumen TXT guardado en: {txt_summary_path}")

    client.close()
    logger.info("Cliente Dask cerrado correctamente.")

    logger.info("==========================================")
    logger.info("    Ejecución finalizada sin errores.     ")
    logger.info("==========================================")


if __name__ == "__main__":
    main()

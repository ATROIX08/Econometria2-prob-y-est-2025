# -*- coding: utf-8 -*-
"""
Autor: Humberto Silva Baltazar
Entrena el mejor modelo (sin GridSearch) con los hiperparámetros óptimos
obtenidos previamente. Genera logs, métricas, plots (mejorados) y guarda
el pipeline entrenado (joblib) para inferencia.

- Lee el Parquet con features del ETL.
- Filtra registros inválidos (fe_invalid_dims == True).
- Elimina columnas con leakage y redundantes.
- Misma selección de variables que el experimento con GridSearch.
- Split 80/20 estratificado (binning por precio).
- Entrena HistGradientBoostingRegressor con hiperparámetros óptimos.
- Evalúa en test y guarda métricas/plots/resumen.
- Guarda el pipeline entrenado.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from joblib import dump

import logging
import warnings

# --------------------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------------------

# Rutas base (ajústalas si cambian)
PARQUET_PATH = Path(
    r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\parquets\diamonds_features_20251012_155956.parquet"
)
PLOTS_DIR = Path(
    r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\plots\modeling"
)
LOGS_DIR = Path(
    r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\logs\modeling"
)
MODEL_DIR = Path(
    r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\model"
)

TEST_SIZE = 0.20
SEED = 42
TOP_K_IMPORTANCE = 20

# Hiperparámetros óptimos (de tu GridSearch)
BEST_PARAMS = dict(
    early_stopping=True,
    l2_regularization=0.1,
    learning_rate=0.1,
    max_bins=255,
    max_depth=10,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    random_state=SEED,
)

# Estilo y colores para plots (profesional)
plt.style.use("seaborn-v0_8-whitegrid")  # estilo limpio incluido en matplotlib
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
    }
)
COLORS = {
    "primary": "#2563eb",   # azul
    "secondary": "#16a34a", # verde
    "accent": "#f59e0b",    # ámbar
    "line": "#64748b",      # gris pizarra
    "danger": "#ef4444",    # rojo
    "purple": "#7c3aed",    # morado
    "gray": "#94a3b8",      # gris
}

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

def make_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"train_best_{ts}.log"

    logger = logging.getLogger("train_best")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-8s | train_best | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(ch)

    logger.info("=" * 41)
    logger.info("   Inicio de ejecución: train_best.py   ")
    logger.info("=" * 41)
    logger.info(f"Logs detallados en: {log_path}")
    return logger

logger = make_logger()

# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------

def human_int(n: int | float) -> str:
    s = f"{n:,.0f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_cols(existing: list[str], desired: list[str]) -> list[str]:
    exist_set = set(existing)
    return [c for c in desired if c in exist_set]

def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def fmt_spanish_axis(x, pos) -> str:
    # Formatea ticks con miles y separador decimal español
    s = f"{x:,.0f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

SPANISH_FORMATTER = FuncFormatter(fmt_spanish_axis)

def add_metrics_box(ax, *, rmse, mae, r2, mape, n=None, loc="upper left"):
    lines = [
        f"RMSE: {rmse:,.2f}",
        f"MAE : {mae:,.2f}",
        f"R²  : {r2:.4f}",
        f"MAPE: {mape:.2f}%",
    ]
    if n is not None:
        lines.append(f"N: {human_int(n)}")
    txt = "\n".join(lines).replace(",", "X").replace(".", ",").replace("X", ".")
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1")
    ax.text(
        0.02 if "left" in loc else 0.98,
        0.98 if "upper" in loc else 0.02,
        txt,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top" if "upper" in loc else "bottom",
        horizontalalignment="left" if "left" in loc else "right",
        bbox=bbox_props,
    )

# --------------------------------------------------------------------------------------
# Carga de datos y selección de features
# --------------------------------------------------------------------------------------

logger.info(f"Parquet: {PARQUET_PATH}")
logger.info(f"Plots  : {PLOTS_DIR}")
logger.info(f"Logs   : {LOGS_DIR}")
logger.info(f"Model  : {MODEL_DIR}")
logger.info(f"test_size={TEST_SIZE} | seed={SEED} | top_k_imp={TOP_K_IMPORTANCE}")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Leyendo Parquet...")
df = pl.read_parquet(str(PARQUET_PATH))
logger.info(f"Parquet leído: {human_int(df.shape[0])} filas x {df.shape[1]} columnas.")

# Filtrar registros con dimensiones inválidas
removed = df.filter(pl.col("fe_invalid_dims") == True).shape[0] if "fe_invalid_dims" in df.columns else 0
if "fe_invalid_dims" in df.columns:
    df = df.filter(pl.col("fe_invalid_dims") == False)
logger.info(f"Filas eliminadas por 'fe_invalid_dims'==True: {removed}")

# Polars -> pandas para scikit-learn
pdf: pd.DataFrame = df.to_pandas()

# Target
TARGET = "price"
if TARGET not in pdf.columns:
    raise ValueError(f"No encuentro la columna objetivo '{TARGET}' en el Parquet.")

# Columnas con leakage (contienen 'price' o 'ppc')
leak_cols = [c for c in pdf.columns if ("price" in c and c != TARGET) or ("ppc" in c.lower())]
logger.info(f"Columnas excluidas por leakage (contienen 'price'/'ppc'): {leak_cols}")

# Columnas redundantes (alto multicolinealidad de tamaño y derivados)
redundant_cols = [
    "x", "y", "z", "fe_volume_mm3", "fe_area_mm2", "fe_spread_mm", "fe_aspect_ratio",
    "fe_depth_pct_recalc", "fe_depth_pct_diff", "fe_table_to_depth_ratio",
    "fe_depth_dev", "fe_table_dev",
]
logger.info(f"Columnas redundantes (tamaño) eliminadas: {ensure_cols(pdf.columns.tolist(), redundant_cols)}")

# Variables candidatas (consistentes con tu corrida previa)
num_candidates = [
    "carat",
    "fe_log_carat",
    "fe_cut_ord",
    "fe_color_ord",
    "fe_clarity_ord",
    "fe_quality_score",
    "fe_carat_x_quality",
    "depth",
    "table",
    "fe_depth_pct_is_consistent",
    "fe_invalid_dims",          # tras filtrar debería ser 0; lo dejamos por reproducibilidad
    "fe_symmetry_dev_pct",
    "fe_z_to_spread_ratio",
    "fe_spread_per_carat",
    "fe_area_per_carat",
    "fe_is_square",
]
cat_candidates = ["cut", "color", "clarity", "fe_carat_bin"]

# Listas finales respetando existencia
cols_exclude = set([TARGET] + leak_cols + redundant_cols)
available_cols = [c for c in pdf.columns if c not in cols_exclude]

num_cols = ensure_cols(available_cols, num_candidates)
cat_cols = ensure_cols(available_cols, cat_candidates)
feature_cols = num_cols + cat_cols

logger.info(f"Total columnas seleccionadas para X: {len(feature_cols)}")
logger.info(f"Numéricas ({len(num_cols)}): {num_cols}")
logger.info(f"Categóricas ({len(cat_cols)}): {cat_cols}")

X = pdf[feature_cols].copy()
y = pdf[TARGET].astype(float).copy()

# --------------------------------------------------------------------------------------
# Split estratificado (binning por precio)
# --------------------------------------------------------------------------------------

bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=bins
)
logger.info(
    f"Split realizado. Train: {human_int(X_train.shape[0])} filas; "
    f"Test: {human_int(X_test.shape[0])} filas."
)

# --------------------------------------------------------------------------------------
# Pipeline de preprocesamiento + modelo
# --------------------------------------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]
)

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
    n_jobs=None,
)

regressor = HistGradientBoostingRegressor(**BEST_PARAMS)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("regressor", regressor),
    ]
)

# --------------------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------------------

logger.info("Entrenando HistGradientBoostingRegressor con hiperparámetros óptimos...")
pipe.fit(X_train, y_train)
logger.info("Entrenamiento finalizado.")

# --------------------------------------------------------------------------------------
# Evaluación en Test
# --------------------------------------------------------------------------------------

y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = (np.abs((y_test - y_pred) / np.maximum(y_test, 1e-9))).mean() * 100.0
corr = float(np.corrcoef(y_test, y_pred)[0, 1])

logger.info("=" * 41)
logger.info("        MÉTRICAS EN CONJUNTO TEST        ")
logger.info("=" * 41)
logger.info(f"MSE  : {mse:,.4f}".replace(",", "X").replace(".", ",").replace("X", "."))
logger.info(f"RMSE : {rmse:,.4f}".replace(",", "X").replace(".", ",").replace("X", "."))
logger.info(f"MAE  : {mae:,.4f}".replace(",", "X").replace(".", ",").replace("X", "."))
logger.info(f"R^2  : {r2:.6f}")
logger.info(f"MAPE : {mape:.3f}%")
logger.info(f"Corr : {corr:.6f}")

# Guardar métricas en JSON
ts = now_tag()
metrics_path = LOGS_DIR / f"metrics_best_{ts}.json"
metrics_payload = {
    "timestamp": ts,
    "parquet": str(PARQUET_PATH),
    "model": "HistGradientBoostingRegressor",
    "best_params": BEST_PARAMS,
    "test_metrics": {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "corr": corr,
    },
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "features_numeric": num_cols,
    "features_categorical": cat_cols,
}
save_json(metrics_payload, metrics_path)
logger.info(f"Métricas guardadas en JSON: {metrics_path}")

# --------------------------------------------------------------------------------------
# Gráficas (mejoradas)
# --------------------------------------------------------------------------------------

def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# 1) Observado vs Predicho (con línea identidad y tendencia)
fig, ax = plt.subplots(figsize=(7.8, 6.6))
ax.scatter(y_test, y_pred, s=14, alpha=0.45, color=COLORS["primary"], label="Observaciones")

# Línea identidad
mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
ax.plot([mn, mx], [mn, mx], color=COLORS["danger"], linestyle="--", linewidth=1.2, label="Línea identidad")

# Tendencia lineal simple
coef = np.polyfit(y_test, y_pred, deg=1)
poly = np.poly1d(coef)
ax.plot([mn, mx], poly([mn, mx]), color=COLORS["secondary"], linewidth=1.8, label="Tendencia (OLS)")

ax.set_title("Precio Observado vs Predicho")
ax.set_xlabel("Precio observado")
ax.set_ylabel("Precio predicho")
ax.xaxis.set_major_formatter(SPANISH_FORMATTER)
ax.yaxis.set_major_formatter(SPANISH_FORMATTER)
ax.legend(frameon=True)
add_metrics_box(ax, rmse=rmse, mae=mae, r2=r2, mape=mape, n=len(y_test), loc="upper left")
obs_pred_path = PLOTS_DIR / f"observado_vs_predicho_best_{ts}.png"
save_plot(obs_pred_path)

# 2) Histograma de residuales (con media/±1σ y curva normal teórica)
resid = y_test - y_pred
mu = float(np.mean(resid))
sigma = float(np.std(resid))

fig, ax = plt.subplots(figsize=(7.8, 5.2))
counts, bins_hist, patches = ax.hist(resid, bins=50, color=COLORS["purple"], alpha=0.65, edgecolor="white", label="Residuales")
ax.axvline(mu, color=COLORS["accent"], linestyle="-", linewidth=1.6, label=f"Media = {mu:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
ax.axvline(mu + sigma, color=COLORS["gray"], linestyle="--", linewidth=1.2, label=f"+1σ = {mu+sigma:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
ax.axvline(mu - sigma, color=COLORS["gray"], linestyle="--", linewidth=1.2, label=f"-1σ = {mu-sigma:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# Curva normal teórica
x_vals = np.linspace(resid.min(), resid.max(), 400)
normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
# Escalar la PDF para que se compare con el histograma (altura)
scale = max(counts) / max(normal_pdf) if max(normal_pdf) > 0 else 1.0
ax.plot(x_vals, normal_pdf * scale, color=COLORS["secondary"], linewidth=1.6, label="Normal teórica (esc.)")

ax.set_title("Histograma de residuales (test)")
ax.set_xlabel("Residuales (observado - predicho)")
ax.set_ylabel("Frecuencia")
ax.legend(frameon=True)
add_metrics_box(ax, rmse=rmse, mae=mae, r2=r2, mape=mape, loc="upper right")
resid_hist_path = PLOTS_DIR / f"residuales_hist_best_{ts}.png"
save_plot(resid_hist_path)

# 3) Residuales vs Predicho (con línea 0 y tendencia binned)
fig, ax = plt.subplots(figsize=(7.8, 5.2))
ax.scatter(y_pred, resid, s=14, alpha=0.45, color=COLORS["danger"], label="Residuales")
ax.axhline(0, color=COLORS["line"], linewidth=1.2, linestyle="--", label="Residual = 0")

# Tendencia por bins (promedio por cuantiles de y_pred)
q = pd.qcut(y_pred, q=20, duplicates="drop")
trend = pd.DataFrame({"pred": y_pred, "resid": resid, "q": q}).groupby("q").agg({"pred": "mean", "resid": "mean"}).sort_values("pred")
ax.plot(trend["pred"].values, trend["resid"].values, color=COLORS["secondary"], linewidth=2.0, label="Tendencia binned")

ax.set_title("Residuales vs Predicho (test)")
ax.set_xlabel("Precio predicho")
ax.set_ylabel("Residual (obs - pred)")
ax.xaxis.set_major_formatter(SPANISH_FORMATTER)
ax.legend(frameon=True, loc="upper right")
add_metrics_box(ax, rmse=rmse, mae=mae, r2=r2, mape=mape, loc="lower left")
resid_vs_pred_path = PLOTS_DIR / f"residuales_vs_predicho_best_{ts}.png"
save_plot(resid_vs_pred_path)

logger.info("Gráficos generados:")
logger.info(f"  - Observado vs Predicho: {obs_pred_path}")
logger.info(f"  - Histograma residuales: {resid_hist_path}")
logger.info(f"  - Residuales vs Predicho: {resid_vs_pred_path}")

# --------------------------------------------------------------------------------------
# Permutation importance (Top-K, barras con error)
# --------------------------------------------------------------------------------------

logger.info("Calculando permutation importance (esto puede tardar un poco)...")
perm = permutation_importance(
    pipe, X_test, y_test, n_repeats=5, random_state=SEED, n_jobs=-1
)
pi_df = (
    pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    )
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)
top_k = min(TOP_K_IMPORTANCE, pi_df.shape[0])
pi_top = pi_df.head(top_k)

fig, ax = plt.subplots(figsize=(8.6, 0.45 * top_k + 1.6))
ax.barh(
    pi_top["feature"][::-1],
    pi_top["importance_mean"][::-1],
    xerr=pi_top["importance_std"][::-1],
    color=COLORS["primary"],
    alpha=0.85,
    ecolor=COLORS["line"],
    capsize=3,
)
ax.set_xlabel("Disminución media del score (perm.)")
ax.set_title(f"Permutation Importance (Top-{top_k})")
for i, v in enumerate(pi_top["importance_mean"][::-1].values):
    ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
pi_path = PLOTS_DIR / f"permutation_importance_best_top{top_k}_{ts}.png"
save_plot(pi_path)
logger.info(f"Permutation importance guardado en: {pi_path}")

# --------------------------------------------------------------------------------------
# Guardado del modelo (pipeline completo)
# --------------------------------------------------------------------------------------

model_path = MODEL_DIR / f"best_hgbr_pipeline_{ts}.joblib"
dump(pipe, model_path)
logger.info(f"Pipeline entrenado guardado en: {model_path}")

# --------------------------------------------------------------------------------------
# Resumen TXT
# --------------------------------------------------------------------------------------

summary_txt = LOGS_DIR / f"model_best_summary_{ts}.txt"
with summary_txt.open("w", encoding="utf-8") as f:
    f.write("==========================================\n")
    f.write("      Entrenamiento Modelo Óptimo (HGBR)  \n")
    f.write("==========================================\n")
    f.write(f"Timestamp: {ts}\n")
    f.write(f"Parquet  : {PARQUET_PATH}\n")
    f.write(f"Modelo   : HistGradientBoostingRegressor\n")
    f.write(f"Hipers   : {json.dumps(BEST_PARAMS, indent=2)}\n")
    f.write("\n")
    f.write(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")
    f.write(f"Numéricas ({len(num_cols)}): {num_cols}\n")
    f.write(f"Categóricas ({len(cat_cols)}): {cat_cols}\n")
    f.write("\n")
    f.write("Métricas (test):\n")
    f.write(f"  - MSE  : {mse:.6f}\n")
    f.write(f"  - RMSE : {rmse:.6f}\n")
    f.write(f"  - MAE  : {mae:.6f}\n")
    f.write(f"  - R^2  : {r2:.6f}\n")
    f.write(f"  - MAPE : {mape:.3f}%\n")
    f.write(f"  - Corr : {corr:.6f}\n")
    f.write("\n")
    f.write(f"Modelo guardado en: {model_path}\n")
    f.write(f"Plot obs vs pred : {obs_pred_path}\n")
    f.write(f"Plot resid hist  : {resid_hist_path}\n")
    f.write(f"Plot resid vs pr : {resid_vs_pred_path}\n")
    f.write(f"Perm. importance : {pi_path}\n")
logger.info(f"Resumen TXT guardado en: {summary_txt}")

logger.info("=" * 41)
logger.info("   Ejecución finalizada sin errores.     ")
logger.info("=" * 41)

# -*- coding: utf-8 -*-
"""
EDA breve + XGBoost para predecir la calificación (última columna),
compatible con entornos "viejos" de scikit-learn y xgboost.

Ajustes de compatibilidad clave (corrige error 'squared' y otros):
- RMSE: si sklearn no acepta mean_squared_error(..., squared=False),
  se calcula como sqrt(mean_squared_error(...)) manualmente.
- CV: usa 'neg_mean_squared_error' y convierte a RMSE con sqrt para
  soportar versiones sin 'neg_root_mean_squared_error'.
- OneHotEncoder: fallback si no existe min_frequency.
- permutation_importance: si no está disponible, se omite la gráfica.
- Heatmap de correlación: fallback si 'numeric_only' no existe.
- Early stopping: intenta con 'early_stopping_rounds'; si la versión no lo
  permite, entrena sin ES.

Requisitos:
    pip install pandas numpy scikit-learn xgboost joblib matplotlib
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")  # Permite guardar figuras sin display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import opcional de permutation_importance (puede no existir en versiones viejas)
try:
    from sklearn.inspection import permutation_importance as skl_permutation_importance
except Exception:
    skl_permutation_importance = None

import xgboost as xgb
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =========================
# CONFIGURACIÓN DE RUTAS
# =========================
OUTPUT_DIR = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\econometria-2\images\concurso"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATE_CSV_PATHS = [
    r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\econometria-2\data\desmpeno_escolar.csv",
    r".\desmpeno_escolar.csv",
    r"./desmpeno_escolar.csv",
    "/mnt/data/desmpeno_escolar.csv",
]


# =========================
# UTILIDADES
# =========================
def encontrar_csv(candidatos):
    for p in candidatos:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No se encontró el archivo CSV en las rutas candidatas. "
        "Actualiza CANDIDATE_CSV_PATHS al inicio del script."
    )

def leer_csv_robusto(path):
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No pude leer el CSV con encodings comunes. Último error: {last_err}")

def limpiar_nombres_columnas(df):
    cols = []
    for c in df.columns:
        cc = str(c).strip().replace("\n", " ").replace("\r", " ")
        cc = " ".join(cc.split())
        cols.append(cc)
    df.columns = cols
    return df

def estratificar_regresion(y, bins=10):
    y = pd.Series(y).astype(float)
    uniq = y.dropna().unique()
    n_bins = min(bins, max(2, len(uniq)))
    try:
        binned = pd.qcut(y, q=n_bins, duplicates="drop")
    except Exception:
        binned = pd.cut(y, bins=n_bins, duplicates="drop")
    return binned.astype(str)

def _rmse(y_true, y_pred):
    # Compatibilidad para sklearn sin 'squared' en mean_squared_error
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def metricas_regresion(y_true, y_pred):
    rmse = _rmse(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

def grafica_simple_guardar(fig, nombre_salida):
    ruta = os.path.join(OUTPUT_DIR, nombre_salida)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)

def resumen_eda(df, target_col, ruta_reporte_txt):
    lines = []
    lines.append("===== EDA BREVE =====")
    lines.append(f"Shape: {df.shape}")
    lines.append(f"Columnas: {list(df.columns)}")
    lines.append(f"Objetivo (última columna): {target_col}")
    lines.append("\n--- Tipos de datos ---")
    lines.append(df.dtypes.astype(str).to_string())
    lines.append("\n--- Nulos por columna (top 20) ---")
    nulos = df.isna().sum().sort_values(ascending=False)
    lines.append(nulos.head(20).to_string())
    lines.append("\n--- Descripción numérica (top 20) ---")
    desc = df.select_dtypes(include=[np.number]).describe().T
    try:
        lines.append(desc.head(20).to_string())
    except Exception:
        lines.append("No numéricas o error al describir.")
    with open(ruta_reporte_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def plot_distribucion_objetivo(y, nombre="01_target_distribution.png"):
    fig = plt.figure(figsize=(8, 5))
    plt.hist(y, bins=20, edgecolor="black")
    plt.title("Distribución de la calificación (y)")
    plt.xlabel("Calificación")
    plt.ylabel("Frecuencia")
    grafica_simple_guardar(fig, nombre)

def plot_correlacion_numerica(df_num, target_col, nombre="02_correlation_heatmap.png"):
    if df_num.shape[1] < 2:
        return
    try:
        corr = df_num.corr(numeric_only=True)
    except TypeError:
        corr = df_num.corr()
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(corr, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Mapa de correlación (variables numéricas)")
    grafica_simple_guardar(fig, nombre)

def _extract_train_valid_from_evals(evals_result, metric_name):
    # Soporta llaves 'train'/'valid' o 'validation_0'/'validation_1'
    if "train" in evals_result and metric_name in evals_result["train"]:
        train_vals = evals_result["train"][metric_name]
        if "valid" in evals_result and metric_name in evals_result["valid"]:
            valid_vals = evals_result["valid"][metric_name]
        elif "validation_1" in evals_result and metric_name in evals_result["validation_1"]:
            valid_vals = evals_result["validation_1"][metric_name]
        else:
            valid_vals = None
        return train_vals, valid_vals
    if "validation_0" in evals_result and metric_name in evals_result["validation_0"]:
        train_vals = evals_result["validation_0"][metric_name]
        valid_vals = None
        if "validation_1" in evals_result and metric_name in evals_result["validation_1"]:
            valid_vals = evals_result["validation_1"][metric_name]
        return train_vals, valid_vals
    return None, None

def plot_learning_curve(evals_result, metric_name="rmse", nombre="03_learning_curve.png"):
    train_vals, valid_vals = _extract_train_valid_from_evals(evals_result, metric_name)
    if train_vals is None or valid_vals is None:
        return
    fig = plt.figure(figsize=(8, 5))
    plt.plot(train_vals, label="Train")
    plt.plot(valid_vals, label="Valid")
    plt.xlabel("Iteraciones")
    plt.ylabel(metric_name.upper())
    plt.title(f"Curva de aprendizaje ({metric_name.upper()})")
    plt.legend()
    grafica_simple_guardar(fig, nombre)

def plot_pred_vs_real(y_true, y_pred, nombre="04_pred_vs_real_test.png"):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    minv = min(np.min(y_true), np.min(y_pred))
    maxv = max(np.max(y_true), np.max(y_pred))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Real")
    plt.ylabel("Predicción")
    plt.title("Predicción vs Real (Test)")
    grafica_simple_guardar(fig, nombre)

def plot_residuales(y_true, y_pred, nombre="05_residuales_hist.png"):
    resid = y_true - y_pred
    fig = plt.figure(figsize=(8, 5))
    plt.hist(resid, bins=20, edgecolor="black")
    plt.title("Histograma de residuales (Test)")
    plt.xlabel("Residual")
    plt.ylabel("Frecuencia")
    grafica_simple_guardar(fig, nombre)

def plot_importancia_ganancia(booster, feature_names, top=25, nombre="06_feature_importance_gain.png"):
    imp_gain = booster.get_score(importance_type="gain")
    if not imp_gain:
        return
    mapped = {}
    for k, v in imp_gain.items():
        if k.startswith("f"):
            try:
                idx = int(k[1:])
                feat = feature_names[idx] if idx < len(feature_names) else k
            except Exception:
                feat = k
        else:
            feat = k
        mapped[feat] = v
    items = sorted(mapped.items(), key=lambda x: x[1], reverse=True)[:top]
    labels = [it[0] for it in items][::-1]
    vals = [it[1] for it in items][::-1]
    fig = plt.figure(figsize=(8, max(4, 0.3 * len(labels) + 2)))
    plt.barh(range(len(labels)), vals)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Gain")
    plt.title("Importancia por ganancia (XGBoost)")
    grafica_simple_guardar(fig, nombre)

def plot_importancia_permutacion(pipeline, X_test, y_test, top=25, nombre="07_feature_importance_permutation.png", ruta_reporte_txt=None):
    if skl_permutation_importance is None:
        if ruta_reporte_txt:
            with open(ruta_reporte_txt, "a", encoding="utf-8") as f:
                f.write("\n[AVISO] sklearn.inspection.permutation_importance no disponible; se omite esta gráfica.\n")
        return
    result = skl_permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=os.cpu_count())
    importancias = result.importances_mean
    stds = result.importances_std
    pre = pipeline.named_steps["preprocessor"]
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feat_{i}" for i in range(len(importancias))])
    idx_top = np.argsort(importancias)[::-1][:top]
    labels = feature_names[idx_top][::-1]
    vals = importancias[idx_top][::-1]
    errs = stds[idx_top][::-1]
    fig = plt.figure(figsize=(8, max(4, 0.3 * len(labels) + 2)))
    plt.barh(range(len(labels)), vals, xerr=errs)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Importancia (perm)")
    plt.title("Importancia por permutación (Test)")
    grafica_simple_guardar(fig, nombre)

def make_ohe():
    # OneHotEncoder con fallback por compatibilidad de scikit-learn
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


# =========================
# MAIN
# =========================
def main():
    np.random.seed(42)

    csv_path = encontrar_csv(CANDIDATE_CSV_PATHS)
    df = leer_csv_robusto(csv_path)
    df = limpiar_nombres_columnas(df)

    # Objetivo: última columna
    target_col = df.columns[-1]
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    df = df[~y_raw.isna()].copy()
    y = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    X = df.drop(columns=[target_col])

    # Variables numéricas y categóricas
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Estratificación aproximada
    try:
        strat_labels = estratificar_regresion(y, bins=10)
    except Exception:
        strat_labels = None

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat_labels
    )

    # Preprocesador
    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe())
        ]), cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # ===== EDA & Gráficas iniciales =====
    ruta_reporte_txt = os.path.join(OUTPUT_DIR, "eda_model_report.txt")
    resumen_eda(df, target_col, ruta_reporte_txt)
    plot_distribucion_objetivo(y, "01_target_distribution.png")
    if len(num_cols) >= 2:
        plot_correlacion_numerica(df[num_cols + [target_col]], target_col, "02_correlation_heatmap.png")

    # ===== Baseline CV (RMSE) =====
    xgb_base = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=1.0,
        random_state=42,
        n_jobs=os.cpu_count(),
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",  # definir métrica en el constructor
    )

    pipe_cv = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_base)
    ])

    # Usar neg_mean_squared_error y convertir a RMSE para máxima compatibilidad
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipe_cv, X_train, y_train,
        scoring="neg_mean_squared_error",
        cv=kf, n_jobs=os.cpu_count()
    )
    cv_rmse = np.sqrt(-cv_scores)
    with open(ruta_reporte_txt, "a", encoding="utf-8") as f:
        f.write("\n===== Baseline CV (5 folds) =====\n")
        f.write(f"RMSE por fold: {np.round(cv_rmse, 4).tolist()}\n")
        f.write(f"RMSE promedio: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}\n")

    # ===== Train/Valid para Early Stopping =====
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42,
        stratify=estratificar_regresion(y_train, bins=10) if len(pd.Series(y_train).unique()) > 10 else None
    )

    preprocessor.fit(X_tr)
    X_tr_enc = preprocessor.transform(X_tr)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(X_tr_enc.shape[1])])

    # ===== Modelo final =====
    xgb_final = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=2.0,
        min_child_weight=1.0,
        random_state=42,
        n_jobs=os.cpu_count(),
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
    )

    # Early stopping cuando sea posible
    fitted_with_es = False
    try:
        xgb_final.fit(
            X_tr_enc, y_tr,
            eval_set=[(X_tr_enc, y_tr), (X_val_enc, y_val)],
            early_stopping_rounds=200,
            verbose=False
        )
        fitted_with_es = True
    except TypeError:
        # Fallback sin ES
        xgb_final.fit(
            X_tr_enc, y_tr,
            eval_set=[(X_tr_enc, y_tr), (X_val_enc, y_val)],
            verbose=False
        )

    # Curva de aprendizaje (si existe)
    try:
        evals_result = xgb_final.evals_result()
        plot_learning_curve(evals_result, metric_name="rmse", nombre="03_learning_curve.png")
    except Exception:
        pass

    # ===== Evaluación en TEST =====
    y_pred_test = xgb_final.predict(X_test_enc)
    rmse, mae, r2 = metricas_regresion(y_test, y_pred_test)

    # Mejor iteración (robusto)
    best_iter = getattr(xgb_final, "best_iteration", None)
    if best_iter is None:
        best_iter = getattr(xgb_final, "best_ntree_limit", None)
    if best_iter is None:
        try:
            best_iter = xgb_final.get_booster().best_iteration
        except Exception:
            best_iter = "N/D"

    with open(ruta_reporte_txt, "a", encoding="utf-8") as f:
        f.write("\n===== Resultados en Test =====\n")
        f.write(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\n")
        f.write(f"Best Iteration: {best_iter}\n")
        f.write(f"xgboost.__version__: {xgb.__version__}\n")
        f.write(f"Early stopping aplicado: {fitted_with_es}\n")

    # Gráficas finales
    plot_pred_vs_real(y_test.values, y_pred_test, "04_pred_vs_real_test.png")
    plot_residuales(y_test.values, y_pred_test, "05_residuales_hist.png")

    booster = xgb_final.get_booster()
    plot_importancia_ganancia(booster, feature_names, top=25, nombre="06_feature_importance_gain.png")

    pipeline_final = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_final)
    ])
    plot_importancia_permutacion(
        pipeline_final, X_test, y_test, top=25,
        nombre="07_feature_importance_permutation.png",
        ruta_reporte_txt=ruta_reporte_txt
    )

    modelo_path = os.path.join(OUTPUT_DIR, "xgb_pipeline_calificacion.joblib")
    joblib.dump(pipeline_final, modelo_path)

    # Console summary
    print("=== PROCESO COMPLETADO ===")
    print(f"- CSV leído desde: {csv_path}")
    print(f"- Columnas numéricas: {len(num_cols)} | categóricas: {len(cat_cols)} | total: {X.shape[1]}")
    print(f"- Objetivo: '{target_col}' | muestras totales: {len(df)}")
    print(f"- CV RMSE (5-fold): {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    print(f"- TEST -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    print(f"- Mejor iteración (early stopping): {best_iter}")
    print(f"- xgboost.__version__: {xgb.__version__}")
    print(f"- Reporte: {ruta_reporte_txt}")
    print(f"- Modelo guardado: {modelo_path}")
    print(f"- Gráficas guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

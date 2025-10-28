# -*- coding: utf-8 -*-
# REGRESIÓN LINEAL SÚPER SENCILLA (paso a paso, sin funciones)
# - Lee el mismo CSV (última columna = calificación del examen).
# - Limpia nombres de columnas.
# - Separa numéricas y categóricas.
# - Imputa faltantes de forma simple y mapea categóricas a numéricas con One-Hot.
# - Entrena LinearRegression.
# - Evalúa con RMSE y R^2.
# - Genera varias gráficas didácticas.
# Requisitos: pandas, numpy, scikit-learn, matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para guardar imágenes sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 1) RUTAS
# ---------------------------------------------------------
OUTPUT_DIR = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\econometria-2\images\concurso"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rutas candidatas del dataset (usa la que exista en tu equipo)
csv_candidates = [
    r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\econometria-2\data\desmpeno_escolar.csv",
    r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\econometria-2\datasets\desmpeno_escolar.csv",
    r".\desmpeno_escolar.csv",
    r"./desmpeno_escolar.csv",
    "/mnt/data/desmpeno_escolar.csv",
]
CSV_PATH = None
for p in csv_candidates:
    if os.path.exists(p):
        CSV_PATH = p
        break
if CSV_PATH is None:
    raise FileNotFoundError("No encontré el CSV. Ajusta la ruta en 'csv_candidates'.")

# ---------------------------------------------------------
# 2) LECTURA Y LIMPIEZA BÁSICA
# ---------------------------------------------------------
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
except Exception:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")

# Limpio nombres de columnas (espacios, saltos de línea)
df.columns = [" ".join(str(c).strip().replace("\n", " ").split()) for c in df.columns]

# Tomo la ÚLTIMA columna como variable objetivo (calificación del examen)
target_col = df.columns[-1]
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])  # quito filas sin target

# ---------------------------------------------------------
# 3) EDA MUY BREVE
# ---------------------------------------------------------
# Histograma del target
plt.figure(figsize=(8,5))
plt.hist(df[target_col], bins=20, edgecolor="black")
plt.title("Distribución de la calificación")
plt.xlabel("Calificación"); plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "linreg_01_hist_target.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# 4) SEPARO FEATURES Y TARGET
# ---------------------------------------------------------
X = df.drop(columns=[target_col]).copy()
y = df[target_col].astype(float).copy()

# Identifico columnas numéricas y categóricas
num_cols = list(X.select_dtypes(include=[np.number]).columns)
cat_cols = [c for c in X.columns if c not in num_cols]

# ---------------------------------------------------------
# 5) IMPUTACIÓN SUPER SIMPLE
# ---------------------------------------------------------
# Numéricas: reemplazo NaN por la mediana de cada columna
for c in num_cols:
    if X[c].isna().any():
        med = X[c].median()
        X[c] = X[c].fillna(med)

# Categóricas: convierto a string y reemplazo NaN por "Missing"
for c in cat_cols:
    X[c] = X[c].astype(str).fillna("Missing")

# ---------------------------------------------------------
# 6) MAPEOS CATEGÓRICOS A NUMÉRICOS (DIDÁCTICO)
#    Hay 2 formas sencillas:
#    A) One-Hot Encoding (recomendado para regresión) -> se usa en el modelo.
#    B) Un mapeo "label -> número" (solo como ejemplo educativo; NO lo usamos para entrenar).
# ---------------------------------------------------------

# A) ONE-HOT (crea columnas 0/1 por cada categoría). drop_first=True evita colinealidad perfecta.
X_onehot = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# B) EJEMPLO EDUCATIVO de label mapping (NO se usa en el modelo)
if len(cat_cols) > 0:
    ejemplo_col = cat_cols[0]
    categorias_unicas = sorted(X[ejemplo_col].unique().tolist())
    mapping = {cat: i for i, cat in enumerate(categorias_unicas)}
    X[f"{ejemplo_col}_labelmap_ejemplo"] = X[ejemplo_col].map(mapping)
    # Guardamos el mapeo a texto para que lo veas
    with open(os.path.join(OUTPUT_DIR, "linreg_label_mapping_ejemplo.txt"), "w", encoding="utf-8") as f:
        f.write(f"Columna: {ejemplo_col}\n\n")
        for k, v in mapping.items():
            f.write(f"{k} -> {v}\n")

# ---------------------------------------------------------
# 7) TRAIN / TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_onehot, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 8) ENTRENAR REGRESIÓN LINEAL
# ---------------------------------------------------------
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Predicciones
y_pred_train = linreg.predict(X_train)
y_pred_test  = linreg.predict(X_test)

# Métricas (RMSE y R^2)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
r2_train   = r2_score(y_train, y_pred_train)
r2_test    = r2_score(y_test,  y_pred_test)

print("=== RESULTADOS REGRESIÓN LINEAL ===")
print(f"Columnas numéricas: {len(num_cols)} | categóricas: {len(cat_cols)} | total features (onehot): {X_onehot.shape[1]}")
print(f"RMSE Train: {rmse_train:.4f} | R2 Train: {r2_train:.4f}")
print(f"RMSE Test : {rmse_test:.4f}  | R2 Test : {r2_test:.4f}")

# ---------------------------------------------------------
# 9) GRÁFICAS DIDÁCTICAS
# ---------------------------------------------------------

# 9.1 Predicción vs Real (Test)
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred_test, alpha=0.7)
miv = min(np.min(y_test), np.min(y_pred_test))
mav = max(np.max(y_test), np.max(y_pred_test))
plt.plot([miv, mav], [miv, mav], 'k--', lw=1)
plt.title("Predicción vs Real (Test) — Regresión Lineal")
plt.xlabel("Real"); plt.ylabel("Predicción")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "linreg_02_pred_vs_real_test.png"), dpi=150)
plt.close()

# 9.2 Histograma de residuales (Test)
resid_test = y_test - y_pred_test
plt.figure(figsize=(8,5))
plt.hist(resid_test, bins=20, edgecolor="black")
plt.title("Histograma de residuales (Test)")
plt.xlabel("Residual = Real - Predicción"); plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "linreg_03_residuales_hist.png"), dpi=150)
plt.close()

# 9.3 Residuales vs Predicción (busca patrones)
plt.figure(figsize=(8,5))
plt.scatter(y_pred_test, resid_test, alpha=0.7)
plt.axhline(0, color="k", linestyle="--", lw=1)
plt.title("Residuales vs Predicción (Test)")
plt.xlabel("Predicción"); plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "linreg_04_residuales_vs_pred.png"), dpi=150)
plt.close()
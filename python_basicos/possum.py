# -*- coding: utf-8 -*-
"""
Análisis exploratorio y regresión lineal para predecir 'age'
Dataset: possum.csv

Requisitos:
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn (solo para métricas)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Estilo general de las gráficas
sns.set_theme(style="whitegrid", context="talk")

# -------------------------------------------------------------------
# 1. Rutas: CSV y carpeta de plots
# -------------------------------------------------------------------
csv_path = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\python_basicos\possum.csv"

# Carpeta base (python_basicos) y carpeta de plots
sv_path = os.path.dirname(csv_path)
plots_dir = os.path.join(sv_path, "plots")
os.makedirs(plots_dir, exist_ok=True)

print(f"Las gráficas se guardarán en: {plots_dir}")

# -------------------------------------------------------------------
# 2. Cargar datos
# -------------------------------------------------------------------
df = pd.read_csv(csv_path)

print("\n=== Vista general del dataset ===")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print("\nPrimeras filas:")
print(df.head())

print("\nInformación de tipos de datos:")
print(df.info())

print("\n=== Valores faltantes por columna ===")
print(df.isna().sum())

print("\n=== Estadísticas descriptivas variables numéricas ===")
print(df.describe().T)

# Identificar columnas categóricas y numéricas
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# -------------------------------------------------------------------
# 3. Distribución de variables categóricas con etiquetas
# -------------------------------------------------------------------
if len(cat_cols) > 0:
    print("\n=== Distribución de variables categóricas ===")
    for col in cat_cols:
        print(f"\nColumna: {col}")
        print(df[col].value_counts())

        plt.figure(figsize=(6, 4))
        ax = sns.countplot(data=df, x=col)
        plt.title(f"Frecuencias de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")

        # Etiquetas con conteos encima de cada barra
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"count_{col}.png"),
                    dpi=300, bbox_inches="tight")
        plt.show()

# -------------------------------------------------------------------
# 4. Histograma de age con media y mediana
# -------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.histplot(df["age"], kde=True, bins=10)
plt.title("Distribución de la edad (age)")
plt.xlabel("age")
plt.ylabel("Frecuencia")

mean_age = df["age"].mean()
median_age = df["age"].median()
plt.axvline(mean_age, color="red", linestyle="--", label=f"Media = {mean_age:.2f}")
plt.axvline(median_age, color="green", linestyle=":", label=f"Mediana = {median_age:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "hist_age.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# 5. Matriz de correlación con valores
# -------------------------------------------------------------------
# Usamos todas las columnas numéricas; corr() maneja NaN internamente
num_cols_corr = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
corr_matrix = df[num_cols_corr].corr()

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    square=True,
    cbar_kws={"shrink": 0.8}
)
plt.title("Matriz de correlación (variables numéricas)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "corr_matrix.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# Correlación de cada variable numérica con age (como barplot)
corr_age = corr_matrix["age"].sort_values(ascending=False)
print("\n=== Correlación de las variables numéricas con 'age' ===")
print(corr_age)

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=corr_age.values, y=corr_age.index, orient="h")
plt.title("Correlación de variables numéricas con age")
plt.xlabel("Correlación con age")
plt.ylabel("Variable")

for i, v in enumerate(corr_age.values):
    ax.text(
        v + 0.01 * np.sign(v),
        i,
        f"{v:.2f}",
        va="center"
    )

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "corr_age_barplot.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# 6. Gráficas age vs X con regresión (X elegidas MANUALMENTE)
# -------------------------------------------------------------------
# Aquí eliges las X: están hardcodeadas las 3 primeras de la gráfica
selected_numeric_features_for_plots = ["belly", "chest", "hdlngth"]

print("\nVariables seleccionadas para scatter con regresión:", selected_numeric_features_for_plots)

for col in selected_numeric_features_for_plots:
    plt.figure(figsize=(7, 5))
    ax = sns.regplot(
        data=df,
        x=col,
        y="age",
        line_kws={"color": "red"}
    )
    plt.title(f"Relación entre age y {col}")
    plt.xlabel(col)
    plt.ylabel("age")

    # Cálculo de pendiente y R² de regresión simple age ~ col
    x = df[col].values
    y = df["age"].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) > 1:
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        r = np.corrcoef(x_clean, y_clean)[0, 1]
        r2_simple = r ** 2
        text_str = f"Pendiente: {slope:.3f}\nR² simple: {r2_simple:.3f}"
    else:
        text_str = "No hay suficientes datos"

    ax.text(
        0.05, 0.95,
        text_str,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"scatter_age_vs_{col}.png"),
                dpi=300, bbox_inches="tight")
    plt.show()

# -------------------------------------------------------------------
# 7. Boxplots de age por variables categóricas
# -------------------------------------------------------------------
for col in cat_cols:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x=col, y="age")
    plt.title(f"Distribución de age por {col}")
    plt.xlabel(col)
    plt.ylabel("age")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"box_age_by_{col}.png"),
                dpi=300, bbox_inches="tight")
    plt.show()

# -------------------------------------------------------------------
# 8. Preparación de datos para el modelo (sin tratamiento explícito)
# -------------------------------------------------------------------
# Variable dependiente y regresores tomados DIRECTO de df
y = df["age"]

# Aquí eliges las X DEL MODELO (mismas 3, hardcodeadas)
selected_numeric_features = ["belly", "chest", "hdlngth"]

# Construimos X solo con esas columnas
X = df[selected_numeric_features]

print("\n=== X numéricas usadas en el modelo ===")
print(selected_numeric_features)
print("Tamaño de X:", X.shape)

# Añadimos constante para el intercepto
X_sm = sm.add_constant(X)

# -------------------------------------------------------------------
# 9. Ajuste del modelo OLS SIN dropna explícito
# -------------------------------------------------------------------
# missing='drop' hace que statsmodels ignore internamente filas con NaN
ols_model = sm.OLS(y, X_sm, missing='drop').fit()

print("\n=== Resumen del modelo OLS (statsmodels) ===")
print(ols_model.summary())

# Predicciones in-sample: usamos las que el modelo ya calculó
# (solo para las observaciones efectivamente usadas)
y_used = ols_model.model.endog        # y sin las filas con NaN
y_pred = ols_model.fittedvalues       # predicciones correspondientes

# -------------------------------------------------------------------
# 10. Métricas en la muestra completa usada por el modelo
# -------------------------------------------------------------------
mae = mean_absolute_error(y_used, y_pred)
mse = mean_squared_error(y_used, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_used, y_pred)

print("\n=== Métricas en la muestra completa (in-sample) ===")
print(f"MAE  (error absoluto medio): {mae:.3f}")
print(f"RMSE (raíz del error cuadrático medio): {rmse:.3f}")
print(f"R²   (coeficiente de determinación): {r2:.3f}")
print(f"R² (statsmodels): {ols_model.rsquared:.3f}")
print(f"R² ajustado (statsmodels): {ols_model.rsquared_adj:.3f}")

# -------------------------------------------------------------------
# 11. Análisis de residuos (en las observaciones usadas)
# -------------------------------------------------------------------
residuals = y_used - y_pred

plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, linestyle="--", color="red")
plt.xlabel("Predicciones (muestra usada)")
plt.ylabel("Residuos")
plt.title("Residuos vs predicciones (muestra usada)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "residuals_vs_pred_full.png"),
            dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True, bins=10)
plt.title("Distribución de residuos (muestra usada)")
plt.xlabel("Residuo")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "residuals_hist_full.png"),
            dpi=300, bbox_inches="tight")
plt.show()

print("\nAnálisis completado. Revisa las gráficas en la carpeta 'plots'.")

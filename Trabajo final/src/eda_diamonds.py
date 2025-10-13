# -*- coding: utf-8 -*-
"""
Autor: Humberto Silva Baltazar

eda_diamonds.py

Análisis Exploratorio de Datos (EDA) robusto para el dataset de diamantes con
características ingenierizadas. Lee un archivo Parquet, genera visualizaciones
y reportes estructurados, y guarda logs y figuras en rutas específicas.
"""

from __future__ import annotations

import os
import sys
import io
import math
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================
# Configuración por defecto
# =========================

INPUT_PARQUET = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\parquets\diamonds_features_20251012_155956.parquet"
PLOTS_DIR     = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\plots"
LOGS_DIR      = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\logs"

SCRIPT_NAME   = "eda_diamonds.py"


# =========================
# Utilidades de formato
# =========================

def sep_line(width: int = 120, char: str = "-") -> str:
    return char * width

def center_text(text: str, width: int = 120) -> str:
    return text.center(width)

def human_int(x: int) -> str:
    return f"{int(x):,}".replace(",", ",")

def human_float(x: float, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)

def human_bytes(num_bytes: int) -> str:
    units = ["bytes", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:,.2f} {u}"
        size /= 1024.0

def pct(n: int, d: int, nd: int = 2) -> str:
    if d == 0:
        return "0.00%"
    return f"{(n/d)*100:.{nd}f}%"

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ================================================
# ReportBuffer: captura y volcado de texto a disco
# ================================================

class ReportBuffer:
    def __init__(self):
        self._buf = io.StringIO()

    def write(self, text: str = ""):
        self._buf.write(text + ("\n" if not text.endswith("\n") else ""))

    def get_value(self) -> str:
        return self._buf.getvalue()

    def dump_to_file(self, path: Path):
        path.write_text(self.get_value(), encoding="utf-8")


# ====================================
# Diccionario de variables dinámico
# ====================================

def variable_dictionary_from_columns(cols: List[str]) -> Dict[str, Dict[str, str]]:
    known: Dict[str, Dict[str, str]] = {
        "index": {"tipo": "Numérica", "descripcion": "Índice o identificador de fila original."},
        "price": {"tipo": "Numérica", "descripcion": "Precio en dólares."},
        "carat": {"tipo": "Numérica", "descripcion": "Peso del diamante en quilates."},
        "cut": {"tipo": "Categórica", "descripcion": "Calidad de corte (Fair < Good < Very Good < Premium < Ideal)."},
        "color": {"tipo": "Categórica", "descripcion": "Color (J peor → D mejor)."},
        "clarity": {"tipo": "Categórica", "descripcion": "Claridad (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)."},
        "x": {"tipo": "Numérica", "descripcion": "Longitud (mm)."},
        "y": {"tipo": "Numérica", "descripcion": "Ancho (mm)."},
        "z": {"tipo": "Numérica", "descripcion": "Profundidad (mm)."},
        "depth": {"tipo": "Numérica", "descripcion": "Porcentaje de profundidad total."},
        "table": {"tipo": "Numérica", "descripcion": "Ancho de la mesa (parte superior) en %."},
        # Features del ETL:
        "fe_cut_ord": {"tipo": "Numérica", "descripcion": "Codificación ordinal del corte."},
        "fe_color_ord": {"tipo": "Numérica", "descripcion": "Codificación ordinal del color."},
        "fe_clarity_ord": {"tipo": "Numérica", "descripcion": "Codificación ordinal de la claridad."},
        "fe_volume_mm3": {"tipo": "Numérica", "descripcion": "Volumen aproximado (x*y*z)."},
        "fe_area_mm2": {"tipo": "Numérica", "descripcion": "Área aproximada de la cara superior (x*y)."},
        "fe_spread_mm": {"tipo": "Numérica", "descripcion": "Promedio de x e y; 'spread' visible."},
        "fe_aspect_ratio": {"tipo": "Numérica", "descripcion": "Relación x/y; ~1 indica forma cercana a circular."},
        "fe_invalid_dims": {"tipo": "Booleana", "descripcion": "Banderín si alguna dimensión x,y,z ≤ 0."},
        "fe_depth_pct_recalc": {"tipo": "Numérica", "descripcion": "Profundidad recalculada: 100*z/((x+y)/2)."},
        "fe_depth_pct_diff": {"tipo": "Numérica", "descripcion": "Diferencia depth vs depth recalculado."},
        "fe_depth_pct_is_consistent": {"tipo": "Booleana", "descripcion": "Consistencia entre depth y el recalculado."},
        "fe_price_per_carat": {"tipo": "Numérica", "descripcion": "Precio normalizado por quilate."},
        "fe_log_price": {"tipo": "Numérica", "descripcion": "Logaritmo del precio."},
        "fe_log_carat": {"tipo": "Numérica", "descripcion": "Logaritmo de carat."},
        "fe_log_price_per_carat": {"tipo": "Numérica", "descripcion": "Logaritmo del precio por quilate."},
        "fe_depth_dev": {"tipo": "Numérica", "descripcion": "Desviación de depth respecto a ~61.5%."},
        "fe_table_dev": {"tipo": "Numérica", "descripcion": "Desviación de table respecto a ~57%."},
        "fe_table_to_depth_ratio": {"tipo": "Numérica", "descripcion": "Relación table/depth."},
        "fe_quality_score": {"tipo": "Numérica", "descripcion": "Índice sintético de calidad ponderado."},
        "fe_carat_bin": {"tipo": "Categórica", "descripcion": "Rangos discretos de carat."},
        "fe_carat_x_quality": {"tipo": "Numérica", "descripcion": "Interacción carat * quality_score."},
        "fe_ppc_z_by_cqc": {"tipo": "Numérica", "descripcion": "Z-score de PPC por (cut,color,clarity)."},
    }
    out = {}
    for c in cols:
        out[c] = known.get(c, {
            "tipo": "Desconocido",
            "descripcion": "Descripción no disponible (columna no documentada explícitamente)."
        })
    return out


# ==============================
# Carga de datos (Parquet → DF)
# ==============================

def read_parquet(path: Path) -> pd.DataFrame:
    """Lee Parquet con engine pyarrow."""
    return pd.read_parquet(path, engine="pyarrow")


# ==========================
# EDA: funciones de gráficos
# ==========================

def save_fig(fig, filepath: Path):
    fig.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_column_distribution(df: pd.DataFrame, n_graph_shown: int, n_graph_per_row: int,
                                 plots_dir: Path, prefix: str):
    """
    Histogramas para numéricas y barras para categóricas (baja cardinalidad).
    Las columnas booleanas se grafican como categóricas (barras).
    """
    nunique = df.nunique(dropna=False)

    cols_cat = [c for c in df if (not pd.api.types.is_numeric_dtype(df[c])) and (1 < nunique[c] <= 50)]
    cols_num = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]

    # Forzar booleanas a categóricas
    bool_cols = [c for c in df if pd.api.types.is_bool_dtype(df[c])]
    cols_num = [c for c in cols_num if c not in bool_cols]
    for c in bool_cols:
        if c not in cols_cat:
            cols_cat.append(c)

    cols = (cols_cat + cols_num)[:n_graph_shown]
    if not cols:
        return None

    nCol = len(cols)
    nRow = math.ceil(nCol / n_graph_per_row)
    fig = plt.figure(figsize=(6 * n_graph_per_row, 4.5 * nRow))

    for i, c in enumerate(cols, start=1):
        ax = fig.add_subplot(nRow, n_graph_per_row, i)
        series = df[c].dropna()

        if pd.api.types.is_bool_dtype(df[c]):
            vc = series.value_counts(dropna=False).sort_index()
            vc.plot(kind="bar", ax=ax)
            ax.set_title(f"{c} (bool)")
            ax.set_ylabel("counts")
            ax.tick_params(axis='x', rotation=0)

        elif pd.api.types.is_numeric_dtype(df[c]):
            if series.nunique() > 1:
                ax.hist(series, bins=40)
            else:
                vc = series.value_counts(dropna=False)
                vc.plot(kind="bar", ax=ax)
            ax.set_title(f"{c} (num)")
            ax.set_ylabel("counts")

        else:
            vc = series.value_counts(dropna=False).sort_values(ascending=False)
            vc.plot(kind="bar", ax=ax)
            ax.set_title(f"{c} (cat)")
            ax.set_ylabel("counts")
            ax.tick_params(axis='x', rotation=90)

    fig.suptitle("Distribución por columna", y=1.02, fontsize=12)
    out = plots_dir / f"{prefix}_per_column_distribution.png"
    save_fig(fig, out)
    return out


def plot_correlation_matrix(df: pd.DataFrame, plots_dir: Path, prefix: str, max_cols: int = 35):
    """
    Matriz de correlación para columnas numéricas (limita cantidad para legibilidad).
    """
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if num.shape[1] < 2:
        return None
    if num.shape[1] > max_cols:
        priority = [c for c in ["price", "carat", "fe_price_per_carat", "fe_quality_score",
                                "fe_log_price", "fe_log_carat", "fe_log_price_per_carat",
                                "x", "y", "z", "depth", "table"] if c in num.columns]
        rest = [c for c in num.columns if c not in priority]
        selected = priority + rest[: max_cols - len(priority)]
        num = num[selected]

    corr = num.corr(numeric_only=True)

    fig = plt.figure(figsize=(max(8, len(num.columns) * 0.45), max(8, len(num.columns) * 0.45)))
    im = plt.imshow(corr, interpolation="nearest")
    plt.title("Matriz de correlación (numéricas)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = range(len(num.columns))
    plt.xticks(ticks, num.columns, rotation=90, fontsize=8)
    plt.yticks(ticks, num.columns, fontsize=8)

    out = plots_dir / f"{prefix}_correlation_matrix.png"
    save_fig(fig, out)
    return out


def plot_scatter_matrix(df: pd.DataFrame, plots_dir: Path, prefix: str):
    """
    Scatter matrix (solo numéricas). Se limita a 10 columnas para evitar matrices gigantes,
    pero **no** se muestrean filas (usa todo el dataset).
    """
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    if num.empty:
        return None

    cols = list(num.columns)
    if len(cols) > 10:
        priority = [c for c in ["price", "carat", "fe_price_per_carat",
                                "fe_log_price", "fe_log_carat", "fe_log_price_per_carat",
                                "x", "y", "z"] if c in cols]
        rest = [c for c in cols if c not in priority]
        cols = priority + rest[: (10 - len(priority))]
        num = num[cols]

    # Usar todo el dataset, puntos pequeños y marcador '.'
    fig = plt.figure(figsize=(max(8, len(cols) * 1.5), max(8, len(cols) * 1.5)))
    pd.plotting.scatter_matrix(
        num, alpha=0.6, figsize=fig.get_size_inches(), diagonal='kde', marker='.'
    )
    plt.suptitle("Scatter & Density", y=1.02)
    out = plots_dir / f"{prefix}_scatter_matrix.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def plot_3d_xyz(df: pd.DataFrame, plots_dir: Path, prefix: str):
    """
    Gráfico 3D de x, y, z. Sin muestreo: usa todas las filas disponibles.
    """
    if not all(c in df.columns for c in ["x", "y", "z"]):
        return None

    data = df[["x", "y", "z"]].replace([np.inf, -np.inf], np.nan).dropna()
    extra_label = None

    if "cut" in df.columns:
        extra_label = df.loc[data.index, "cut"]
    elif "carat" in df.columns:
        extra_label = df.loc[data.index, "carat"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    if extra_label is not None and not pd.api.types.is_numeric_dtype(extra_label):
        codes = pd.Categorical(extra_label).codes
        ax.scatter(data["x"], data["y"], data["z"], c=codes, s=2)  
        categories = pd.Categorical(extra_label).categories
        handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=4) for _ in categories]
        ax.legend(handles, [str(c) for c in categories], title="cut", loc="best")
    else:
        c = extra_label if extra_label is not None else data["z"]
        sc = ax.scatter(data["x"], data["y"], data["z"], c=c, s=2)  
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("3D: dimensiones físicas")
    out = plots_dir / f"{prefix}_3d_xyz.png"
    save_fig(fig, out)
    return out


def plot_basic_relationships(df: pd.DataFrame, plots_dir: Path, prefix: str):
    """
    Relación precio vs carat; precio/quilate vs carat; boxplots por categorías (sin muestreo).
    """
    outs = []

    # price vs carat (sin muestreo)
    dsmall = df[["price", "carat"]].dropna()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(dsmall["carat"], dsmall["price"], s=3, alpha=0.5)  
    ax.set_xlabel("carat")
    ax.set_ylabel("price")
    ax.set_title("Precio vs. Carat")
    out = plots_dir / f"{prefix}_price_vs_carat.png"
    save_fig(fig, out)
    outs.append(out)

    # price_per_carat vs carat (si existe)
    if "fe_price_per_carat" in df.columns:
        dppc = df[["fe_price_per_carat", "carat"]].dropna()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(dppc["carat"], dppc["fe_price_per_carat"], s=3, alpha=0.5)  
        ax.set_xlabel("carat")
        ax.set_ylabel("price per carat")
        ax.set_title("Precio por quilate vs. Carat")
        out = plots_dir / f"{prefix}_ppc_vs_carat.png"
        save_fig(fig, out)
        outs.append(out)

    # Boxplot PPC por categorías (sin muestreo)
    def boxplot_by_cat(val_col: str, cat_col: str, fname: str):
        sub = df[[val_col, cat_col]].dropna()
        if sub.empty:
            return None
        groups = [sub[sub[cat_col] == g][val_col] for g in sub[cat_col].dropna().unique()]
        labels = list(sub[cat_col].dropna().unique())
        fig = plt.figure(figsize=(max(7, len(labels) * 1.2), 5))
        ax = fig.add_subplot(111)
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title(f"{val_col} por {cat_col}")
        ax.set_ylabel(val_col)
        ax.tick_params(axis='x', rotation=90)
        out = plots_dir / f"{prefix}_{fname}.png"
        save_fig(fig, out)
        return out

    for cat in ["cut", "color", "clarity", "fe_carat_bin"]:
        if "fe_price_per_carat" in df.columns and cat in df.columns:
            o = boxplot_by_cat("fe_price_per_carat", cat, f"ppc_by_{cat}")
            if o: outs.append(o)

    return outs


def plot_missing_values(df: pd.DataFrame, plots_dir: Path, prefix: str):
    na_counts = df.isna().sum()
    if (na_counts > 0).any():
        fig = plt.figure(figsize=(max(7, len(df.columns) * 0.4), 5))
        ax = fig.add_subplot(111)
        na_counts.plot(kind="bar", ax=ax)
        ax.set_title("Valores nulos por columna")
        ax.set_ylabel("conteo nulos")
        ax.tick_params(axis='x', rotation=90)
        out = plots_dir / f"{prefix}_missing_values.png"
        save_fig(fig, out)
        return out
    return None


def plot_outlier_rates(df: pd.DataFrame, plots_dir: Path, prefix: str):
    """
    Detecta outliers por IQR para columnas numéricas y grafica el % de outliers.
    """
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if num.empty:
        return None

    rates = {}
    for c in num.columns:
        s = num[c].dropna()
        if s.shape[0] < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out_cnt = ((s < lower) | (s > upper)).sum()
        rates[c] = 100.0 * out_cnt / s.shape[0]

    if not rates:
        return None

    sr = pd.Series(rates).sort_values(ascending=False)
    fig = plt.figure(figsize=(max(8, len(sr) * 0.4), 5))
    ax = fig.add_subplot(111)
    sr.plot(kind="bar", ax=ax)
    ax.set_title("Porcentaje de outliers por columna numérica (IQR)")
    ax.set_ylabel("% outliers")
    ax.tick_params(axis='x', rotation=90)
    out = plots_dir / f"{prefix}_outlier_rates.png"
    save_fig(fig, out)
    return out


# ==========================
# EDA: funciones de texto/tabla
# ==========================

def print_variable_dictionary(cols_meta: Dict[str, Dict[str, str]], rb: ReportBuffer):
    rb.write(sep_line())
    rb.write(center_text("Diccionario de Variables"))
    rb.write(sep_line())
    rb.write(f"{'Variable':<25}{'Tipo':<15}{'Descripción'}")
    rb.write(sep_line())
    for col, meta in cols_meta.items():
        rb.write(f"{col:<25}{meta.get('tipo',''):<15}{meta.get('descripcion','')}")
    rb.write("")


def kpis_globales(df: pd.DataFrame, rb: ReportBuffer):
    n_rows, n_cols = df.shape
    dups = df.duplicated().sum()
    n_null = int(df.isna().sum().sum())
    mem = int(df.memory_usage(deep=True).sum())

    rb.write(sep_line())
    rb.write(center_text("KPIs Globales del Dataset"))
    rb.write(sep_line())
    rb.write(f"{'KPI':<30}{'Valor'}")
    rb.write(sep_line())
    rb.write(f"{'Registros':<30}{human_int(n_rows)}")
    rb.write(f"{'Columnas':<30}{human_int(n_cols)}")
    rb.write(f"{'Duplicados (filas)':<30}{human_int(dups)}")
    rb.write(f"{'Nulos totales':<30}{human_int(n_null)}")
    rb.write(f"{'Memoria aprox.':<30}{human_bytes(mem)}")
    rb.write("")


def resumen_por_columna(df: pd.DataFrame, rb: ReportBuffer, topk: int = 3):
    rb.write(sep_line())
    rb.write(center_text("Resumen por Columna"))
    rb.write(sep_line())
    header = f"{'Columna':<25}{'Tipo':<12}{'Nulos':>8}{'Únicos':>10}  {'Top-1':<28}{'Top-2':<28}{'Top-3':<28}"
    rb.write(header)
    rb.write(sep_line())

    n = df.shape[0]
    for col in df.columns:
        s = df[col]
        tipo = "Numérica" if pd.api.types.is_numeric_dtype(s) else ("Booleana" if pd.api.types.is_bool_dtype(s) else "Categórica")
        n_null = s.isna().sum()
        nunique = s.nunique(dropna=False)

        vc = s.value_counts(dropna=False)
        tops = []
        for i in range(min(topk, len(vc))):
            v = vc.index[i]
            cnt = vc.iloc[i]
            label = str(v)
            frac = f"{pct(cnt, n)}"
            tops.append(f"{label[:16]} ({frac})")

        while len(tops) < 3:
            tops.append("")

        rb.write(f"{col:<25}{tipo:<12}{n_null:>8}{nunique:>10}  {tops[0]:<28}{tops[1]:<28}{tops[2]:<28}")
    rb.write("")


def estadisticos_numericos(df: pd.DataFrame, rb: ReportBuffer):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        rb.write("No hay columnas numéricas para estadísticos.")
        rb.write("")
        return

    rb.write(sep_line())
    rb.write(center_text("Estadísticos de Variables Numéricas"))
    rb.write(sep_line())
    header = f"{'Columna':<22}{'min':>12}{'p25':>12}{'p50':>12}{'p75':>12}{'max':>12}{'mean':>14}{'std':>14}"
    rb.write(header)
    rb.write(sep_line())
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
    for col in desc.index:
        row = desc.loc[col]
        rb.write(
            f"{col:<22}"
            f"{human_float(row['min']):>12}"
            f"{human_float(row['25%']):>12}"
            f"{human_float(row['50%']):>12}"
            f"{human_float(row['75%']):>12}"
            f"{human_float(row['max']):>12}"
            f"{human_float(row['mean']):>14}"
            f"{human_float(row['std']):>14}"
        )
    rb.write("")


def calidad_de_datos(df: pd.DataFrame, rb: ReportBuffer):
    rb.write(sep_line())
    rb.write(center_text("Calidad de Datos (Resumen)"))
    rb.write(sep_line())
    rb.write(f"{'Chequeo':<35}{'Detalle/Conteo'}")
    rb.write(sep_line())

    expected_cut = {"Fair", "Good", "Very Good", "Premium", "Ideal"}
    expected_color = {"D", "E", "F", "G", "H", "I", "J"}
    expected_clarity = {"I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"}

    def check_domain(col: str, expected: set) -> str:
        if col not in df.columns:
            return "N/A"
        vals = set(df[col].dropna().unique())
        invalid = vals - expected
        return "OK" if len(invalid) == 0 else f"Valores no esperados: {sorted(invalid)}"

    rb.write(f"{'cut_invalid_values':<35}{check_domain('cut', expected_cut)}")
    rb.write(f"{'color_invalid_values':<35}{check_domain('color', expected_color)}")
    rb.write(f"{'clarity_invalid_values':<35}{check_domain('clarity', expected_clarity)}")

    def count_leq_zero(col: str) -> int:
        if col not in df.columns:
            return -1
        s = pd.to_numeric(df[col], errors="coerce")
        return int((s <= 0).sum())

    price_out_of_range = count_leq_zero("price")
    carat_out_of_range = count_leq_zero("carat")
    x_nonpos = count_leq_zero("x")
    y_nonpos = count_leq_zero("y")
    z_nonpos = count_leq_zero("z")

    rb.write(f"{'price_out_of_range':<35}{max(price_out_of_range, 0)}")
    rb.write(f"{'carat_out_of_range':<35}{max(carat_out_of_range, 0)}")
    rb.write(f"{'x_nonpositive':<35}{max(x_nonpos, 0)}")
    rb.write(f"{'y_nonpositive':<35}{max(y_nonpos, 0)}")
    rb.write(f"{'z_nonpositive':<35}{max(z_nonpos, 0)}")

    def count_outside(col: str, lo: float, hi: float) -> int:
        if col not in df.columns:
            return -1
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return int(((s < lo) | (s > hi)).sum())

    depth_suspect = count_outside("depth", 55.0, 70.0)
    table_suspect = count_outside("table", 50.0, 70.0)

    rb.write(f"{'depth_suspect':<35}{max(depth_suspect, 0)}")
    rb.write(f"{'table_suspect':<35}{max(table_suspect, 0)}")

    if "fe_invalid_dims" in df.columns:
        cnt = int(pd.to_numeric(df["fe_invalid_dims"], errors="coerce").fillna(0).astype(bool).sum())
        rb.write(f"{'fe_invalid_dims_true':<35}{cnt}")

    rb.write("")


def insights_automaticos(df: pd.DataFrame, rb: ReportBuffer):
    rb.write(sep_line())
    rb.write(center_text("Insights Automáticos (Hallazgos Clave)"))
    rb.write(sep_line())

    def try_skew(col: str) -> Tuple[str, str]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                return (col, "Sin datos numéricos no nulos.")
            skew = s.skew()
            if skew > 1.0:
                msg = f"{col}: Distribución sesgada a la derecha (skew={skew:.2f}). Considerar log."
            elif skew < -1.0:
                msg = f"{col}: Distribución sesgada a la izquierda (skew={skew:.2f}). Revisar valores."
            else:
                msg = f"{col}: Sesgo moderado (skew={skew:.2f})."
            return (col, msg)
        return (col, "No es numérica o no existe.")

    for c in ["price", "carat", "fe_price_per_carat", "x", "y", "z", "depth", "table"]:
        col, msg = try_skew(c)
        rb.write(f"- {msg}")

    def corr_pair(a: str, b: str) -> str:
        if a in df.columns and b in df.columns:
            s = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
            if s.shape[0] >= 30:
                r = s[a].corr(s[b])
                return f"{a} ~ {b}: correlación={r:.3f} (n={s.shape[0]})."
        return f"{a} ~ {b}: sin datos suficientes o columnas ausentes."

    rb.write("- Relaciones lineales aproximadas (correlaciones):")
    rb.write(f"  • {corr_pair('price', 'carat')}")
    if "fe_price_per_carat" in df.columns:
        rb.write(f"  • {corr_pair('fe_price_per_carat', 'fe_quality_score')}")
        rb.write(f"  • {corr_pair('fe_price_per_carat', 'carat')}")
    rb.write("")


def vif_analysis(df: pd.DataFrame, rb: ReportBuffer, max_vars: int = 12):
    rb.write(sep_line())
    rb.write(center_text("Diagnóstico de Multicolinealidad (VIF)"))
    rb.write(sep_line())

    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
    if num.shape[1] < 2 or num.shape[0] < 100:
        rb.write("Datos insuficientes para VIF (se requieren varias columnas y ≥100 filas limpias).")
        rb.write("")
        return

    priority = [c for c in ["price", "carat", "fe_price_per_carat", "fe_quality_score",
                            "fe_log_price", "fe_log_carat", "fe_log_price_per_carat",
                            "x", "y", "z", "depth", "table"] if c in num.columns]
    rest = [c for c in num.columns if c not in priority]
    cols = priority + rest[: max_vars - len(priority)]
    X = num[cols].copy()

    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

    rb.write(f"{'Variable':<25}{'VIF':>10}")
    rb.write(sep_line())
    for i in range(X.shape[1]):
        try:
            vif_val = variance_inflation_factor(X.values, i)
            rb.write(f"{X.columns[i]:<25}{vif_val:>10.3f}")
        except Exception as e:
            rb.write(f"{X.columns[i]:<25}{'ERROR':>10}  ({str(e)})")
    rb.write("")


def top_over_under_priced(df: pd.DataFrame, rb: ReportBuffer, k: int = 10):
    if "fe_ppc_z_by_cqc" not in df.columns:
        return

    cols_show = [c for c in ["index", "price", "carat", "cut", "color", "clarity", "fe_price_per_carat", "fe_ppc_z_by_cqc"] if c in df.columns]
    s = df[cols_show].dropna(subset=["fe_ppc_z_by_cqc"])
    if s.empty:
        return

    rb.write(sep_line())
    rb.write(center_text("Diamantes con Precio por Quilate Atípico (Z-score por categoría)"))
    rb.write(sep_line())

    high = s.sort_values("fe_ppc_z_by_cqc", ascending=False).head(k)
    low = s.sort_values("fe_ppc_z_by_cqc", ascending=True).head(k)

    def print_table(title: str, data: pd.DataFrame):
        rb.write(title)
        rb.write("-" * len(title))
        header = " | ".join([f"{c:<12}" for c in data.columns])
        rb.write(header)
        rb.write("-" * len(header))
        for _, row in data.iterrows():
            rb.write(" | ".join([f"{str(row[c])[:12]:<12}" for c in data.columns]))
        rb.write("")

    print_table("Top sobreprecio relativo (Z-score alto):", high)
    print_table("Top descuento relativo (Z-score bajo):", low)


def resumen_bins_carat(df: pd.DataFrame, rb: ReportBuffer):
    if "fe_carat_bin" not in df.columns:
        return
    value_cols = [c for c in ["price", "fe_price_per_carat", "fe_quality_score"] if c in df.columns]
    sub = df[["fe_carat_bin"] + value_cols].dropna(subset=["fe_carat_bin"])
    if sub.empty:
        return

    rb.write(sep_line())
    rb.write(center_text("Resumen por Rangos de Carat (fe_carat_bin)"))
    rb.write(sep_line())

    g = sub.groupby("fe_carat_bin")[value_cols].agg(["count", "mean", "median"])
    rb.write(g.to_string())
    rb.write("")


# ===================
# Flujo principal EDA
# ===================

def main():
    parser = argparse.ArgumentParser(description="EDA robusto para diamonds (Parquet).")
    parser.add_argument("--input", type=str, default=INPUT_PARQUET, help="Ruta al Parquet de entrada.")
    parser.add_argument("--plots", type=str, default=PLOTS_DIR, help="Directorio para guardar figuras.")
    parser.add_argument("--logs", type=str, default=LOGS_DIR, help="Directorio para logs y reporte de texto.")
    parser.add_argument("--seed", type=int, default=123, help="Semilla de aleatoriedad.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    input_path = Path(args.input)
    plots_dir = Path(args.plots)
    logs_dir = Path(args.logs)

    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_ts = now_ts()

    log_path = logs_dir / f"eda_diamonds_{run_ts}.log"
    sys.stdout.write(f"Iniciando EDA. Log: {log_path}\n")
    sys.stdout.flush()

    rb = ReportBuffer()

    rb.write(sep_line())
    rb.write(center_text(f"EDA ROBUSTO PARA DIAMONDS - {run_ts}"))
    rb.write(sep_line())
    rb.write(f"Script: {SCRIPT_NAME}")
    rb.write(f"Input Parquet: {input_path}")
    rb.write(f"Plots dir: {plots_dir}")
    rb.write(f"Logs dir: {logs_dir}")
    rb.write("")

    t0 = time.time()

    try:
        df = read_parquet(input_path)
    except Exception as e:
        msg = f"ERROR al leer Parquet: {e}"
        rb.write(msg)
        log_path.write_text(rb.get_value(), encoding="utf-8")
        print(msg)
        sys.exit(1)

    setattr(df, "dataframeName", input_path.name)

    cols_meta = variable_dictionary_from_columns(list(df.columns))
    print_variable_dictionary(cols_meta, rb)

    kpis_globales(df, rb)

    resumen_por_columna(df, rb, topk=3)

    estadisticos_numericos(df, rb)

    calidad_de_datos(df, rb)

    insights_automaticos(df, rb)

    vif_analysis(df, rb, max_vars=12)

    top_over_under_priced(df, rb, k=10)

    resumen_bins_carat(df, rb)

    prefix = f"diamonds_eda_{run_ts}"
    figs_paths = []

    p = plot_per_column_distribution(df, n_graph_shown=30, n_graph_per_row=5,
                                     plots_dir=plots_dir, prefix=prefix)
    if p: figs_paths.append(p)

    p = plot_correlation_matrix(df, plots_dir=plots_dir, prefix=prefix, max_cols=35)
    if p: figs_paths.append(p)

    p = plot_scatter_matrix(df, plots_dir=plots_dir, prefix=prefix)
    if p: figs_paths.append(p)

    p = plot_3d_xyz(df, plots_dir=plots_dir, prefix=prefix)
    if p: figs_paths.append(p)

    outs = plot_basic_relationships(df, plots_dir=plots_dir, prefix=prefix)
    figs_paths.extend(outs)

    p = plot_missing_values(df, plots_dir=plots_dir, prefix=prefix)
    if p: figs_paths.append(p)

    p = plot_outlier_rates(df, plots_dir=plots_dir, prefix=prefix)
    if p: figs_paths.append(p)

    t1 = time.time()
    rb.write(sep_line())
    rb.write(center_text("Resumen de Ejecución"))
    rb.write(sep_line())
    rb.write(f"Tiempo total: {human_float(t1 - t0, 2)} s")
    rb.write(f"Figuras generadas: {len(figs_paths)}")
    rb.write("")

    report_txt = logs_dir / f"eda_diamonds_reporte_{run_ts}.txt"
    rb.dump_to_file(report_txt)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] EDA finalizado.\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Reporte: {report_txt}\n")
        f.write(f"Figuras individuales:\n")
        for p in figs_paths:
            f.write(f" - {p}\n")

    print(sep_line())
    print(center_text("EDA COMPLETADO"))
    print(sep_line())
    print(f"Reporte de texto: {report_txt}")
    print(f"Log: {log_path}")
    print(f"Total figuras: {len(figs_paths)}")
    print("")


if __name__ == "__main__":
    main()

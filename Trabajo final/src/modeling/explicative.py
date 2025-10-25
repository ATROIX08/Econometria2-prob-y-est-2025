# -*- coding: utf-8 -*-
"""
Autor: Humberto Silva Baltazar
Modelo explicativo diamantes 
OLS (Ordinary Least Squares)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import OLSInfluence
# Usamos la versión de statsmodels (4 valores: jb, p, skew, kurt)
from statsmodels.stats.stattools import jarque_bera as sm_jarque_bera

# --------------------------------------------------------------------------------------
# Configuración de logging (CORREGIDO)
# --------------------------------------------------------------------------------------
import logging

def _build_logger(logs_dir: Path) -> tuple[logging.Logger, str]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = logs_dir / f"explicativo_{run_ts}.log"

    logger = logging.getLogger("explicative_simple")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Usa %(asctime)s en fmt y define datefmt -> evita el error de %Y en fmt
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Consola
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Archivo
    fh = logging.FileHandler(str(log_name), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger, run_ts

# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------
def find_latest_parquet(parquets_dir: Path) -> Path:
    """Encuentra el parquet más reciente con patrón diamonds_features_*.parquet."""
    files = sorted(
        parquets_dir.glob("diamonds_features_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(
            f"No se encontraron parquets 'diamonds_features_*.parquet' en {parquets_dir}"
        )
    return files[0]

def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (como sklearn), usando la media de y_true del propio conjunto."""
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    if sst == 0:
        return np.nan
    return 1.0 - (sse / sst)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Gráficas
# --------------------------------------------------------------------------------------
def plot_resid_vs_fitted(fitted: np.ndarray, resid: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, resid, alpha=0.5, s=10)
    plt.axhline(0, ls="--")
    plt.xlabel("Ajustados (log-precio)")
    plt.ylabel("Residuales")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_qq(resid: np.ndarray, out_path: Path, title: str) -> None:
    fig = sm.qqplot(resid, line="45", fit=True)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def plot_pred_vs_real_price(y_true_price: np.ndarray, y_pred_price: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true_price, y_pred_price, alpha=0.5, s=10)
    min_v = float(np.nanmin([y_true_price.min(), y_pred_price.min()]))
    max_v = float(np.nanmax([y_true_price.max(), y_pred_price.max()]))
    plt.plot([min_v, max_v], [min_v, max_v], ls="--")
    plt.xlabel("Precio real (USD)")
    plt.ylabel("Precio predicho (USD)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    # Rutas base relativas a este archivo
    this_file = Path(__file__).resolve()
    base_dir = this_file.parents[2]  # Trabajo final/
    parquets_dir = base_dir / "output" / "parquets"
    plots_dir   = base_dir / "plots" / "modeling" / "explicativo"
    logs_dir    = base_dir / "output" / "logs" / "modeling" / "explicativo"

    ensure_dirs(plots_dir, logs_dir)
    logger, RUN_TS = _build_logger(logs_dir)

    logger.info("=" * 108)
    logger.info("INICIO — MODELO EXPLICATIVO (Diamantes) — versión simple (solo OLS clásico)")
    logger.info("=" * 108)
    logger.info(f"RUN_TS: {RUN_TS}")
    logger.info(f"Python: {sys.version.split()[0]}, pandas: {pd.__version__}, numpy: {np.__version__}, statsmodels: {sm.__version__}")
    logger.info(f"OS: {os.name}")
    logger.info(f"Proyecto: {base_dir}")
    logger.info(f"Parquets: {parquets_dir}")
    logger.info(f"Plots:    {plots_dir}")
    logger.info(f"Logs:     {logs_dir}")
    logger.info("-" * 110)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    logger.info("Buscando parquet más reciente...")
    dataset_path = find_latest_parquet(parquets_dir)
    logger.info(f"Dataset: {dataset_path}")
    logger.info("Leyendo parquet (pyarrow)...")
    df = pd.read_parquet(dataset_path, engine="pyarrow")
    logger.info(f"Leído | shape={df.shape}")

    # Remover registros inválidos por dimensiones no positivas (x,y,z)
    mask_bad = (df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0)
    n_bad = int(mask_bad.sum())
    if n_bad > 0:
        df = df.loc[~mask_bad].copy()
    logger.info(f"Registros inválidos removidos (dims no positivas): {n_bad} | shape={df.shape}")

    # -------------------------------------------------------------------------
    # Variables del modelo (SIN interacciones)
    # -------------------------------------------------------------------------
    target_var = "fe_log_price"  # log(precio)
    base_formula = (
        "fe_log_price ~ fe_log_carat + fe_depth_dev + fe_table_dev "
        "+ C(cut, Treatment(reference='Ideal')) "
        "+ C(color, Treatment(reference='G')) "
        "+ C(clarity, Treatment(reference='SI1'))"
    )
    cols_needed = [
        "fe_log_price", "fe_log_carat", "fe_depth_dev", "fe_table_dev",
        "cut", "color", "clarity", "price"
    ]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")

    df_model = df[cols_needed].dropna().copy()
    logger.info(f"Filas tras dropna en columnas del modelo: {df_model.shape[0]:,}")
    logger.info("Fórmula usada (simple, sin interacciones):")
    logger.info(base_formula)

    # -------------------------------------------------------------------------
    # Split train / test (80/20) sin sklearn
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    idx_all = np.arange(df_model.shape[0])
    rng.shuffle(idx_all)
    n_train = int(0.8 * len(idx_all))
    idx_train = idx_all[:n_train]
    idx_test  = idx_all[n_train:]

    df_train = df_model.iloc[idx_train].copy()
    df_test  = df_model.iloc[idx_test].copy()
    logger.info(f"Split train/test: train={df_train.shape[0]:,} | test={df_test.shape[0]:,}")

    # -------------------------------------------------------------------------
    # Ajuste OLS (CLÁSICO). SOLO imprimimos este resumen.
    # -------------------------------------------------------------------------
    logger.info("Ajustando OLS (clásico)...")
    ols_res = smf.ols(formula=base_formula, data=df_train).fit()  # nonrobust

    # Predicciones en train/test (en log-precio)
    y_train = df_train[target_var].to_numpy()
    y_test  = df_test[target_var].to_numpy()

    yhat_train_log = ols_res.predict(df_train)
    yhat_test_log  = ols_res.predict(df_test)

    # R² train / test (log-precio)
    r2_train = float(ols_res.rsquared)
    r2_test  = float(r2_score_manual(y_test, yhat_test_log.to_numpy()))

    # AIC / BIC
    aic = float(ols_res.aic)
    bic = float(ols_res.bic)

    logger.info(f"[OLS] R² train={r2_train:.6f} | R² test={r2_test:.6f}")
    logger.info(f"[OLS] AIC={aic:.3f} | BIC={bic:.3f}")

    # -------------------------------------------------------------------------
    # Métricas en TEST (en escala de precio): y_pred = exp(log_pred)
    # -------------------------------------------------------------------------
    y_test_price = np.exp(y_test)
    yhat_test_price = np.exp(yhat_test_log.to_numpy())
    test_mae  = mae(y_test_price, yhat_test_price)
    test_rmse = rmse(y_test_price, yhat_test_price)
    test_mape = mape(y_test_price, yhat_test_price)
    logger.info(f"[OLS] Métricas TEST (escala precio): MAE={test_mae:,.2f} | RMSE={test_rmse:,.2f} | MAPE={test_mape:.2f}%")

    # -------------------------------------------------------------------------
    # Diagnósticos
    # -------------------------------------------------------------------------
    # Jarque–Bera (statsmodels: jb, p, skew, kurt)
    jb_stat, jb_p, skew_val, kurt_val = sm_jarque_bera(ols_res.resid)
    # Breusch–Pagan y White (usando exog del modelo entrenado)
    lm_bp, p_bp, f_bp, fp_bp = het_breuschpagan(ols_res.resid, ols_res.model.exog)
    lm_w, p_w, f_w, fp_w = het_white(ols_res.resid, ols_res.model.exog)

    logger.info(f"[OLS] Jarque–Bera: stat={jb_stat:.4f}, p={jb_p:.4g}, skew={skew_val:.4f}, kurt={kurt_val:.4f}")
    logger.info(f"[OLS] Breusch–Pagan: stat={lm_bp:.4f}, p={p_bp:.4g} | White: stat={lm_w:.4f}, p={p_w:.4g}")

    # Influencias: Cook’s D, leverage, resid estandarizados (top-10)
    infl = OLSInfluence(ols_res)

    # Fuerza a numpy para evitar indexados por label de pandas (FIX al KeyError)
    cooks_d_arr   = np.asarray(infl.cooks_distance[0], dtype=float).reshape(-1)
    leverage_arr  = np.asarray(infl.hat_matrix_diag, dtype=float).reshape(-1)
    std_resid_arr = np.asarray(infl.resid_studentized_internal, dtype=float).reshape(-1)

    # Top-10 por Cook's D (posiciones)
    order = np.argsort(cooks_d_arr)
    top_pos = order[-10:][::-1]  # posiciones (0..n-1) de mayor a menor

    # Índices originales de df_train
    idx_labels = df_train.index.to_numpy()

    top_df = pd.DataFrame({
        "pos": top_pos,
        "idx": idx_labels[top_pos],
        "cooks_d": cooks_d_arr[top_pos],
        "leverage": leverage_arr[top_pos],
        "std_resid": std_resid_arr[top_pos],
    })
    logger.info("TOP 10 influyentes (Cook’s D) [pos=posición en train, idx=índice original]:")
    logger.info("\n" + str(top_df))

    # -------------------------------------------------------------------------
    # Gráficas
    # -------------------------------------------------------------------------
    resid_vs_fitted_path = plots_dir / f"ols_resid_vs_fitted_{RUN_TS}.png"
    qqplot_path          = plots_dir / f"ols_qqplot_{RUN_TS}.png"
    pred_vs_real_path    = plots_dir / f"pred_vs_real_test_ols_{RUN_TS}.png"

    plot_resid_vs_fitted(
        fitted=ols_res.fittedvalues.to_numpy(),
        resid=ols_res.resid.to_numpy(),
        out_path=resid_vs_fitted_path,
        title="OLS clásico — Residuales vs Ajustados"
    )
    logger.info(f"Figura guardada: {resid_vs_fitted_path}")

    plot_qq(
        resid=ols_res.resid.to_numpy(),
        out_path=qqplot_path,
        title="OLS clásico — QQ-plot residuales"
    )
    logger.info(f"Figura guardada: {qqplot_path}")

    plot_pred_vs_real_price(
        y_true_price=y_test_price,
        y_pred_price=yhat_test_price,
        out_path=pred_vs_real_path,
        title="OLS clásico — Predicho vs Real (TEST, precio)"
    )
    logger.info(f"Figura guardada: {pred_vs_real_path}")

    # -------------------------------------------------------------------------
    # Resumen (SOLO el clásico)
    # -------------------------------------------------------------------------
    logger.info("Resumen OLS (clásico):")
    logger.info("\n" + str(ols_res.summary()))

    logger.info("=" * 108)
    logger.info("FIN — MODELO EXPLICATIVO (simple, OLS clásico)")
    logger.info("=" * 108)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Autor: Humberto Silva Baltazar

ETL para diamonds.csv:
- Tipificación, validación y limpieza suave
- Ingeniería de características (features) 
- Reporte tabular ASCII en terminal (y guardado a .txt)
- Guardado de Parquet (base + features) SIN remover filas
- Log detallado a archivo
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import math
import json
import textwrap
import traceback
import logging
import polars as pl

# ==========================
# CONFIGURACIÓN DE RUTAS
# ==========================
PATH_IN_CSV = Path(r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\data\diamonds.csv")
OUT_PARQUET_DIR = Path(r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\parquets")
OUT_LOGS_DIR = Path(r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\7mo Semestre\econometria2_estyprob\codigo-py\Econometria2-prob-y-est-2025\Trabajo final\output\logs")

OUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
OUT_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# LOGGING
# ==========================
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = OUT_LOGS_DIR / f"etl_diamonds_{RUN_TS}.log"
REPORT_FILE = OUT_LOGS_DIR / f"etl_diamonds_reporte_{RUN_TS}.txt"
SCHEMA_JSON = OUT_LOGS_DIR / f"etl_diamonds_schema_{RUN_TS}.json"

logger = logging.getLogger("etl_diamonds")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(fmt)
logger.addHandler(sh)

# ==========================
# BUFFER DE REPORTE (para guardar lo impreso en terminal)
# ==========================
class ReportBuffer:
    def __init__(self):
        self.lines = []
    def write(self, s: str):
        self.lines.append(s)
    def text(self) -> str:
        return "".join(self.lines)

REPORT = ReportBuffer()

# ==========================
# UTILIDADES DE FORMATO
# ==========================
LINE_WIDTH = 120

def sep_line(width: int = LINE_WIDTH, char: str = "-") -> str:
    return char * width

def center_text(text: str, width: int = LINE_WIDTH) -> str:
    return text.center(width)

def emit(block: str, end: str = "\n"):
    print(block, end=end)
    REPORT.write(block + ("" if end == "" else end))

def human_int(n: int | float | None) -> str:
    if n is None:
        return "-"
    try:
        return f"{int(n):,}".replace(",", ",")
    except Exception:
        try:
            return f"{float(n):,.0f}".replace(",", ",")
        except Exception:
            return str(n)

def human_float(x: float | None, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:,.{decimals}f}".replace(",", ",")

def pct(x: float | None, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{100 * x:.{decimals}f}%"

def safe_div(a: float | int, b: float | int) -> float | None:
    try:
        if b == 0:
            return None
        return float(a) / float(b)
    except Exception:
        return None

def crop(s: str, maxlen: int) -> str:
    if s is None:
        return "-"
    s = str(s)
    return (s[: max(0, maxlen - 1)] + "…") if len(s) > maxlen else s

def is_numeric_dtype(dt: pl.DataType) -> bool:
    s = str(dt)
    return any(tok in s for tok in ["Int", "UInt", "Float", "Decimal"])

def is_float_dtype(dt: pl.DataType) -> bool:
    s = str(dt)
    return "Float" in s

# ==========================
# DICCIONARIO DE VARIABLES
# ==========================
def variable_dictionary():
    return {
        "price":   ("Numérica", "Precio en dólares ($326–$18,823)."),
        "carat":   ("Numérica", "Peso del diamante (0.2–5.01)."),
        "cut":     ("Categórica", "Calidad del corte (Fair, Good, Very Good, Premium, Ideal)."),
        "color":   ("Categórica", "Color, de J (peor) a D (mejor)."),
        "clarity": ("Categórica", "Claridad (I1 (peor), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (mejor))."),
        "x":       ("Numérica", "Longitud en mm."),
        "y":       ("Numérica", "Ancho en mm."),
        "z":       ("Numérica", "Profundidad en mm."),
        "depth":   ("Numérica", "Porcentaje de profundidad total."),
        "table":   ("Numérica", "Ancho de la parte superior del diamante."),
    }

EXPECTED_CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
EXPECTED_COLOR = ["J","I","H","G","F","E","D"]  # J peor … D mejor
EXPECTED_CLARITY = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]

MAP_CUT_ORD = {k:i+1 for i,k in enumerate(EXPECTED_CUT)}               # 1=Fair, 5=Ideal
MAP_COLOR_ORD = {k:i+1 for i,k in enumerate(EXPECTED_COLOR)}           # 1=J, 7=D
MAP_CLARITY_ORD = {k:i+1 for i,k in enumerate(EXPECTED_CLARITY)}       # 1=I1, 8=IF

# ==========================
# CARGA Y TIPIFICACIÓN
# ==========================
def load_csv_polars(path: Path) -> pl.DataFrame:
    logger.info(f"Leyendo CSV desde: {path}")
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo CSV: {path}")
    df = pl.read_csv(
        path,
        infer_schema_length=10000,
        try_parse_dates=False,
        null_values=["", "NA", "NaN", "None", "null"],
        encoding="utf8"
    )
    logger.info(f"Leído OK: {df.height:,} filas x {df.width} columnas")
    return df

def coerce_schema(df: pl.DataFrame) -> pl.DataFrame:
    cols = df.columns
    logger.info("Tipificando columnas clave y normalizando nombres…")

    # Normalizar nombres (strip y espacios -> underscores); si queda vacío, usa 'index'
    rename_map = {c: (c.strip().replace(" ", "_") or "index") for c in cols}
    df = df.rename(rename_map)

    if "index" in df.columns and str(df["index"].dtype).startswith(("Int", "UInt")):
        logger.info("Se detectó columna 'index'. Se conservará tal cual (útil para trazabilidad).")

    expected_types = {
        "price": pl.Int64,
        "carat": pl.Float64,
        "cut": pl.Utf8,
        "color": pl.Utf8,
        "clarity": pl.Utf8,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
        "depth": pl.Float64,
        "table": pl.Float64,
    }

    for col, dtype in expected_types.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))

    return df

# ==========================
# VALIDACIONES Y CALIDAD
# ==========================
def _scalar_from_select(df_sel: pl.DataFrame) -> int:
    return int(df_sel.to_series(0).item())

def data_quality_checks(df: pl.DataFrame) -> dict:
    checks = {}

    duplicated_rows = df.height - df.unique().height
    checks["duplicated_rows"] = int(duplicated_rows)

    nulls = {c: _scalar_from_select(df.select(pl.col(c).is_null().sum())) for c in df.columns}
    checks["nulls_by_col"] = nulls
    checks["total_nulls"] = int(sum(nulls.values()))

    nunique = {c: _scalar_from_select(df.select(pl.col(c).n_unique())) for c in df.columns}
    checks["nunique_by_col"] = nunique

    dom_violations = {}
    if "cut" in df.columns:
        dom_violations["cut_invalid_values"] = sorted(list(set(df["cut"].drop_nulls().unique().to_list()) - set(EXPECTED_CUT)))
    if "color" in df.columns:
        dom_violations["color_invalid_values"] = sorted(list(set(df["color"].drop_nulls().unique().to_list()) - set(EXPECTED_COLOR)))
    if "clarity" in df.columns:
        dom_violations["clarity_invalid_values"] = sorted(list(set(df["clarity"].drop_nulls().unique().to_list()) - set(EXPECTED_CLARITY)))
    checks["domain_violations"] = dom_violations

    # Contar también valores == 0 (no-positivos)
    def count_outside(col: str, low=None, high=None, include_low_eq=False, include_high_eq=False) -> int | None:
        if col not in df.columns:
            return None
        expr = None
        if low is not None:
            expr = (pl.col(col) <= low) if include_low_eq else (pl.col(col) < low)
        if high is not None:
            expr_high = (pl.col(col) >= high) if include_high_eq else (pl.col(col) > high)
            expr = (expr | expr_high) if expr is not None else expr_high
        if expr is None:
            return 0
        return int(df.filter(expr).height)

    range_flags = {
        "price_out_of_range": count_outside("price", 0, None, include_low_eq=True),  # <=0
        "carat_out_of_range": count_outside("carat", 0, None, include_low_eq=True),  # <=0
        "x_nonpositive":      count_outside("x", 0, None, include_low_eq=True),      # <=0
        "y_nonpositive":      count_outside("y", 0, None, include_low_eq=True),      # <=0
        "z_nonpositive":      count_outside("z", 0, None, include_low_eq=True),      # <=0
        "depth_suspect":      count_outside("depth", 40, 80),                        # <40 o >80
        "table_suspect":      count_outside("table", 40, 100),                       # <40 o >100
    }
    checks["range_flags"] = range_flags

    return checks

# ==========================
# INGENIERÍA DE CARACTERÍSTICAS (AMPLIADA)
# ==========================
def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Aplicando ingeniería de características…")
    eps = 1e-12

    # Ordinales (compatibles)
    if "cut" in df.columns:
        df = df.with_columns(
            pl.col("cut").map_elements(lambda v: MAP_CUT_ORD.get(v), return_dtype=pl.Int64).alias("fe_cut_ord")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("fe_cut_ord"))

    if "color" in df.columns:
        df = df.with_columns(
            pl.col("color").map_elements(lambda v: MAP_COLOR_ORD.get(v), return_dtype=pl.Int64).alias("fe_color_ord")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("fe_color_ord"))

    if "clarity" in df.columns:
        df = df.with_columns(
            pl.col("clarity").map_elements(lambda v: MAP_CLARITY_ORD.get(v), return_dtype=pl.Int64).alias("fe_clarity_ord")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("fe_clarity_ord"))

    # Geometría y proporciones
    has_dims = all(c in df.columns for c in ["x","y","z"])
    if has_dims:
        df = df.with_columns([
            (pl.col("x") * pl.col("y") * pl.col("z")).alias("fe_volume_mm3"),
            (pl.col("x") * pl.col("y")).alias("fe_area_mm2"),
            ((pl.col("x") + pl.col("y"))/2.0).alias("fe_spread_mm"),
            (pl.col("x") / (pl.col("y") + eps)).alias("fe_aspect_ratio"),
            ((pl.col("x") <= 0) | (pl.col("y") <= 0) | (pl.col("z") <= 0)).alias("fe_invalid_dims"),
            (pl.when((pl.col("x").is_not_null()) & (pl.col("y").is_not_null()))
               .then(((pl.col("x") - pl.col("y")).abs() / ((pl.col("x") + pl.col("y"))/2.0 + eps)))
               .otherwise(None)).alias("fe_symmetry_dev_pct"),
            (pl.when((pl.col("x").is_not_null()) & (pl.col("y").is_not_null()) & (pl.col("z").is_not_null()))
               .then(100.0 * pl.col("z") / (((pl.col("x") + pl.col("y"))/2.0) + eps))
               .otherwise(None)).alias("fe_depth_pct_recalc"),
        ])

        # Consistencia de depth reportado vs recalculado
        if "depth" in df.columns:
            df = df.with_columns([
                (pl.col("fe_depth_pct_recalc") - pl.col("depth")).alias("fe_depth_pct_diff"),
                (pl.when(pl.col("fe_depth_pct_recalc").is_not_null() & pl.col("depth").is_not_null())
                   .then((pl.col("fe_depth_pct_recalc") - pl.col("depth")).abs() <= 0.5)
                   .otherwise(None)).alias("fe_depth_pct_is_consistent"),
            ])

        # Ratios verticales
        df = df.with_columns([
            (pl.col("z") / (pl.col("fe_spread_mm") + eps)).alias("fe_z_to_spread_ratio"),
        ])

    else:
        df = df.with_columns([
            pl.lit(None).alias("fe_volume_mm3"),
            pl.lit(None).alias("fe_area_mm2"),
            pl.lit(None).alias("fe_spread_mm"),
            pl.lit(None).alias("fe_aspect_ratio"),
            pl.lit(None).alias("fe_invalid_dims"),
            pl.lit(None).alias("fe_symmetry_dev_pct"),
            pl.lit(None).alias("fe_depth_pct_recalc"),
            pl.lit(None).alias("fe_depth_pct_diff"),
            pl.lit(None).alias("fe_depth_pct_is_consistent"),
            pl.lit(None).alias("fe_z_to_spread_ratio"),
        ])

    # Precio por carat y logs
    has_price_carat = all(c in df.columns for c in ["price","carat"])
    if has_price_carat:
        df = df.with_columns([
            (pl.col("price") / (pl.col("carat") + eps)).alias("fe_price_per_carat"),
            pl.when(pl.col("price") > 0).then(pl.col("price").cast(pl.Float64).log()).otherwise(None).alias("fe_log_price"),
            pl.when(pl.col("carat") > 0).then(pl.col("carat").cast(pl.Float64).log()).otherwise(None).alias("fe_log_carat"),
            pl.when((pl.col("price") > 0) & (pl.col("carat") > 0))
              .then((pl.col("price")/pl.col("carat")).cast(pl.Float64).log())
              .otherwise(None)
              .alias("fe_log_price_per_carat"),
        ])
    else:
        df = df.with_columns([
            pl.lit(None).alias("fe_price_per_carat"),
            pl.lit(None).alias("fe_log_price"),
            pl.lit(None).alias("fe_log_carat"),
            pl.lit(None).alias("fe_log_price_per_carat"),
        ])

    # Desviaciones respecto a valores típicos e índices adicionales
    if "depth" in df.columns:
        df = df.with_columns((pl.col("depth") - 61.5).alias("fe_depth_dev"))
    else:
        df = df.with_columns(pl.lit(None).alias("fe_depth_dev"))

    if "table" in df.columns:
        df = df.with_columns([
            (pl.col("table") - 57.0).alias("fe_table_dev"),
            (pl.col("table") / (pl.col("depth") + eps)).alias("fe_table_to_depth_ratio"),
        ])
    else:
        df = df.with_columns([
            pl.lit(None).alias("fe_table_dev"),
            pl.lit(None).alias("fe_table_to_depth_ratio"),
        ])

    # Métricas por carat (escala)
    if "carat" in df.columns:
        df = df.with_columns([
            (pl.col("fe_spread_mm") / (pl.col("carat") + eps)).alias("fe_spread_per_carat"),
            (pl.col("fe_area_mm2") / (pl.col("carat") + eps)).alias("fe_area_per_carat"),
        ])
    else:
        df = df.with_columns([
            pl.lit(None).alias("fe_spread_per_carat"),
            pl.lit(None).alias("fe_area_per_carat"),
        ])

    # Score simple de calidad (ponderaciones heurísticas)
    df = df.with_columns(
        (0.5 * pl.col("fe_cut_ord").fill_null(0)
         + 0.3 * pl.col("fe_color_ord").fill_null(0)
         + 0.2 * pl.col("fe_clarity_ord").fill_null(0)).alias("fe_quality_score")
    )

    # Bins de carat
    if "carat" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("carat") < 0.5).then(pl.lit("<0.5"))
             .when(pl.col("carat") < 1.0).then(pl.lit("0.5–1.0"))
             .when(pl.col("carat") < 1.5).then(pl.lit("1.0–1.5"))
             .when(pl.col("carat") < 2.0).then(pl.lit("1.5–2.0"))
             .otherwise(pl.lit("≥2.0"))
             .alias("fe_carat_bin")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("fe_carat_bin"))

    # Interacciones
    df = df.with_columns(
        (pl.col("carat").fill_null(0.0) * pl.col("fe_quality_score").fill_null(0.0)).alias("fe_carat_x_quality")
    )

    # Z-score de price_per_carat por grupo (cut, color, clarity)
    if all(c in df.columns for c in ["fe_price_per_carat","cut","color","clarity"]):
        group_cols = ["cut","color","clarity"]
        mu = pl.col("fe_price_per_carat").mean().over(group_cols)
        sd = pl.col("fe_price_per_carat").std().over(group_cols)
        df = df.with_columns([
            ((pl.col("fe_price_per_carat") - mu) / (sd + eps)).alias("fe_ppc_z_by_cqc")
        ])
    else:
        df = df.with_columns(pl.lit(None).alias("fe_ppc_z_by_cqc"))

    # Flag cuadrado aproximado (continuidad)
    if has_dims:
        df = df.with_columns(
            (pl.when((pl.col("x").is_not_null()) & (pl.col("y").is_not_null()))
               .then(((pl.col("x") - pl.col("y")).abs() <= 0.05))
               .otherwise(None)).alias("fe_is_square")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("fe_is_square"))

    return df

# ==========================
# ESTADÍSTICOS / MODA
# ==========================
def top_k_values(df: pl.DataFrame, col: str, k: int = 3):
    try:
        vc = (
            df.group_by(col)
              .agg(pl.len().alias("_count"))
              .sort("_count", descending=True)
              .head(k)
              .to_dict(as_series=False)
        )
        values = vc.get(col, [])
        counts = vc.get("_count", [])
        total = df.height if df.height else 1
        out = []
        for v, c in zip(values, counts):
            p = safe_div(c, total)
            out.append((v, c, p))
        return out
    except Exception:
        return []

def numeric_summary(df: pl.DataFrame, col: str) -> dict:
    try:
        s = df[col]
        clean = s.drop_nulls()
        if is_float_dtype(s.dtype):
            clean = clean.drop_nans()
        if clean.is_empty():
            return {}
        tmp = pl.DataFrame({col: clean})
        res = tmp.select([
            pl.col(col).min().alias("min"),
            pl.col(col).quantile(0.25, interpolation="nearest").alias("p25"),
            pl.col(col).median().alias("p50"),
            pl.col(col).quantile(0.75, interpolation="nearest").alias("p75"),
            pl.col(col).max().alias("max"),
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std")
        ]).to_dicts()[0]
        return {k: float(v) if v is not None else None for k, v in res.items()}
    except Exception:
        return {}

def print_numeric_stats(df: pl.DataFrame):
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c].dtype)]
    if not numeric_cols:
        return
    print_section("Estadísticos de Variables Numéricas")
    header = f"{'Columna':<16} {'min':>12} {'p25':>12} {'p50':>12} {'p75':>12} {'max':>12} {'mean':>12} {'std':>12}"
    emit(header)
    emit("-" * LINE_WIDTH)
    for c in numeric_cols:
        stats = numeric_summary(df, c)
        emit(f"{crop(c,16):<16} "
             f"{human_float(stats.get('min')):>12} {human_float(stats.get('p25')):>12} {human_float(stats.get('p50')):>12} "
             f"{human_float(stats.get('p75')):>12} {human_float(stats.get('max')):>12} {human_float(stats.get('mean')):>12} {human_float(stats.get('std')):>12}")
    emit("")

# ==========================
# SALIDAS ASCII EN TERMINAL
# ==========================
def print_section(title: str):
    emit(sep_line())
    emit(center_text(title))
    emit(sep_line())

def print_dictionary_table():
    dic = variable_dictionary()
    print_section("Diccionario de Variables")
    emit(f"{'Variable':<15} {'Tipo':<12} {'Descripción'}")
    emit("-" * LINE_WIDTH)
    for k,(t,desc) in dic.items():
        emit(f"{k:<15} {t:<12} {desc}")
    emit("")

def print_global_kpis(df: pl.DataFrame, checks: dict):
    total_rows = df.height
    total_cols = df.width
    duplicated = checks.get("duplicated_rows", 0)
    total_nulls = checks.get("total_nulls", 0)
    mem_approx = df.estimated_size()

    print_section("KPIs Globales del Dataset")
    emit(f"{'KPI':<25} {'Valor'}")
    emit("-" * LINE_WIDTH)
    emit(f"{'Registros':<25} {human_int(total_rows)}")
    emit(f"{'Columnas':<25} {human_int(total_cols)}")
    emit(f"{'Duplicados (filas)':<25} {human_int(duplicated)}")
    emit(f"{'Nulos totales':<25} {human_int(total_nulls)}")
    emit(f"{'Memoria aprox.':<25} {human_int(mem_approx)} bytes")
    emit("")

def print_schema_overview(df: pl.DataFrame, checks: dict):
    nulls = checks.get("nulls_by_col", {})
    nunique = checks.get("nunique_by_col", {})

    print_section("Resumen por Columna")
    header = f"{'Columna':<22} {'Tipo':<18} {'Nulos':>8} {'Únicos':>8} {'Top-1':<30} {'Top-2':<30} {'Top-3':<30}"
    emit(header)
    emit("-" * LINE_WIDTH)
    for c in df.columns:
        dtype = str(df[c].dtype)
        nnull = nulls.get(c, 0)
        nu = nunique.get(c, 0)
        top = top_k_values(df, c, k=3)

        def fmt_top_ascii(t):
            if not t:
                return "-"
            v, cnt, p = t
            val = "-" if v is None else crop(v, 18)
            return f"{val} ({pct(p)})"

        r1 = fmt_top_ascii(top[0]) if len(top) > 0 else "-"
        r2 = fmt_top_ascii(top[1]) if len(top) > 1 else "-"
        r3 = fmt_top_ascii(top[2]) if len(top) > 2 else "-"
        emit(f"{crop(c,22):<22} {crop(dtype,18):<18} {nnull:>8} {nu:>8} {crop(r1,30):<30} {crop(r2,30):<30} {crop(r3,30):<30}")
    emit("")

def print_quality_summary(checks: dict):
    dom = checks.get("domain_violations", {})
    rng = checks.get("range_flags", {})

    print_section("Calidad de Datos (Resumen)")
    emit(f"{'Chequeo':<35} {'Detalle/Conteo'}")
    emit("-" * LINE_WIDTH)

    # Paréntesis balanceados y sin duplicados
    emit(f"{'cut_invalid_values':<35} {(', '.join(dom.get('cut_invalid_values', [])) or 'OK')}")
    emit(f"{'color_invalid_values':<35} {(', '.join(dom.get('color_invalid_values', [])) or 'OK')}")
    emit(f"{'clarity_invalid_values':<35} {(', '.join(dom.get('clarity_invalid_values', [])) or 'OK')}")

    for k,v in rng.items():
        emit(f"{crop(k,35):<35} {human_int(v)}")
    emit("")

def print_outputs_summary(out_parquet_full: Path):
    print_section("Salidas Generadas")
    msg = textwrap.dedent(f"""
    - Parquet (base + features, SIN filtros): {out_parquet_full}
    - Log detallado:                           {LOG_FILE}
    - Schema/metadata (JSON):                  {SCHEMA_JSON}
    - Captura reporte terminal:                {REPORT_FILE}
    """).strip()
    emit(msg)
    emit("")

# ==========================
# GUARDAR PARQUET
# ==========================
def save_parquet(df: pl.DataFrame, out_dir: Path) -> Path:
    out_path = out_dir / f"diamonds_features_{RUN_TS}.parquet"
    logger.info(f"Escribiendo Parquet: {out_path}")
    df.write_parquet(out_path, compression="zstd", compression_level=3)
    logger.info("Parquet escrito con éxito.")
    return out_path

# ==========================
# MAIN
# ==========================
def main():
    emit(sep_line(100, "="))
    emit(center_text("INICIO ETL diamonds", 100))
    emit(sep_line(100, "="))

    try:
        # 1) Cargar
        df0 = load_csv_polars(PATH_IN_CSV)
        logger.info(f"Columnas iniciales: {df0.columns}")

        # 2) Tipificar/normalizar
        df = coerce_schema(df0)

        # 3) Chequeos calidad
        checks_before = data_quality_checks(df)

        # 4) Impresiones requeridas
        print_dictionary_table()
        print_global_kpis(df, checks_before)
        print_schema_overview(df, checks_before)
        print_numeric_stats(df)
        print_quality_summary(checks_before)

        # 5) Ingeniería de características (ampliada)
        df_fe = feature_engineering(df)

        # 6) Re-chequeo rápido en features (solo log informativo)
        if "fe_invalid_dims" in df_fe.columns:
            invalid_count = int(df_fe.select(pl.col("fe_invalid_dims").sum()).to_series(0).item())
            logger.info(f"fe_invalid_dims (True) = {invalid_count}")

        # 7) Guardar Parquet (FULL, sin filtros)
        out_parquet_full = save_parquet(df_fe, OUT_PARQUET_DIR)

        # 8) Guardar schema/metadata a JSON
        schema_info = {
            "run_ts": RUN_TS,
            "input_csv": str(PATH_IN_CSV),
            "output_parquet_full": str(out_parquet_full),
            "n_rows_full": df_fe.height,
            "n_cols_full": df_fe.width,
            "columns_full": [
                {"name": c, "dtype": str(df_fe[c].dtype)} for c in df_fe.columns
            ],
            "quality_checks": checks_before
        }
        with open(SCHEMA_JSON, "w", encoding="utf-8") as f:
            json.dump(schema_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Schema/metadata guardado en: {SCHEMA_JSON}")

        # 9) Guardar el reporte que se imprimió
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(REPORT.text())
        logger.info(f"Reporte de terminal guardado en: {REPORT_FILE}")

        # 10) Epílogo
        print_outputs_summary(out_parquet_full)
        logger.info("ETL finalizado con éxito.")

    except Exception as e:
        logger.error("Error en ETL", exc_info=True)
        err_msg = f"Ocurrió un error: {e}\n{traceback.format_exc()}"
        emit(sep_line())
        emit(center_text("ERROR"))
        emit(sep_line())
        emit(err_msg)
        try:
            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                f.write(REPORT.text())
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()

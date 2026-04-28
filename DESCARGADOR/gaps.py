"""
Detección y relleno de huecos temporales.

Reglas de relleno (por tipo de columna):
  - open/high/low/close (klines):          copiar close anterior a las 4 columnas del gap
  - volume/quote_volume/num_trades/taker*: rellenar con 0
  - mark/index/premium OHLC:              forward-fill

NO se usa interpolación. Solo forward-fill o cero.
"""
import polars as pl

from .utils import get_logger

log = get_logger(__name__)

_MICROS_POR_MINUTO = 60_000_000

_FLUJO_KLINES = ["volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]
_INT_KLINES   = ["num_trades"]
_OHLC_KLINES  = ["open", "high", "low", "close"]

_OHLC_MARK    = ["mark_open",    "mark_high",    "mark_low",    "mark_close"]
_OHLC_INDEX   = ["index_open",   "index_high",   "index_low",   "index_close"]
_OHLC_PREMIUM = ["premium_open", "premium_high", "premium_low", "premium_close"]


def rellenar_y_validar(df: pl.DataFrame) -> pl.DataFrame:
    """
    1. Genera el índice temporal completo (sin huecos) basado en klines.
    2. Left join del índice contra los datos reales.
    3. Rellena gaps según tipo de columna.
    4. Valida integridad del resultado.
    5. Retorna el DataFrame limpio.
    """
    df = df.sort("timestamp").unique(subset=["timestamp"], keep="first")

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()

    indice = pl.DataFrame({
        "timestamp": pl.datetime_range(
            ts_min, ts_max, interval="1m",
            time_unit="us", time_zone="UTC", eager=True,
        )
    })

    df = indice.join(df, on="timestamp", how="left")

    # ----------------------------------------------------------------
    # klines precio: marcar gaps ANTES de modificar close
    # ----------------------------------------------------------------
    df = df.with_columns(pl.col("close").is_null().alias("_gap_klines"))

    # close forward-fill
    df = df.with_columns(pl.col("close").forward_fill())

    # open/high/low en gaps = close (precio del último trade conocido)
    df = df.with_columns([
        pl.when(pl.col("_gap_klines")).then(pl.col("close")).otherwise(pl.col("open")).alias("open"),
        pl.when(pl.col("_gap_klines")).then(pl.col("close")).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("_gap_klines")).then(pl.col("close")).otherwise(pl.col("low")).alias("low"),
    ])

    # ----------------------------------------------------------------
    # klines flujos: cero (no hubo actividad)
    # ----------------------------------------------------------------
    df = df.with_columns([
        pl.col(c).fill_null(0.0) for c in _FLUJO_KLINES if c in df.columns
    ])
    df = df.with_columns([
        pl.col(c).fill_null(0) for c in _INT_KLINES if c in df.columns
    ])

    # ----------------------------------------------------------------
    # mark / index / premium: forward-fill
    # ----------------------------------------------------------------
    ffill_cols = [c for c in _OHLC_MARK + _OHLC_INDEX + _OHLC_PREMIUM if c in df.columns]
    if ffill_cols:
        df = df.with_columns([pl.col(c).forward_fill() for c in ffill_cols])

    df = df.drop("_gap_klines")

    _validar(df)
    return df


def _validar(df: pl.DataFrame) -> None:
    # 1. Sin nulos en ninguna columna
    for col in df.columns:
        n = df[col].null_count()
        if n > 0:
            raise ValueError(f"Columna '{col}' tiene {n} nulos tras el relleno de gaps")

    # 2. Timestamps ordenados y sin duplicados
    ts_int = df["timestamp"].cast(pl.Int64)
    if ts_int.is_duplicated().any():
        raise ValueError("Hay timestamps duplicados en el DataFrame final")

    diffs = ts_int.diff().drop_nulls()
    if (diffs != _MICROS_POR_MINUTO).any():
        bad_idx = (diffs != _MICROS_POR_MINUTO).arg_true()[0]
        bad_ts  = df["timestamp"][bad_idx + 1]
        raise ValueError(f"Diferencia de timestamp inesperada en {bad_ts}")

    # 3. Coherencia OHLC klines
    if all(c in df.columns for c in ("high", "low", "open", "close")):
        if (df["high"] < df["low"]).any():
            raise ValueError("Datos inválidos: high < low en klines")
        if (df["open"] > df["high"]).any() or (df["open"] < df["low"]).any():
            raise ValueError("Datos inválidos: open fuera del rango [low, high]")
        if (df["close"] > df["high"]).any() or (df["close"] < df["low"]).any():
            raise ValueError("Datos inválidos: close fuera del rango [low, high]")

    # 4. Volúmenes no negativos
    for col in _FLUJO_KLINES:
        if col in df.columns and (df[col] < 0).any():
            raise ValueError(f"Volumen negativo en columna '{col}'")

    log.info(f"  Validación OK: {len(df):,} filas, sin nulos, timestamps perfectos")

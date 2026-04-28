"""
Exporta el DataFrame final a Parquet con compresión snappy.
Usa escritura atómica (archivo .tmp + rename) para garantizar
que nunca quede un Parquet parcial en HISTORICO/.
"""
import shutil
from pathlib import Path

import polars as pl

from .utils import get_logger

log = get_logger(__name__)

# Orden canónico de columnas en el Parquet final
ORDEN_COLUMNAS = [
    "timestamp",
    # klines
    "open", "high", "low", "close",
    "volume", "quote_volume", "num_trades",
    "taker_buy_volume", "taker_buy_quote_volume",
    "taker_sell_volume", "vol_delta",
    # markPriceKlines
    "mark_open", "mark_high", "mark_low", "mark_close",
    # indexPriceKlines
    "index_open", "index_high", "index_low", "index_close",
    # premiumIndexKlines
    "premium_open", "premium_high", "premium_low", "premium_close",
]


def guardar(df: pl.DataFrame, destino: Path) -> None:
    """
    Guarda df en Parquet snappy en destino.
    Escribe primero a .tmp.parquet y renombra al final para garantizar atomicidad.
    Si falla, el .tmp es eliminado y el destino anterior no se toca.
    """
    # Columnas derivadas de microestructura (calculadas aquí, con gaps ya rellenos)
    df = df.with_columns([
        (pl.col("volume") - pl.col("taker_buy_volume")).alias("taker_sell_volume"),
        (pl.col("taker_buy_volume") - (pl.col("volume") - pl.col("taker_buy_volume"))).alias("vol_delta"),
    ])

    _validar_microestructura(df)

    cols_faltantes = [c for c in ORDEN_COLUMNAS if c not in df.columns]
    if cols_faltantes:
        raise ValueError(f"Columnas esperadas no encontradas en el DataFrame: {cols_faltantes}")

    df = df.select(ORDEN_COLUMNAS)

    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp.parquet")

    try:
        df.write_parquet(tmp, compression="snappy")
        shutil.move(str(tmp), str(destino))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tam_mb = destino.stat().st_size / 1_048_576
    log.info(f"Parquet guardado: {destino.name}  ({tam_mb:.1f} MB, {len(df):,} filas)")


def _validar_microestructura(df: pl.DataFrame) -> None:
    # taker_sell_volume >= 0 (si es negativo, taker_buy_volume > volume en origen)
    mask_neg = df["taker_sell_volume"] < 0
    if mask_neg.any():
        bad_ts = df.filter(mask_neg)["timestamp"][0]
        raise ValueError(
            f"taker_sell_volume negativo en {bad_ts}: "
            f"taker_buy_volume supera a volume en los datos de origen"
        )

    # |vol_delta| <= volume (si lo supera, igualmente indica corrupción en origen)
    mask_delta = df["vol_delta"].abs() > df["volume"]
    if mask_delta.any():
        bad_ts = df.filter(mask_delta)["timestamp"][0]
        raise ValueError(
            f"|vol_delta| > volume en {bad_ts}: "
            f"los datos de taker_buy_volume son inconsistentes con volume"
        )

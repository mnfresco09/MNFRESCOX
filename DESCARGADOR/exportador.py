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
    # markPriceKlines
    "mark_open", "mark_high", "mark_low", "mark_close",
    # indexPriceKlines
    "index_open", "index_high", "index_low", "index_close",
    # premiumIndexKlines
    "premium_open", "premium_high", "premium_low", "premium_close",
    # metrics
    "open_interest", "open_interest_value",
    "long_short_ratio", "long_account_ratio", "short_account_ratio",
    "top_trader_long_short_ratio", "top_trader_long_account_ratio",
    "top_trader_short_account_ratio",
]


def guardar(df: pl.DataFrame, destino: Path) -> None:
    """
    Guarda df en Parquet snappy en destino.
    Escribe primero a .tmp.parquet y renombra al final para garantizar atomicidad.
    Si falla, el .tmp es eliminado y el destino anterior no se toca.
    """
    cols_presentes = [c for c in ORDEN_COLUMNAS if c in df.columns]
    df = df.select(cols_presentes)

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

"""
Une los cinco tipos de datos en un único DataFrame usando klines como columna vertebral.
Aplica prefijos a las columnas OHLC de mark, index y premium para evitar colisiones.
"""
import polars as pl

from .utils import get_logger

log = get_logger(__name__)


def combinar(
    klines:   pl.DataFrame,
    mark:     pl.DataFrame,
    index:    pl.DataFrame,
    premium:  pl.DataFrame,
    metrics:  pl.DataFrame,
) -> pl.DataFrame:
    """
    Left join de todos los tipos contra los timestamps de klines.
    Los períodos sin datos en un tipo secundario quedan como null
    (serán rellenados por gaps.py).
    """
    mark    = _prefixar_ohlc(mark,    "mark")
    index   = _prefixar_ohlc(index,   "index")
    premium = _prefixar_ohlc(premium, "premium")

    df = klines
    for otro in (mark, index, premium, metrics):
        df = df.join(otro, on="timestamp", how="left")

    return df


def _prefixar_ohlc(df: pl.DataFrame, prefijo: str) -> pl.DataFrame:
    return df.rename({
        "open":  f"{prefijo}_open",
        "high":  f"{prefijo}_high",
        "low":   f"{prefijo}_low",
        "close": f"{prefijo}_close",
    })

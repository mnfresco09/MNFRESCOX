"""
Une klines y premiumIndexKlines en un único DataFrame usando klines como columna vertebral.
Aplica prefijo a las columnas OHLC de premium para evitar colisiones.
"""
import polars as pl


def combinar(
    klines:   pl.DataFrame,
    premium:  pl.DataFrame,
) -> pl.DataFrame:
    """
    Left join de premium contra los timestamps de klines.
    Los períodos sin datos en premium quedan como null (serán rellenados por gaps.py).
    """
    premium = premium.rename({
        "open":  "premium_open",
        "high":  "premium_high",
        "low":   "premium_low",
        "close": "premium_close",
    })

    return klines.join(premium, on="timestamp", how="left")

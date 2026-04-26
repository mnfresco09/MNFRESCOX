import polars as pl

# Timeframe base que siempre se almacena en HISTORICO
TF_BASE = "1m"

# Jerarquía de menor a mayor para validar la dirección del resampleo
_JERARQUIA = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]

# Mapeo a la cadena de duración que entiende Polars group_by_dynamic
_DURACION = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# Columnas OHLCV estándar que se agregan con reglas fijas
_COLS_OHLCV = {"open", "high", "low", "close", "volume"}


def resamplear(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Construye velas del timeframe pedido a partir de velas de 1m.
    Solo permite ir hacia timeframes más grandes, nunca más pequeños.
    Las columnas extra (OI, funding rate, etc.) se agregan como media.
    """
    if timeframe not in _JERARQUIA:
        raise ValueError(
            f"Timeframe '{timeframe}' no reconocido. Opciones: {_JERARQUIA}"
        )

    if timeframe == TF_BASE:
        return df

    idx_base   = _JERARQUIA.index(TF_BASE)
    idx_pedido = _JERARQUIA.index(timeframe)
    if idx_pedido < idx_base:
        raise ValueError(
            f"No se puede resamplear de '{TF_BASE}' a '{timeframe}': "
            f"solo se puede ir hacia timeframes más grandes."
        )

    duracion = _DURACION[timeframe]
    columnas_presentes = set(df.columns)

    # Reglas de agregación para columnas OHLCV estándar
    aggs = [
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
    ]

    if "volume" in columnas_presentes:
        aggs.append(pl.col("volume").sum())

    # Columnas extra: media (conserva información sin inventar valores)
    extras = columnas_presentes - _COLS_OHLCV - {"timestamp"}
    for col in sorted(extras):
        aggs.append(pl.col(col).mean())

    df_resampled = (
        df.sort("timestamp")
        .group_by_dynamic("timestamp", every=duracion, closed="left")
        .agg(aggs)
        .sort("timestamp")
    )

    return df_resampled

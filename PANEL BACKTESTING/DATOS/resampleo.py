import polars as pl

# Jerarquía de menor a mayor para validar la dirección del resampleo
_JERARQUIA = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
TIMEFRAMES_ORDENADOS = tuple(_JERARQUIA)

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
_SEGUNDOS_A_TF = {
    60: "1m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    14400: "4h",
    86400: "1d",
}

# Columnas OHLCV estándar que se agregan con reglas fijas
_COLS_OHLCV = {"open", "high", "low", "close", "volume"}


def resamplear(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Construye velas del timeframe pedido a partir del timeframe mas bajo disponible.
    Solo permite ir hacia timeframes más grandes, nunca más pequeños.
    Las columnas extra (OI, funding rate, etc.) se agregan como media.
    """
    if timeframe not in _JERARQUIA:
        raise ValueError(
            f"Timeframe '{timeframe}' no reconocido. Opciones: {_JERARQUIA}"
        )

    timeframe_base = inferir_timeframe(df)

    if timeframe == timeframe_base:
        return df

    idx_base   = _JERARQUIA.index(timeframe_base)
    idx_pedido = _JERARQUIA.index(timeframe)
    if idx_pedido < idx_base:
        raise ValueError(
            f"No se puede resamplear de '{timeframe_base}' a '{timeframe}': "
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

    df_ordenado = df.sort("timestamp")
    ultimo_timestamp = df_ordenado["timestamp"][-1]

    # Ventanas estrictamente cerradas y etiquetadas a la derecha:
    # una vela 10:00-11:00 queda marcada en 11:00, no en 10:00.
    df_resampled = (
        df_ordenado
        .group_by_dynamic(
            "timestamp",
            every=duracion,
            closed="right",
            label="right",
            start_by="window",
        )
        .agg(aggs)
        .filter(pl.col("timestamp") <= ultimo_timestamp)
        .sort("timestamp")
    )

    return df_resampled


def inferir_timeframe(df: pl.DataFrame) -> str:
    if df.height < 2:
        raise ValueError("No se puede inferir timeframe con menos de 2 filas.")

    timestamps = (
        df.sort("timestamp")
        .select("timestamp")
        .head(min(df.height, 1_000))
        .to_series()
        .to_list()
    )
    diffs = []
    for previo, actual in zip(timestamps, timestamps[1:]):
        delta = actual - previo
        segundos = int(delta.total_seconds())
        if segundos > 0:
            diffs.append(segundos)

    if not diffs:
        raise ValueError("No se pudo inferir timeframe: timestamps sin avance temporal.")

    segundos_base = min(diffs)
    if segundos_base not in _SEGUNDOS_A_TF:
        raise ValueError(
            "Timeframe base no soportado por el sistema: "
            f"delta_minimo={segundos_base} segundos."
        )
    return _SEGUNDOS_A_TF[segundos_base]

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
_TF_A_SEGUNDOS = {tf: segundos for segundos, tf in _SEGUNDOS_A_TF.items()}

# Regla de agregación por columna. Si se añade una columna nueva al histórico,
# debe declararse aquí para evitar resampleos con semántica incorrecta.
_REGLAS_AGREGACION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_volume": "sum",
    "num_trades": "sum",
    "taker_buy_volume": "sum",
    "taker_buy_quote_volume": "sum",
    "taker_sell_volume": "sum",
    "vol_delta": "sum",
    "premium_close": "last",
    "predicted_funding_rate": "last",
    "open_interest": "last",
    "funding_rate": "last",
}


def resamplear(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Construye velas del timeframe pedido a partir del timeframe mas bajo disponible.
    Solo permite ir hacia timeframes más grandes, nunca más pequeños.
    Cada columna se agrega con una regla explícita según lo que mide.
    """
    if timeframe not in _JERARQUIA:
        raise ValueError(
            f"Timeframe '{timeframe}' no reconocido. Opciones: {_JERARQUIA}"
        )

    df_ordenado = _asegurar_orden_timestamp(df)
    timeframe_base = inferir_timeframe(df_ordenado)

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
    filas_esperadas = _filas_esperadas_por_ventana(timeframe_base, timeframe)
    aggs = _construir_agregaciones(df.columns)

    # Ventanas [inicio, fin) sin lookahead. El timestamp final no es la apertura
    # de la ventana, sino la última vela base incluida: 00:00..00:14 -> 00:14.
    # Así la señal queda disponible en 00:14 y el motor entra en N+1 (00:15).
    df_resampled = (
        df_ordenado
        .group_by_dynamic(
            "timestamp",
            every=duracion,
            closed="left",
            label="left",
            start_by="window",
        )
        .agg([
            pl.col("timestamp").last().alias("_timestamp_operativo"),
            pl.len().alias("_filas_ventana"),
            *aggs,
        ])
        .filter(pl.col("_filas_ventana") == filas_esperadas)
        .with_columns(pl.col("_timestamp_operativo").alias("timestamp"))
        .drop(["_timestamp_operativo", "_filas_ventana"])
    )

    return df_resampled


def _filas_esperadas_por_ventana(timeframe_base: str, timeframe: str) -> int:
    segundos_base = _TF_A_SEGUNDOS[timeframe_base]
    segundos_destino = _TF_A_SEGUNDOS[timeframe]
    if segundos_destino % segundos_base != 0:
        raise ValueError(
            f"No se puede resamplear de '{timeframe_base}' a '{timeframe}': "
            "la duración destino no es múltiplo exacto de la base."
        )
    return segundos_destino // segundos_base


def _construir_agregaciones(columnas: list[str]) -> list[pl.Expr]:
    columnas_datos = [col for col in columnas if col != "timestamp"]
    desconocidas = sorted(col for col in columnas_datos if col not in _REGLAS_AGREGACION)
    if desconocidas:
        raise ValueError(
            "Columnas sin regla de resampleo declarada: "
            f"{desconocidas}. Añade su semántica a _REGLAS_AGREGACION."
        )

    return [_expresion_agregacion(col, _REGLAS_AGREGACION[col]) for col in columnas_datos]


def _expresion_agregacion(columna: str, regla: str) -> pl.Expr:
    if regla == "first":
        return pl.col(columna).first()
    if regla == "max":
        return pl.col(columna).max()
    if regla == "min":
        return pl.col(columna).min()
    if regla == "last":
        return pl.col(columna).last()
    if regla == "sum":
        return pl.col(columna).sum()

    raise ValueError(f"Regla de resampleo no soportada para '{columna}': {regla}")


def _asegurar_orden_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    if df["timestamp"].is_sorted():
        return df.set_sorted("timestamp")
    return df.sort("timestamp").set_sorted("timestamp")


def inferir_timeframe(df: pl.DataFrame) -> str:
    if df.height < 2:
        raise ValueError("No se puede inferir timeframe con menos de 2 filas.")

    timestamps = (
        df.select("timestamp")
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

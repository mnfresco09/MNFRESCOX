import polars as pl

from DATOS.tiempo import (
    DURACION_POLARS,
    SEGUNDOS_POR_TIMEFRAME,
    TIMEFRAMES_ORDENADOS,
    inferir_timeframe,
    segundos_timeframe,
)

# Reexportados para compatibilidad: parte del codigo importa historicamente
# estos nombres desde `DATOS.resampleo`. La fuente de verdad ahora es
# `DATOS.tiempo`.
__all__ = [
    "TIMEFRAMES_ORDENADOS",
    "segundos_timeframe",
    "inferir_timeframe",
    "regla_agregacion",
    "resamplear",
]

# Regla de agregacion por columna. Si se anade una columna nueva al historico,
# debe declararse aqui para evitar resampleos con semantica incorrecta.
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


def regla_agregacion(columna: str) -> str:
    """Devuelve la regla declarada para una columna resampleable."""
    try:
        return _REGLAS_AGREGACION[columna]
    except KeyError as exc:
        raise ValueError(
            "Columnas sin regla de resampleo declarada: "
            f"['{columna}']. Anade su semantica a _REGLAS_AGREGACION."
        ) from exc


def resamplear(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Construye velas del timeframe pedido a partir del timeframe mas bajo disponible.
    Solo permite ir hacia timeframes mas grandes, nunca mas pequenos.
    Cada columna se agrega con una regla explicita segun lo que mide.
    """
    if timeframe not in TIMEFRAMES_ORDENADOS:
        raise ValueError(
            f"Timeframe '{timeframe}' no reconocido. Opciones: {list(TIMEFRAMES_ORDENADOS)}"
        )

    df_ordenado = _asegurar_orden_timestamp(df)
    timeframe_base = inferir_timeframe(df_ordenado)

    if timeframe == timeframe_base:
        return df

    idx_base = TIMEFRAMES_ORDENADOS.index(timeframe_base)
    idx_pedido = TIMEFRAMES_ORDENADOS.index(timeframe)
    if idx_pedido < idx_base:
        raise ValueError(
            f"No se puede resamplear de '{timeframe_base}' a '{timeframe}': "
            f"solo se puede ir hacia timeframes mas grandes."
        )

    duracion = DURACION_POLARS[timeframe]
    filas_esperadas = _filas_esperadas_por_ventana(timeframe_base, timeframe)
    aggs = _construir_agregaciones(df.columns)

    # Ventanas [inicio, fin) sin lookahead. El timestamp visible es la apertura
    # natural de la vela: 00:00..00:14 -> 00:00. La proyeccion al timeframe base
    # calcula aparte el cierre operativo para que el motor siga entrando despues
    # de que la vela resampleada este confirmada.
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
            pl.len().alias("_filas_ventana"),
            *aggs,
        ])
        .filter(pl.col("_filas_ventana") == filas_esperadas)
        .drop("_filas_ventana")
    )

    return df_resampled


def _filas_esperadas_por_ventana(timeframe_base: str, timeframe: str) -> int:
    segundos_base = SEGUNDOS_POR_TIMEFRAME[timeframe_base]
    segundos_destino = SEGUNDOS_POR_TIMEFRAME[timeframe]
    if segundos_destino % segundos_base != 0:
        raise ValueError(
            f"No se puede resamplear de '{timeframe_base}' a '{timeframe}': "
            "la duracion destino no es multiplo exacto de la base."
        )
    return segundos_destino // segundos_base


def _construir_agregaciones(columnas: list[str]) -> list[pl.Expr]:
    columnas_datos = [col for col in columnas if col != "timestamp"]
    desconocidas = sorted(col for col in columnas_datos if col not in _REGLAS_AGREGACION)
    if desconocidas:
        raise ValueError(
            "Columnas sin regla de resampleo declarada: "
            f"{desconocidas}. Anade su semantica a _REGLAS_AGREGACION."
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

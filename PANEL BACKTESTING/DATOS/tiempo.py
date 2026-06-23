"""Constantes y utilidades temporales compartidas: unica fuente de verdad.

Aqui viven la jerarquia de timeframes, sus conversiones (segundos /
microsegundos / cadena de duracion de Polars) y los helpers para obtener el
timestamp en microsegundos epoch UTC desde un DataFrame.

Lo usan el cargador, el resampleo, la proyeccion, los validadores y la
verificacion de integridad. Antes cada uno tenia su propia copia de estos
diccionarios y funciones; centralizarlos evita que la semantica temporal se
desincronice al anadir o cambiar un timeframe.
"""

from __future__ import annotations

import polars as pl

# Jerarquia de menor a mayor. El resampleo solo puede ir hacia timeframes
# mas grandes (indice mayor).
TIMEFRAMES_ORDENADOS: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")

SEGUNDOS_POR_TIMEFRAME: dict[str, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}
TIMEFRAME_POR_SEGUNDOS: dict[int, str] = {
    segundos: tf for tf, segundos in SEGUNDOS_POR_TIMEFRAME.items()
}
MICROSEGUNDOS_POR_TIMEFRAME: dict[str, int] = {
    tf: segundos * 1_000_000 for tf, segundos in SEGUNDOS_POR_TIMEFRAME.items()
}
# Cadena de duracion que entiende `group_by_dynamic` de Polars.
DURACION_POLARS: dict[str, str] = {tf: tf for tf in SEGUNDOS_POR_TIMEFRAME}


def segundos_timeframe(timeframe: str) -> int:
    """Segundos que dura una vela del timeframe dado."""
    try:
        return int(SEGUNDOS_POR_TIMEFRAME[timeframe])
    except KeyError as exc:
        raise ValueError(
            f"Timeframe '{timeframe}' no reconocido. Opciones: {list(TIMEFRAMES_ORDENADOS)}"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers de timestamp en microsegundos epoch
# ---------------------------------------------------------------------------

def expr_timestamp_us(df: pl.DataFrame) -> pl.Expr:
    """Expresion Polars que devuelve `timestamp` como Int64 en microsegundos.

    Acepta tanto columnas `Datetime` (cualquier zona/precision) como enteras ya
    en microsegundos.
    """
    dtype = df.schema.get("timestamp")
    if dtype is None:
        raise ValueError("El DataFrame no contiene columna 'timestamp'.")
    if isinstance(dtype, pl.Datetime):
        return pl.col("timestamp").dt.epoch("us")
    return pl.col("timestamp").cast(pl.Int64)


def serie_timestamp_us(df: pl.DataFrame) -> pl.Series:
    """Serie Int64 con el timestamp en microsegundos epoch."""
    return df.select(expr_timestamp_us(df).alias("_ts_us")).to_series()


def intervalo_us(df: pl.DataFrame) -> int:
    """Menor delta positivo entre timestamps consecutivos, en microsegundos."""
    if df.height < 2:
        raise ValueError("Se necesitan al menos 2 velas para inferir el intervalo.")
    diffs = serie_timestamp_us(df).diff().drop_nulls()
    diffs = diffs.filter(diffs > 0)
    if diffs.is_empty():
        raise ValueError("No se pudo inferir el intervalo temporal.")
    return int(diffs.min())


def inferir_timeframe(df: pl.DataFrame) -> str:
    """Infiere el timeframe base por el menor delta de las primeras filas."""
    if df.height < 2:
        raise ValueError("No se puede inferir timeframe con menos de 2 filas.")

    timestamps = (
        df.select(expr_timestamp_us(df).alias("_ts_us"))
        .head(min(df.height, 1_000))
        .to_series()
        .to_list()
    )
    diffs = [
        actual - previo
        for previo, actual in zip(timestamps, timestamps[1:])
        if actual - previo > 0
    ]
    if not diffs:
        raise ValueError("No se pudo inferir timeframe: timestamps sin avance temporal.")

    segundos_base = min(diffs) // 1_000_000
    if segundos_base not in TIMEFRAME_POR_SEGUNDOS:
        raise ValueError(
            "Timeframe base no soportado por el sistema: "
            f"delta_minimo={segundos_base} segundos."
        )
    return TIMEFRAME_POR_SEGUNDOS[segundos_base]

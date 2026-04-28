"""Mapeo y proyección de señales entre el timeframe de estrategia y el base."""

from __future__ import annotations

import numpy as np
import polars as pl

from DATOS.resampleo import segundos_timeframe


def construir_mapeo(
    df_base: pl.DataFrame,
    df_tf: pl.DataFrame,
    timeframe: str | None = None,
) -> np.ndarray:
    """Devuelve un array int64 con la posición en `df_base` que corresponde
    al cierre operativo de cada vela de `df_tf` (`join_asof` backward).

    Las velas resampleadas se etiquetan por apertura para que los graficos
    muestren 00:00, 00:15, 00:30, etc. Para ejecutar sin lookahead, la senal
    de esa vela se proyecta a la ultima vela base incluida en la ventana.
    """
    base_intervalo_us = _intervalo_us(df_base)
    tf_intervalo_us = (
        segundos_timeframe(timeframe) * 1_000_000
        if timeframe is not None
        else _intervalo_us(df_tf)
    )
    offset_cierre_us = max(0, int(tf_intervalo_us) - int(base_intervalo_us))

    base_con_idx = (
        df_base.with_row_index("base_idx")
        .select([_timestamp_us_expr(df_base).alias("_ts_us"), "base_idx"])
        .sort("_ts_us")
    )
    tf_operativo = df_tf.select(
        (_timestamp_us_expr(df_tf) + offset_cierre_us).alias("_ts_us")
    )
    mapeo_serie = (
        tf_operativo
        .sort("_ts_us")
        .join_asof(base_con_idx, on="_ts_us", strategy="backward")
        .get_column("base_idx")
    )

    if mapeo_serie.null_count() > 0:
        nulos = (
            mapeo_serie.to_frame("base_idx")
            .with_row_index("pos")
            .filter(pl.col("base_idx").is_null())
            .get_column("pos")
            .head(10)
            .to_list()
        )
        muestras = ", ".join(str(int(p)) for p in nulos)
        raise ValueError(
            "[PROYECCION] Hay timestamps del timeframe estrategia anteriores al "
            f"inicio del timeframe base. Indices: {muestras}."
        )

    arr = mapeo_serie.cast(pl.Int64).to_numpy()
    if arr.dtype != np.int64 or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr


def proyectar_senales_a_base(
    senales_tf,
    tf_to_base_idx: np.ndarray,
    base_len: int,
) -> pl.Series:
    """Proyecta señales del TF de estrategia al TF base.

    El mapeo marca el cierre operativo de cada vela de estrategia. El motor
    entra siempre en la siguiente vela base, manteniendo la regla N+1.
    """
    if len(senales_tf) != tf_to_base_idx.shape[0]:
        raise ValueError(
            "[PROYECCION] Longitud de senales y mapeo no coincide: "
            f"{len(senales_tf):,} != {tf_to_base_idx.shape[0]:,}."
        )

    # Acepta pl.Series o np.ndarray
    if isinstance(senales_tf, np.ndarray):
        valores_tf = senales_tf if senales_tf.dtype == np.int8 else senales_tf.astype(np.int8)
    else:
        valores_tf = senales_tf.cast(pl.Int8).to_numpy()

    arr = np.zeros(int(base_len), dtype=np.int8)
    if tf_to_base_idx.shape[0] == 0:
        return pl.Series("senal", arr)

    validos = (tf_to_base_idx >= 0) & (tf_to_base_idx < int(base_len))
    arr[tf_to_base_idx[validos]] = valores_tf[validos]
    return pl.Series("senal", arr)


def _timestamp_us_expr(df: pl.DataFrame) -> pl.Expr:
    dtype = df.schema.get("timestamp")
    if dtype is None:
        raise ValueError("[PROYECCION] Falta columna timestamp.")
    if isinstance(dtype, pl.Datetime):
        return pl.col("timestamp").dt.epoch("us")
    return pl.col("timestamp").cast(pl.Int64)


def _intervalo_us(df: pl.DataFrame) -> int:
    if df.height < 2:
        raise ValueError("[PROYECCION] Se necesitan al menos 2 velas para inferir intervalo.")

    ts = df.select(_timestamp_us_expr(df).alias("_ts_us")).get_column("_ts_us")
    diffs = ts.diff().drop_nulls()
    diffs = diffs.filter(diffs > 0)
    if diffs.is_empty():
        raise ValueError("[PROYECCION] No se pudo inferir intervalo temporal.")
    return int(diffs.min())

"""Mapeo y proyección de señales entre el timeframe de estrategia y el base."""

from __future__ import annotations

import numpy as np
import polars as pl


def construir_mapeo(df_base: pl.DataFrame, df_tf: pl.DataFrame) -> np.ndarray:
    """Devuelve un array int64 con la posición en `df_base` que corresponde
    al cierre de cada vela de `df_tf` (`join_asof` backward).
    """
    base_con_idx = (
        df_base.with_row_index("base_idx")
        .select(["timestamp", "base_idx"])
        .sort("timestamp")
    )
    mapeo_serie = (
        df_tf.select("timestamp")
        .sort("timestamp")
        .join_asof(base_con_idx, on="timestamp", strategy="backward")
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
    """Proyecta señales del TF de estrategia al TF base. Las velas resampleadas
    están etiquetadas en la última vela base incluida en la ventana, así que la
    señal se marca ahí; el motor entra siempre en la siguiente vela base.
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

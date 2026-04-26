from __future__ import annotations

import numpy as np
import polars as pl


def construir_mapeo(df_base: pl.DataFrame, df_tf: pl.DataFrame) -> list[int]:
    base_con_idx = (
        df_base.with_row_index("base_idx")
        .select(["timestamp", "base_idx"])
        .sort("timestamp")
    )
    # join_asof backward: cada vela del TF se mapea a la última vela base
    # cuyo timestamp sea <= al timestamp del TF (necesario en mercados con huecos).
    mapeo = (
        df_tf.select("timestamp")
        .sort("timestamp")
        .join_asof(base_con_idx, on="timestamp", strategy="backward")
        .get_column("base_idx")
        .to_list()
    )

    faltantes = [idx for idx, valor in enumerate(mapeo) if valor is None]
    if faltantes:
        muestras = ", ".join(str(i) for i in faltantes[:10])
        raise ValueError(
            "[PROYECCION] Hay timestamps del timeframe estrategia anteriores al "
            f"inicio del timeframe base. Indices: {muestras}."
        )

    return [int(valor) for valor in mapeo]


def proyectar_senales_a_base(
    senales_tf: pl.Series,
    tf_to_base_idx: list[int],
    base_len: int,
) -> pl.Series:
    if len(senales_tf) != len(tf_to_base_idx):
        raise ValueError(
            "[PROYECCION] Longitud de senales y mapeo no coincide: "
            f"{len(senales_tf):,} != {len(tf_to_base_idx):,}."
        )

    arr = np.zeros(int(base_len), dtype=np.int8)
    if len(tf_to_base_idx) < 2:
        return pl.Series("senal", arr)

    valores_tf = senales_tf.cast(pl.Int8).to_numpy()
    targets = np.asarray(tf_to_base_idx[1:], dtype=np.int64) - 1
    validos = (targets >= 0) & (targets < int(base_len))
    arr[targets[validos]] = valores_tf[:-1][validos]
    return pl.Series("senal", arr)

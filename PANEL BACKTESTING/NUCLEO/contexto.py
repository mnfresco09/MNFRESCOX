"""Contexto de combinación: buffers numpy estables compartidos entre todos
los trials de Optuna sobre un mismo (activo, timeframe).

Diseño de memoria
-----------------
- Cada columna OHLCV se materializa **una sola vez** como `np.ndarray`
  contiguo (zero-copy desde Polars cuando es Float64). Estos buffers se
  pasan al motor Rust sin copia adicional gracias al crate `numpy`.
- `ArraysMotor` agrupa los buffers de un único timeframe. No duplica datos.
- No existe `CacheDF`: la verificación de integridad usa los mismos buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from COMUN.numpy_utiles import a_contiguo
from DATOS.tiempo import serie_timestamp_us
from NUCLEO.proyeccion import construir_mapeo


@dataclass(frozen=True)
class ArraysMotor:
    """Buffers numpy contiguos del DataFrame, listos para el motor Rust."""

    timestamps: np.ndarray  # int64 (microsegundos epoch UTC)
    opens: np.ndarray       # float64
    highs: np.ndarray       # float64
    lows: np.ndarray        # float64
    closes: np.ndarray      # float64
    volumes: np.ndarray     # float64
    salidas_neutras: np.ndarray  # int8, todo ceros (reusable como default)

    def __len__(self) -> int:
        return int(self.opens.shape[0])


@dataclass(frozen=True)
class SimConfigMotor:
    saldo_inicial: float
    saldo_por_trade: float
    apalancamiento: float
    saldo_minimo: float
    comision_pct: float
    comision_lados: int
    exit_type: str
    exit_sl_pct: float
    exit_tp_pct: float
    exit_velas: int
    exit_trail_act_pct: float = 0.0
    exit_trail_dist_pct: float = 0.0


@dataclass(frozen=True)
class ContextoCombinacion:
    df_tf: pl.DataFrame
    df_base: pl.DataFrame
    arrays_tf: ArraysMotor
    arrays_base: ArraysMotor
    tf_to_base_idx: np.ndarray  # int64
    es_min_tf: bool


def crear_contexto(
    df_base: pl.DataFrame,
    df_tf: pl.DataFrame,
    timeframe: str | None = None,
) -> ContextoCombinacion:
    arrays_base = construir_arrays_motor(df_base)
    if _mismo_dataframe_temporal(df_base, df_tf):
        arrays_tf = arrays_base
        tf_to_base_idx = np.arange(df_base.height, dtype=np.int64)
        es_min_tf = True
    else:
        arrays_tf = construir_arrays_motor(df_tf)
        tf_to_base_idx = construir_mapeo(df_base, df_tf, timeframe=timeframe)
        es_min_tf = False

    return ContextoCombinacion(
        df_tf=df_tf,
        df_base=df_base,
        arrays_tf=arrays_tf,
        arrays_base=arrays_base,
        tf_to_base_idx=tf_to_base_idx,
        es_min_tf=es_min_tf,
    )


def construir_arrays_motor(df: pl.DataFrame) -> ArraysMotor:
    n = df.height
    return ArraysMotor(
        timestamps=_timestamps_us(df),
        opens=_columna_f64(df, "open"),
        highs=_columna_f64(df, "high"),
        lows=_columna_f64(df, "low"),
        closes=_columna_f64(df, "close"),
        volumes=_volume_array(df),
        salidas_neutras=np.zeros(n, dtype=np.int8),
    )


def _columna_f64(df: pl.DataFrame, nombre: str) -> np.ndarray:
    return a_contiguo(df[nombre].cast(pl.Float64).to_numpy(), np.float64)


def _mismo_dataframe_temporal(df_base: pl.DataFrame, df_tf: pl.DataFrame) -> bool:
    return df_base.height == df_tf.height and df_base["timestamp"].equals(df_tf["timestamp"])


def _timestamps_us(df: pl.DataFrame) -> np.ndarray:
    return a_contiguo(serie_timestamp_us(df).to_numpy(), np.int64)


def _volume_array(df: pl.DataFrame) -> np.ndarray:
    if "volume" not in df.columns:
        return np.zeros(df.height, dtype=np.float64)
    return a_contiguo(df["volume"].cast(pl.Float64).fill_null(0.0).to_numpy(), np.float64)

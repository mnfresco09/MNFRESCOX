from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from NUCLEO.proyeccion import construir_mapeo


@dataclass(frozen=True)
class ArraysMotor:
    """Columnas OHLCV ya convertidas al contrato plano que espera Rust."""

    timestamps: list[int]
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]
    salidas_neutras: list[int]

    def __len__(self) -> int:
        return len(self.opens)


@dataclass(frozen=True)
class CacheDF:
    timestamps: list[int]
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]


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


@dataclass(frozen=True)
class ContextoCombinacion:
    df_tf: pl.DataFrame
    df_base: pl.DataFrame
    cache_tf: CacheDF
    cache_base: CacheDF
    arrays_tf: ArraysMotor
    arrays_base: ArraysMotor
    tf_to_base_idx: list[int]
    es_min_tf: bool


def crear_contexto(df_base: pl.DataFrame, df_tf: pl.DataFrame) -> ContextoCombinacion:
    arrays_base = construir_arrays_motor(df_base)
    if _mismo_dataframe_temporal(df_base, df_tf):
        arrays_tf = arrays_base
        cache_tf = cache_desde_arrays(arrays_base)
        cache_base = cache_tf
        tf_to_base_idx = list(range(df_base.height))
        es_min_tf = True
    else:
        arrays_tf = construir_arrays_motor(df_tf)
        cache_tf = cache_desde_arrays(arrays_tf)
        cache_base = cache_desde_arrays(arrays_base)
        tf_to_base_idx = construir_mapeo(df_base, df_tf)
        es_min_tf = False

    return ContextoCombinacion(
        df_tf=df_tf,
        df_base=df_base,
        cache_tf=cache_tf,
        cache_base=cache_base,
        arrays_tf=arrays_tf,
        arrays_base=arrays_base,
        tf_to_base_idx=tf_to_base_idx,
        es_min_tf=es_min_tf,
    )


def construir_arrays_motor(df: pl.DataFrame) -> ArraysMotor:
    return ArraysMotor(
        timestamps=_timestamps_us(df),
        opens=df["open"].cast(pl.Float64).to_list(),
        highs=df["high"].cast(pl.Float64).to_list(),
        lows=df["low"].cast(pl.Float64).to_list(),
        closes=df["close"].cast(pl.Float64).to_list(),
        volumes=_volume_list(df),
        salidas_neutras=[0] * df.height,
    )


def cache_desde_arrays(arrays: ArraysMotor) -> CacheDF:
    return CacheDF(
        timestamps=arrays.timestamps,
        opens=arrays.opens,
        highs=arrays.highs,
        lows=arrays.lows,
        closes=arrays.closes,
    )


def cachear_df(df: pl.DataFrame) -> CacheDF:
    return CacheDF(
        timestamps=_timestamps_us(df),
        opens=df["open"].cast(pl.Float64).to_list(),
        highs=df["high"].cast(pl.Float64).to_list(),
        lows=df["low"].cast(pl.Float64).to_list(),
        closes=df["close"].cast(pl.Float64).to_list(),
    )


def _mismo_dataframe_temporal(df_base: pl.DataFrame, df_tf: pl.DataFrame) -> bool:
    return df_base.height == df_tf.height and df_base["timestamp"].equals(df_tf["timestamp"])


def _timestamps_us(df: pl.DataFrame) -> list[int]:
    dtype = df.schema.get("timestamp")
    if dtype is None:
        raise ValueError("El DataFrame no contiene columna 'timestamp'.")

    if isinstance(dtype, pl.Datetime):
        return df.select(pl.col("timestamp").dt.epoch("us")).to_series().to_list()

    if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        return df["timestamp"].cast(pl.Int64).to_list()

    raise ValueError(f"timestamp debe ser Datetime o entero en microsegundos, no {dtype}.")


def _volume_list(df: pl.DataFrame) -> list[float]:
    if "volume" not in df.columns:
        return [0.0] * df.height
    return df["volume"].cast(pl.Float64).fill_null(0.0).to_list()

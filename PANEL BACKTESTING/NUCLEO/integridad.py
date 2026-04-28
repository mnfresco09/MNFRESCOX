"""Verificaciones de integridad sobre datos, señales y resultados.

`verificar_resultado` opera sobre el dict columnar devuelto por
`SimResult.take_trades()` (replay). No se invoca en el camino caliente de
Optuna: el motor Rust acumula métricas internamente y queda auditado por
sus propios tests; en cada `_optimizar_combinacion` se replican sólo los
trials que generan reportes y sobre esos sí se hace la verificación
exhaustiva.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import numpy as np
import polars as pl

from MOTOR.wrapper import MOTIVOS

TOL = 1e-7
_TIMEFRAME_US = {
    "1m": 60_000_000,
    "5m": 5 * 60_000_000,
    "15m": 15 * 60_000_000,
    "30m": 30 * 60_000_000,
    "1h": 60 * 60_000_000,
    "4h": 4 * 60 * 60_000_000,
    "1d": 24 * 60 * 60_000_000,
}


@dataclass(frozen=True)
class HuellaDatos:
    etapa: str
    filas: int
    columnas: tuple[str, ...]
    ts_inicio: object
    ts_fin: object


def huella_dataframe(etapa: str, df: pl.DataFrame) -> HuellaDatos:
    if df.is_empty():
        raise ValueError(f"[INTEGRIDAD] {etapa}: DataFrame vacio.")
    if "timestamp" not in df.columns:
        raise ValueError(f"[INTEGRIDAD] {etapa}: falta columna timestamp.")
    _verificar_timestamps(etapa, df)
    _verificar_ohlc(etapa, df)
    return HuellaDatos(
        etapa=etapa,
        filas=df.height,
        columnas=tuple(df.columns),
        ts_inicio=df["timestamp"][0],
        ts_fin=df["timestamp"][-1],
    )


def verificar_resampleo(origen: pl.DataFrame, destino: pl.DataFrame, timeframe: str) -> None:
    huella_dataframe(f"resampleo {timeframe}", destino)
    if timeframe == "1m":
        if destino.height != origen.height:
            raise ValueError("[INTEGRIDAD] Resampleo 1m no debe cambiar el numero de filas.")
        if not destino["timestamp"].equals(origen["timestamp"]):
            raise ValueError("[INTEGRIDAD] Resampleo 1m altero timestamps.")
        return

    if "volume" in origen.columns and "volume" in destino.columns:
        origen_ts_us = _timestamp_us(origen)
        destino_ts_us = _timestamp_us(destino)
        fin_cubierto_us = int(destino_ts_us[-1])
        if timeframe in _TIMEFRAME_US and timeframe != "1m":
            fin_cubierto_us += int(_TIMEFRAME_US[timeframe]) - _intervalo_us(origen)
        origen_cubierto = origen.filter(origen_ts_us <= fin_cubierto_us)
        vol_origen = float(origen_cubierto["volume"].sum())
        vol_destino = float(destino["volume"].sum())
        if not isclose(vol_origen, vol_destino, rel_tol=TOL, abs_tol=TOL):
            raise ValueError(
                "[INTEGRIDAD] Volumen no conservado tras resampleo: "
                f"origen={vol_origen}, destino={vol_destino}."
            )

    origen_ts_us = _timestamp_us(origen)
    destino_ts_us = _timestamp_us(destino)
    if int(destino_ts_us[0]) < int(origen_ts_us[0]):
        raise ValueError("[INTEGRIDAD] Resampleo creo timestamp inicial anterior al origen.")
    if int(destino_ts_us[-1]) > int(origen_ts_us[-1]):
        raise ValueError("[INTEGRIDAD] Resampleo creo timestamp final posterior al origen.")
    if timeframe in _TIMEFRAME_US and timeframe != "1m":
        fin_operativo_us = int(destino_ts_us[-1]) + int(_TIMEFRAME_US[timeframe]) - _intervalo_us(origen)
        if fin_operativo_us > int(origen_ts_us[-1]):
            raise ValueError("[INTEGRIDAD] Resampleo termina despues del origen disponible.")


def verificar_senales(df: pl.DataFrame, senales) -> dict[int, int]:
    """Acepta pl.Series o np.ndarray (int8)."""
    if isinstance(senales, np.ndarray):
        if senales.shape[0] != df.height:
            raise ValueError(
                f"[INTEGRIDAD] Longitud de senales incorrecta: {senales.shape[0]:,} != {df.height:,}."
            )
        arr = senales if senales.dtype == np.int8 else senales.astype(np.int8)
        invalidos = np.unique(arr)
        invalidos = invalidos[(invalidos != -1) & (invalidos != 0) & (invalidos != 1)]
        if invalidos.size:
            raise ValueError(f"[INTEGRIDAD] Senales invalidas: {sorted(int(v) for v in invalidos)}.")
        return {
            -1: int((arr == -1).sum()),
             0: int((arr == 0).sum()),
             1: int((arr == 1).sum()),
        }

    if len(senales) != df.height:
        raise ValueError(
            f"[INTEGRIDAD] Longitud de senales incorrecta: {len(senales):,} != {df.height:,}."
        )
    if senales.null_count() > 0:
        raise ValueError("[INTEGRIDAD] Las senales contienen nulos.")
    valores = set(senales.cast(pl.Int8).unique().to_list())
    invalidos = valores - {-1, 0, 1}
    if invalidos:
        raise ValueError(f"[INTEGRIDAD] Senales invalidas: {sorted(invalidos)}.")
    return {
        -1: int((senales == -1).sum()),
         0: int((senales == 0).sum()),
         1: int((senales == 1).sum()),
    }


def verificar_salidas_custom(df: pl.DataFrame, salidas) -> dict[int, int]:
    return verificar_senales(df, salidas)


def verificar_resultado(arrays, senales, trades: dict, equity_curve: np.ndarray, metricas) -> None:
    """Verificación exhaustiva del replay (sólo top-N trials).

    Recibe los buffers numpy del contexto, las señales utilizadas y el dict
    columnar devuelto por `SimResult.take_trades()`. Toda la verificación es
    vectorial: para 10 K trades dura microsegundos, no segundos.
    """
    n_trades = int(trades["idx_senal"].shape[0])
    if equity_curve.shape[0] != n_trades + 1:
        raise ValueError("[INTEGRIDAD] equity_curve debe tener saldo inicial + un punto por trade.")

    pnl_total = float(trades["pnl"].sum()) if n_trades else 0.0
    saldo_esperado = float(metricas.saldo_inicial) + pnl_total
    if not isclose(float(metricas.saldo_final), saldo_esperado, rel_tol=TOL, abs_tol=TOL):
        raise ValueError("[INTEGRIDAD] saldo_final no coincide con saldo_inicial + pnl_total.")

    if not isclose(float(equity_curve[0]), float(metricas.saldo_inicial), abs_tol=TOL):
        raise ValueError("[INTEGRIDAD] equity_curve no arranca en saldo_inicial.")
    if not isclose(float(equity_curve[-1]), float(metricas.saldo_final), abs_tol=TOL):
        raise ValueError("[INTEGRIDAD] equity_curve no termina en saldo_final.")

    if n_trades == 0:
        return

    timestamps = arrays.timestamps
    opens = arrays.opens
    highs = arrays.highs
    lows = arrays.lows
    closes = arrays.closes
    total = timestamps.shape[0]

    idx_senal = trades["idx_senal"].astype(np.int64)
    idx_entrada = trades["idx_entrada"].astype(np.int64)
    idx_salida = trades["idx_salida"].astype(np.int64)
    direccion = trades["direccion"].astype(np.int8)
    motivos = trades["motivo_salida"].astype(np.int8)

    if (idx_senal < 0).any() or (idx_salida >= total).any():
        raise ValueError("[INTEGRIDAD] Indices de trade fuera de rango.")
    if not ((idx_senal < idx_entrada) & (idx_entrada <= idx_salida)).all():
        raise ValueError("[INTEGRIDAD] Indices de trade fuera de orden.")
    if not (idx_entrada == idx_senal + 1).all():
        raise ValueError("[INTEGRIDAD] Trade con entrada que no ocurre en vela N+1.")

    # Señales en idx_senal deben coincidir con la dirección registrada
    senales_arr = senales.to_numpy() if hasattr(senales, "to_numpy") else senales
    if senales_arr.dtype != np.int8:
        senales_arr = senales_arr.astype(np.int8)
    if not (senales_arr[idx_senal] == direccion).all():
        raise ValueError("[INTEGRIDAD] Trade con direccion que no coincide con la senal.")

    if not (timestamps[idx_senal] == trades["ts_senal"]).all():
        raise ValueError("[INTEGRIDAD] Trade con timestamp de senal incorrecto.")
    if not (timestamps[idx_entrada] == trades["ts_entrada"]).all():
        raise ValueError("[INTEGRIDAD] Trade con timestamp de entrada incorrecto.")
    if not (timestamps[idx_salida] == trades["ts_salida"]).all():
        raise ValueError("[INTEGRIDAD] Trade con timestamp de salida incorrecto.")

    # precio_entrada == open[idx_entrada]
    if not np.allclose(trades["precio_entrada"], opens[idx_entrada], rtol=TOL, atol=TOL):
        raise ValueError("[INTEGRIDAD] Trade con precio_entrada que no es el open de N+1.")

    precio_salida = trades["precio_salida"]
    cod_sl, cod_tp, cod_bars, cod_custom, cod_end = 0, 1, 2, 3, 4

    # Salidas END/BARS -> close de la vela. CUSTOM se confirma en N y ejecuta
    # en el open de N+1 para evitar cierre en el mismo close usado por la señal.
    mask_close = (motivos == cod_end) | (motivos == cod_bars)
    if mask_close.any():
        if not np.allclose(precio_salida[mask_close], closes[idx_salida[mask_close]], rtol=TOL, atol=TOL):
            raise ValueError("[INTEGRIDAD] Salida END/BARS no usa close.")

    mask_custom = motivos == cod_custom
    if mask_custom.any():
        if not np.allclose(precio_salida[mask_custom], opens[idx_salida[mask_custom]], rtol=TOL, atol=TOL):
            raise ValueError("[INTEGRIDAD] Salida CUSTOM no usa open de ejecucion.")

    # Salidas SL/TP → dentro del rango low..high
    mask_intra = (motivos == cod_sl) | (motivos == cod_tp)
    if mask_intra.any():
        ps = precio_salida[mask_intra]
        lo = lows[idx_salida[mask_intra]]
        hi = highs[idx_salida[mask_intra]]
        if (ps < lo - TOL).any() or (ps > hi + TOL).any():
            raise ValueError("[INTEGRIDAD] Salida SL/TP fuera del rango de vela.")

    motivos_validos = (
        (motivos == cod_sl)
        | (motivos == cod_tp)
        | (motivos == cod_bars)
        | (motivos == cod_custom)
        | (motivos == cod_end)
    )
    if not motivos_validos.all():
        raise ValueError(f"[INTEGRIDAD] Trade con motivo_salida desconocido (códigos: {MOTIVOS}).")


def _verificar_timestamps(etapa: str, df: pl.DataFrame) -> None:
    if not df["timestamp"].is_sorted():
        raise ValueError(f"[INTEGRIDAD] {etapa}: timestamps no ordenados.")
    duplicados = df.height - df["timestamp"].n_unique()
    if duplicados:
        raise ValueError(f"[INTEGRIDAD] {etapa}: {duplicados:,} timestamps duplicados.")


def _verificar_ohlc(etapa: str, df: pl.DataFrame) -> None:
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"[INTEGRIDAD] {etapa}: falta columna {col}.")
        if df[col].null_count() > 0:
            raise ValueError(f"[INTEGRIDAD] {etapa}: columna {col} contiene nulos.")

    invalidas = df.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
    ).height
    if invalidas:
        raise ValueError(f"[INTEGRIDAD] {etapa}: {invalidas:,} velas OHLC incoherentes.")


def _timestamp_us(df: pl.DataFrame) -> pl.Series:
    dtype = df.schema.get("timestamp")
    if isinstance(dtype, pl.Datetime):
        return df.select(pl.col("timestamp").dt.epoch("us")).to_series()
    return df["timestamp"].cast(pl.Int64)


def _intervalo_us(df: pl.DataFrame) -> int:
    if df.height < 2:
        raise ValueError("[INTEGRIDAD] No se puede inferir intervalo con menos de 2 filas.")
    diffs = _timestamp_us(df).diff().drop_nulls()
    diffs = diffs.filter(diffs > 0)
    if diffs.is_empty():
        raise ValueError("[INTEGRIDAD] No se pudo inferir intervalo temporal.")
    return int(diffs.min())

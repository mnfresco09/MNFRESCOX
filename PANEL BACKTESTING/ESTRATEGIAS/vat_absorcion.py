"""VWAP Absorption Trend.

Estrategia tendencial basada en una referencia EWM-VWAP y una medida causal de
absorcion de order flow. La senal nace cuando el sesgo de precio contra VWAP y
la presion neta agresiva apuntan en la misma direccion.
"""

from __future__ import annotations

import math
from threading import Lock

import numpy as np
import polars as pl

from NUCLEO.base_estrategia import BaseEstrategia

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback para entornos sin numba
    njit = None


def _jit_cache(func):
    if njit is None:
        return func
    return njit(cache=True)(func)


_JIT_WARMUP_LOCK = Lock()
_JIT_PRECALENTADO = False


class VwapAbsorptionTrend(BaseEstrategia):
    ID = 3
    NOMBRE = "VWAP Absorption Trend"
    COLUMNAS_REQUERIDAS = {"taker_buy_volume", "taker_sell_volume", "vol_delta", "volume"}

    def parametros_por_defecto(self) -> dict:
        return {"hl_vwap": 100, "hl_cvd": 20, "umbral": 0.8}

    def espacio_busqueda(self, trial) -> dict:
        return {
            "hl_vwap": trial.suggest_int("hl_vwap", 6, 300, step=2),
            "hl_cvd": trial.suggest_int("hl_cvd", 6, 300, step=2),
            "umbral": trial.suggest_float("umbral", 0.3, 2.0, step=0.05),
        }

    def bind(self, arrays, cache=None) -> None:
        super().bind(arrays, cache)
        _precalentar_vat_jit()

    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        hl_vwap, hl_cvd, umbral = _normalizar_params(params)
        vwap, vat_z = self._vat(df, hl_vwap=hl_vwap, hl_cvd=hl_cvd)
        warmup = _warmup(hl_vwap, hl_cvd)

        finitos = np.isfinite(self.close) & np.isfinite(vwap) & np.isfinite(vat_z)
        sobre_vwap = self.close > vwap
        long_mask = finitos & sobre_vwap & (vat_z > umbral)
        short_mask = finitos & ~sobre_vwap & (vat_z < -umbral)
        _bloquear_warmup(long_mask, warmup)
        _bloquear_warmup(short_mask, warmup)
        return self.serie_senales(df.height, long_mask, short_mask)

    def generar_salidas(self, df: pl.DataFrame, params: dict) -> pl.Series:
        hl_vwap, hl_cvd, _ = _normalizar_params(params)
        _, vat_z = self._vat(df, hl_vwap=hl_vwap, hl_cvd=hl_cvd)
        warmup = _warmup(hl_vwap, hl_cvd)

        finitos = np.isfinite(vat_z)
        long_exit = finitos & (vat_z < 0.0)
        short_exit = finitos & (vat_z > 0.0)
        _bloquear_warmup(long_exit, warmup)
        _bloquear_warmup(short_exit, warmup)
        return self.serie_senales(df.height, long_exit, short_exit)

    def indicadores_para_grafica(self, df: pl.DataFrame, params: dict) -> list[dict]:
        hl_vwap, hl_cvd, umbral = _normalizar_params(params)
        vwap, vat_z = self._vat(df, hl_vwap=hl_vwap, hl_cvd=hl_cvd)
        return [
            _serie_overlay(df, vwap, "#f59e0b", f"VAT VWAP({hl_vwap})"),
            _serie_pane(
                df,
                vat_z,
                "#22c55e",
                f"VAT Z({hl_cvd})",
                niveles=[
                    {"valor": umbral, "color": "#22c55e66"},
                    {"valor": 0.0, "color": "#64748b88"},
                    {"valor": -umbral, "color": "#ef444466"},
                ],
            ),
        ]

    def _vat(self, df: pl.DataFrame, *, hl_vwap: int, hl_cvd: int) -> tuple[np.ndarray, np.ndarray]:
        volume = self.volume
        vol_delta = _columna_float(df, "vol_delta")
        return self.memo(
            "vat",
            id(self.close),
            int(hl_vwap),
            int(hl_cvd),
            calcular=lambda: _calcular_vat(self.close, volume, vol_delta, int(hl_vwap), int(hl_cvd)),
        )


def _normalizar_params(params: dict) -> tuple[int, int, float]:
    hl_vwap = max(1, int(params.get("hl_vwap", 100)))
    hl_cvd = max(1, int(params.get("hl_cvd", 20)))
    umbral = max(0.0, float(params.get("umbral", 0.8)))
    return hl_vwap, hl_cvd, umbral


def _warmup(hl_vwap: int, hl_cvd: int) -> int:
    return max(int(hl_cvd) * 9, int(hl_vwap) * 3)


def _bloquear_warmup(mask: np.ndarray, warmup: int) -> None:
    if warmup > 0:
        mask[: min(int(warmup), mask.shape[0])] = False


def _precalentar_vat_jit() -> None:
    """Compila VAT antes de que Optuna ejecute trials en paralelo.

    Los arrays zero-copy que llegan desde Polars suelen ser read-only. Numba
    genera una firma distinta para arrays escribibles y no escribibles, por eso
    se precalientan ambas rutas y se evita que varios workers compilen a la vez.
    """
    if njit is None:
        return

    global _JIT_PRECALENTADO
    if _JIT_PRECALENTADO:
        return

    with _JIT_WARMUP_LOCK:
        if _JIT_PRECALENTADO:
            return
        close, volume, vol_delta = _arrays_warmup(writeable=True)
        _calcular_vat(close, volume, vol_delta, 3, 2)
        close_ro, volume_ro, vol_delta_ro = _arrays_warmup(writeable=False)
        _calcular_vat(close_ro, volume_ro, vol_delta_ro, 3, 2)
        _JIT_PRECALENTADO = True


def _arrays_warmup(*, writeable: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = np.array([100.0, 100.4, 100.1, 100.9, 101.2, 100.8], dtype=np.float64)
    volume = np.array([10.0, 12.0, 9.0, 14.0, 11.0, 13.0], dtype=np.float64)
    vol_delta = np.array([0.0, 3.0, -2.0, 5.0, 2.5, -1.0], dtype=np.float64)
    if not writeable:
        close.setflags(write=False)
        volume.setflags(write=False)
        vol_delta.setflags(write=False)
    return close, volume, vol_delta


@_jit_cache
def _calcular_vat(
    close: np.ndarray,
    volume: np.ndarray,
    vol_delta: np.ndarray,
    hl_vwap: int,
    hl_cvd: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(close.shape[0])
    vwap = np.empty(n, dtype=np.float64)
    vat_z = np.zeros(n, dtype=np.float64)
    if n == 0:
        return vwap, vat_z

    alpha_vwap = _alpha_halflife(hl_vwap)
    alpha_norm = _alpha_halflife(hl_cvd * 3)

    pv_ewm = _safe_price(close[0]) * max(float(volume[0]), 0.0)
    v_ewm = max(float(volume[0]), 0.0)
    cvd_media = 0.0
    cvd_media_sq = 0.0
    ret_media = 0.0
    ret_media_sq = 0.0

    for idx in range(n):
        precio = _safe_price(close[idx])
        volumen = max(float(volume[idx]), 0.0)
        pv_actual = precio * volumen

        pv_ewm = alpha_vwap * pv_actual + (1.0 - alpha_vwap) * pv_ewm
        v_ewm = alpha_vwap * volumen + (1.0 - alpha_vwap) * v_ewm
        vwap[idx] = pv_ewm / v_ewm if v_ewm > 1e-12 else precio

        cvd = _cvd_relativo(vol_delta[idx], volumen)
        retorno = _retorno(close, idx)

        cvd_media = alpha_norm * cvd + (1.0 - alpha_norm) * cvd_media
        cvd_media_sq = alpha_norm * cvd * cvd + (1.0 - alpha_norm) * cvd_media_sq
        ret_media = alpha_norm * retorno + (1.0 - alpha_norm) * ret_media
        ret_media_sq = alpha_norm * retorno * retorno + (1.0 - alpha_norm) * ret_media_sq

        cvd_std = math.sqrt(max(0.0, cvd_media_sq - cvd_media * cvd_media))
        ret_std = math.sqrt(max(0.0, ret_media_sq - ret_media * ret_media))
        cvd_z = (cvd - cvd_media) / cvd_std if cvd_std > 1e-12 else 0.0
        ret_z = (retorno - ret_media) / ret_std if ret_std > 1e-12 else 0.0

        acuerdo = math.tanh(ret_z * cvd_z)
        vat_z[idx] = cvd_z * (0.5 + 0.5 * acuerdo)

    return vwap, vat_z


@_jit_cache
def _alpha_halflife(halflife: int) -> float:
    return 1.0 - math.exp(-math.log(2.0) / max(1.0, float(halflife)))


@_jit_cache
def _safe_price(value: float) -> float:
    precio = float(value)
    return precio if math.isfinite(precio) and precio > 0.0 else 1e-12


@_jit_cache
def _cvd_relativo(delta: float, volumen: float) -> float:
    if volumen <= 1e-12:
        return 0.0
    valor = float(delta) / volumen
    if not math.isfinite(valor):
        return 0.0
    return min(1.0, max(-1.0, valor))


@_jit_cache
def _retorno(close: np.ndarray, idx: int) -> float:
    if idx <= 0:
        return 0.0
    previo = _safe_price(close[idx - 1])
    actual = _safe_price(close[idx])
    return (actual - previo) / previo


def _columna_float(df: pl.DataFrame, nombre: str) -> np.ndarray:
    serie = df[nombre].cast(pl.Float64).fill_null(0.0)
    arr = serie.to_numpy()
    if arr.dtype != np.float64 or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def _serie_overlay(df: pl.DataFrame, valores: np.ndarray, color: str, nombre: str) -> dict:
    ts_seg, vals = _puntos_finitos(df, valores, decimales=6)
    return {
        "nombre": nombre,
        "tipo": "overlay",
        "color": color,
        "data": [{"t": int(t), "v": float(v)} for t, v in zip(ts_seg, vals)],
    }


def _serie_pane(
    df: pl.DataFrame,
    valores: np.ndarray,
    color: str,
    nombre: str,
    *,
    niveles: list[dict],
) -> dict:
    ts_seg, vals = _puntos_finitos(df, valores, decimales=4)
    return {
        "nombre": nombre,
        "tipo": "pane",
        "color": color,
        "data": [{"t": int(t), "v": float(v)} for t, v in zip(ts_seg, vals)],
        "niveles": niveles,
    }


def _puntos_finitos(df: pl.DataFrame, valores: np.ndarray, *, decimales: int) -> tuple[np.ndarray, np.ndarray]:
    timestamps_us = _timestamps_us(df)
    finitos = np.isfinite(valores)
    ts_seg = (timestamps_us[finitos] // 1_000_000).astype(np.int64)
    vals = np.round(valores[finitos], int(decimales))
    return ts_seg, vals


def _timestamps_us(df: pl.DataFrame) -> np.ndarray:
    dtype = df.schema.get("timestamp")
    if isinstance(dtype, pl.Datetime):
        serie = df.select(pl.col("timestamp").dt.epoch("us")).to_series()
    else:
        serie = df["timestamp"].cast(pl.Int64)
    arr = serie.to_numpy()
    if arr.dtype != np.int64 or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr

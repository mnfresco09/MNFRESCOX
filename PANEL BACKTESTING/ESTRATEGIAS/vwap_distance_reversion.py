"""VWAP Distance Reversion.

Estrategia de reversion basada solo en la distancia normalizada del precio
respecto a una EWM-VWAP. No usa CVD ni importa helpers de VWAP-CVD.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import ClassVar

import numpy as np
import polars as pl

from NUCLEO.base_estrategia import BaseEstrategia

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback para entornos sin numba
    njit = None


DEFAULT_HALFLIFE_BARS = 15
DEFAULT_NORMALIZATION_MULTIPLIER = 3.0
DEFAULT_CLIP_SIGMAS = 2.5
DEFAULT_UMBRAL_DISTANCE_Z = 0.5
VOLUME_EPSILON = 1e-7

IDX_VWAP = 0
IDX_DISTANCE_RAW = 1
IDX_DISTANCE_SIGNAL = 2
IDX_DISTANCE_Z = 3


def _jit_cache(func):
    if njit is None:
        return func
    return njit(cache=True)(func)


_JIT_WARMUP_LOCK = Lock()
_JIT_PRECALENTADO = False


class VWAPDistanceReversion(BaseEstrategia):
    ID = 5
    NOMBRE = "VWAP Distance Reversion"
    COLUMNAS_REQUERIDAS: ClassVar[set[str]] = {"volume"}

    def parametros_por_defecto(self) -> dict:
        return {
            "halflife_bars": DEFAULT_HALFLIFE_BARS,
            "normalization_multiplier": DEFAULT_NORMALIZATION_MULTIPLIER,
            "vwap_clip_sigmas": DEFAULT_CLIP_SIGMAS,
            "umbral_distance_z": DEFAULT_UMBRAL_DISTANCE_Z,
        }

    def espacio_busqueda(self, trial) -> dict:
        return {
            "halflife_bars": trial.suggest_int("halflife_bars", 35, 65, step=1),
            "normalization_multiplier": trial.suggest_float("normalization_multiplier", 1.1, 2.5, step=0.1),
            "vwap_clip_sigmas": trial.suggest_float("vwap_clip_sigmas", 2.0, 3.5, step=0.1),
            "umbral_distance_z": trial.suggest_float("umbral_distance_z", 0.30, 1.00, step=0.1),
        }

    def bind(self, arrays, cache=None) -> None:
        super().bind(arrays, cache)
        _precalentar_vwap_distance_jit()

    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        halflife, norm_mult, clip_sigmas, umbral = _normalizar_params(params)
        valores = self._indicadores(halflife, norm_mult, clip_sigmas)
        vwap = valores[IDX_VWAP]
        distance_z = valores[IDX_DISTANCE_Z]
        finitos = np.isfinite(self.close) & np.isfinite(vwap) & np.isfinite(distance_z)
        senales = _generar_senales_reversion(distance_z, finitos, umbral, _warmup(halflife, norm_mult))
        return pl.Series("senal", senales)

    def indicadores_para_grafica(self, df: pl.DataFrame, params: dict) -> list[dict]:
        halflife, norm_mult, clip_sigmas, umbral = _normalizar_params(params)
        valores = self._indicadores(halflife, norm_mult, clip_sigmas)
        return [
            _serie_overlay(df, valores[IDX_VWAP], "#00bcd4", f"VWAP Distance VWAP({halflife})"),
            _serie_pane(
                df,
                valores[IDX_DISTANCE_Z],
                "#ab47bc",
                "VWAP DIST Z",
                niveles=[
                    {"valor": umbral, "color": "#22c55e66"},
                    {"valor": 0.0, "color": "#64748b88"},
                    {"valor": -umbral, "color": "#ef444466"},
                ],
            ),
        ]

    def _indicadores(
        self,
        halflife: int,
        normalization_multiplier: float,
        clip_sigmas: float,
    ) -> tuple[np.ndarray, ...]:
        return self.memo(
            "vwap_distance_reversion",
            id(self.close),
            id(self.volume),
            int(halflife),
            float(normalization_multiplier),
            float(clip_sigmas),
            calcular=lambda: _calcular_vwap_distance(
                self.close,
                self.volume,
                int(halflife),
                float(normalization_multiplier),
                float(clip_sigmas),
            ),
        )


def _normalizar_params(params: dict) -> tuple[int, float, float, float]:
    halflife = max(1, int(params.get("halflife_bars", DEFAULT_HALFLIFE_BARS)))
    norm_mult = max(0.1, float(params.get("normalization_multiplier", DEFAULT_NORMALIZATION_MULTIPLIER)))
    clip_sigmas = max(0.1, float(params.get("vwap_clip_sigmas", DEFAULT_CLIP_SIGMAS)))
    umbral = max(1e-12, float(params.get("umbral_distance_z", DEFAULT_UMBRAL_DISTANCE_Z)))
    return halflife, norm_mult, clip_sigmas, umbral


def _warmup(halflife: int, normalization_multiplier: float) -> int:
    normalization_halflife = max(1.0, float(halflife) * float(normalization_multiplier))
    return max(int(halflife) * 9, round(normalization_halflife * 3.0))


def _precalentar_vwap_distance_jit() -> None:
    if njit is None:
        return

    global _JIT_PRECALENTADO
    if _JIT_PRECALENTADO:
        return

    with _JIT_WARMUP_LOCK:
        if _JIT_PRECALENTADO:
            return
        close, volume = _arrays_warmup(writeable=True)
        _calcular_vwap_distance(close, volume, 5, 3.0, 2.5)
        finitos = np.ones(close.shape[0], dtype=np.bool_)
        _generar_senales_reversion(np.array([0.0, 0.6, 0.4, -0.6, -0.4, 0.0]), finitos, 0.5, 0)
        close_ro, volume_ro = _arrays_warmup(writeable=False)
        _calcular_vwap_distance(close_ro, volume_ro, 5, 3.0, 2.5)
        _JIT_PRECALENTADO = True


def _arrays_warmup(*, writeable: bool) -> tuple[np.ndarray, np.ndarray]:
    close = np.array([100.0, 100.8, 100.2, 101.4, 100.7, 99.9], dtype=np.float64)
    volume = np.array([10.0, 12.0, 9.0, 14.0, 11.0, 13.0], dtype=np.float64)
    if not writeable:
        close.setflags(write=False)
        volume.setflags(write=False)
    return close, volume


@_jit_cache
def _generar_senales_reversion(
    distance_z: np.ndarray,
    finitos: np.ndarray,
    umbral: float,
    warmup: int,
) -> np.ndarray:
    n = int(distance_z.shape[0])
    senales = np.zeros(n, dtype=np.int8)
    upper = abs(float(umbral))
    if upper <= 0.0:
        return senales

    lower = -upper
    armado_short = False
    armado_long = False

    for idx in range(1, n):
        if idx < int(warmup):
            armado_short = False
            armado_long = False
            continue
        if not finitos[idx] or not finitos[idx - 1]:
            armado_short = False
            armado_long = False
            continue

        previo = float(distance_z[idx - 1])
        actual = float(distance_z[idx])

        if armado_short and previo >= upper and actual < upper:
            senales[idx] = -1
            armado_short = False
            armado_long = False
            continue
        if armado_long and previo <= lower and actual > lower:
            senales[idx] = 1
            armado_short = False
            armado_long = False
            continue

        if previo <= upper and actual > upper:
            armado_short = True
            armado_long = False
        elif previo >= lower and actual < lower:
            armado_long = True
            armado_short = False

    return senales


@_jit_cache
def _calcular_vwap_distance(
    close: np.ndarray,
    volume: np.ndarray,
    halflife_bars: int,
    normalization_multiplier: float,
    clip_sigmas: float,
) -> tuple[np.ndarray, ...]:
    n = int(close.shape[0])
    vwap = np.empty(n, dtype=np.float64)
    distance_raw = np.zeros(n, dtype=np.float64)
    distance_signal = np.zeros(n, dtype=np.float64)
    distance_z = np.zeros(n, dtype=np.float64)
    if n == 0:
        return vwap, distance_raw, distance_signal, distance_z

    alpha_fast = _alpha_halflife_float(float(halflife_bars))
    alpha_norm = _alpha_halflife_float(max(1.0, float(halflife_bars) * float(normalization_multiplier)))
    clip_sigmas = max(0.1, float(clip_sigmas))

    volume_sum = 0.0
    for idx in range(n):
        volume_sum += max(float(volume[idx]), 0.0)
    usar_volumen_real = volume_sum > VOLUME_EPSILON

    precio0 = _precio_no_negativo(close[0])
    volumen0 = max(float(volume[0]), 0.0) if usar_volumen_real else 1.0
    pv_ewm = precio0 * volumen0
    v_ewm = volumen0
    dist_signal_state = 0.0
    dist_mean = 0.0
    dist_mean_sq = 0.0
    dist_final_mean = 0.0
    dist_final_mean_sq = 0.0

    for idx in range(n):
        precio = _precio_no_negativo(close[idx])
        volumen = max(float(volume[idx]), 0.0)
        volumen_eff = volumen if usar_volumen_real else 1.0
        pv_actual = precio * volumen_eff
        pv_ewm = alpha_fast * pv_actual + (1.0 - alpha_fast) * pv_ewm
        v_ewm = alpha_fast * volumen_eff + (1.0 - alpha_fast) * v_ewm
        vwap_i = pv_ewm / v_ewm if v_ewm >= VOLUME_EPSILON else 0.0
        vwap[idx] = vwap_i

        dist_raw_i = (precio - vwap_i) / vwap_i if precio > 0.0 and vwap_i > 0.0 else 0.0
        distance_raw[idx] = dist_raw_i
        if idx == 0:
            dist_signal_state = dist_raw_i
            dist_mean = dist_signal_state
            dist_mean_sq = dist_signal_state * dist_signal_state
        else:
            dist_signal_state = alpha_fast * dist_raw_i + (1.0 - alpha_fast) * dist_signal_state
            dist_mean = alpha_norm * dist_signal_state + (1.0 - alpha_norm) * dist_mean
            dist_mean_sq = alpha_norm * dist_signal_state * dist_signal_state + (1.0 - alpha_norm) * dist_mean_sq
        distance_signal[idx] = dist_signal_state

        dist_std = math.sqrt(max(0.0, dist_mean_sq - dist_mean * dist_mean))
        if dist_std <= 0.0:
            clipped_dist = dist_signal_state
        else:
            clipped_dist = _clip(dist_signal_state, dist_mean - clip_sigmas * dist_std, dist_mean + clip_sigmas * dist_std)

        if idx == 0:
            dist_final_mean = clipped_dist
            dist_final_mean_sq = clipped_dist * clipped_dist
        else:
            dist_final_mean = alpha_norm * clipped_dist + (1.0 - alpha_norm) * dist_final_mean
            dist_final_mean_sq = alpha_norm * clipped_dist * clipped_dist + (1.0 - alpha_norm) * dist_final_mean_sq
        dist_final_std = math.sqrt(max(0.0, dist_final_mean_sq - dist_final_mean * dist_final_mean))
        distance_z[idx] = (dist_signal_state - dist_final_mean) / dist_final_std if dist_final_std > 0.0 else 0.0

    return vwap, distance_raw, distance_signal, distance_z


@_jit_cache
def _alpha_halflife_float(halflife: float) -> float:
    return 1.0 - math.exp(-math.log(2.0) / max(1.0, float(halflife)))


@_jit_cache
def _precio_no_negativo(value: float) -> float:
    precio = float(value)
    return precio if math.isfinite(precio) and precio > 0.0 else 0.0


@_jit_cache
def _clip(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _serie_overlay(df: pl.DataFrame, valores: np.ndarray, color: str, nombre: str) -> dict:
    ts_seg, vals = _puntos_finitos(df, valores, decimales=6)
    return {
        "nombre": nombre,
        "tipo": "overlay",
        "color": color,
        "data": [{"t": int(t), "v": float(v)} for t, v in zip(ts_seg, vals, strict=False)],
    }


def _serie_pane(
    df: pl.DataFrame,
    valores: np.ndarray,
    color: str,
    nombre: str,
    *,
    niveles: list[dict] | None = None,
) -> dict:
    ts_seg, vals = _puntos_finitos(df, valores, decimales=6)
    payload = {
        "nombre": nombre,
        "tipo": "pane",
        "color": color,
        "data": [{"t": int(t), "v": float(v)} for t, v in zip(ts_seg, vals, strict=False)],
    }
    if niveles:
        payload["niveles"] = niveles
    return payload


def _puntos_finitos(df: pl.DataFrame, valores: np.ndarray, *, decimales: int) -> tuple[np.ndarray, np.ndarray]:
    timestamps_us = df["timestamp"].dt.epoch("us").to_numpy()
    finitos = np.isfinite(valores)
    ts_seg = (timestamps_us[finitos] // 1_000_000).astype(np.int64)
    vals = np.round(valores[finitos], decimales)
    return ts_seg, vals

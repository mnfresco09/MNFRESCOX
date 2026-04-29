"""VWAP-CVD.

Estrategia tendencial con dos piezas separadas:

- la EWM-VWAP define la direccion operable;
- el CVD Z confirma la entrada cuando cruza su umbral absoluto.

La estrategia solo genera entradas. Las salidas las decide el sistema mediante
EXIT_TYPE (FIXED o BARS para esta estrategia).
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


DEFAULT_HALFLIFE_BARS = 15
DEFAULT_NORMALIZATION_MULTIPLIER = 3.0
DEFAULT_CLIP_SIGMAS = 2.5
DEFAULT_UMBRAL_CVD = 1.0
VOLUME_EPSILON = 1e-7

IDX_VWAP = 0
IDX_DISTANCE_RAW = 1
IDX_DISTANCE_SIGNAL = 2
IDX_DISTANCE_Z = 3
IDX_CVD_RAW = 4
IDX_CVD_SIGNAL = 5
IDX_CVD_Z = 6


def _jit_cache(func):
    if njit is None:
        return func
    return njit(cache=True)(func)


_JIT_WARMUP_LOCK = Lock()
_JIT_PRECALENTADO = False


class VWAPCVD(BaseEstrategia):
    ID = 4
    NOMBRE = "VWAP-CVD"
    COLUMNAS_REQUERIDAS = {"volume", "taker_buy_volume", "taker_sell_volume"}

    def parametros_por_defecto(self) -> dict:
        return {
            "halflife_bars": DEFAULT_HALFLIFE_BARS,
            "normalization_multiplier": DEFAULT_NORMALIZATION_MULTIPLIER,
            "vwap_clip_sigmas": DEFAULT_CLIP_SIGMAS,
            "umbral_cvd": DEFAULT_UMBRAL_CVD,
        }

    def espacio_busqueda(self, trial) -> dict:
        return {
            "halflife_bars": trial.suggest_int("halflife_bars", 45, 75, step=1),
            "normalization_multiplier": trial.suggest_float("normalization_multiplier",3.0,5.0,step=0.5,),
            "vwap_clip_sigmas": trial.suggest_float("vwap_clip_sigmas", 2.5, 3.5, step=0.1),
            "umbral_cvd": trial.suggest_float("umbral_cvd",1.3,2.0,step=0.1,),
        }

    def bind(self, arrays, cache=None) -> None:
        super().bind(arrays, cache)
        _precalentar_vwapcvd_jit()

    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        halflife, norm_mult, clip_sigmas, umbral = _normalizar_params(params)
        valores = self._indicadores(df, halflife, norm_mult, clip_sigmas)
        vwap = valores[IDX_VWAP]
        cvd_z = valores[IDX_CVD_Z]
        cvd_z_prev = self.shift(cvd_z, 1)

        finitos = (
            np.isfinite(self.close)
            & np.isfinite(vwap)
            & np.isfinite(cvd_z_prev)
            & np.isfinite(cvd_z)
        )
        long_mask = finitos & (self.close > vwap) & (cvd_z_prev <= umbral) & (cvd_z > umbral)
        short_mask = finitos & (self.close < vwap) & (cvd_z_prev >= -umbral) & (cvd_z < -umbral)
        _bloquear_warmup(long_mask, _warmup(halflife, norm_mult))
        _bloquear_warmup(short_mask, _warmup(halflife, norm_mult))
        return self.serie_senales(df.height, long_mask, short_mask)

    def indicadores_para_grafica(self, df: pl.DataFrame, params: dict) -> list[dict]:
        halflife, norm_mult, clip_sigmas, umbral = _normalizar_params(params)
        valores = self._indicadores(df, halflife, norm_mult, clip_sigmas)
        return [
            _serie_overlay(df, valores[IDX_VWAP], "#00bcd4", f"VWAP-CVD VWAP({halflife})"),
            _serie_pane(
                df,
                valores[IDX_CVD_Z],
                "#22c55e",
                f"CVD Z({halflife})",
                niveles=[
                    {"valor": umbral, "color": "#22c55e66"},
                    {"valor": 0.0, "color": "#64748b88"},
                    {"valor": -umbral, "color": "#ef444466"},
                ],
            ),
            _serie_pane(
                df,
                valores[IDX_DISTANCE_Z],
                "#ab47bc",
                "VWAP DIST Z",
                niveles=[{"valor": 0.0, "color": "#64748b88"}],
            ),
        ]

    def _indicadores(
        self,
        df: pl.DataFrame,
        halflife: int,
        normalization_multiplier: float,
        clip_sigmas: float,
    ) -> tuple[np.ndarray, ...]:
        buy = _columna_float(df, "taker_buy_volume")
        sell = _columna_float(df, "taker_sell_volume")
        return self.memo(
            "vwapcvd",
            id(self.close),
            id(self.volume),
            int(halflife),
            float(normalization_multiplier),
            float(clip_sigmas),
            calcular=lambda: _calcular_vwap_cvd(
                self.close,
                self.volume,
                buy,
                sell,
                int(halflife),
                float(normalization_multiplier),
                float(clip_sigmas),
            ),
        )


def _normalizar_params(params: dict) -> tuple[int, float, float, float]:
    halflife = max(1, int(params.get("halflife_bars", DEFAULT_HALFLIFE_BARS)))
    norm_mult = max(0.1, float(params.get("normalization_multiplier", DEFAULT_NORMALIZATION_MULTIPLIER)))
    clip_sigmas = max(0.1, float(params.get("vwap_clip_sigmas", DEFAULT_CLIP_SIGMAS)))
    umbral = max(0.0, float(params.get("umbral_cvd", params.get("umbral_inicio_tendencia", DEFAULT_UMBRAL_CVD))))
    return halflife, norm_mult, clip_sigmas, umbral


def _warmup(halflife: int, normalization_multiplier: float) -> int:
    normalization_halflife = max(1.0, float(halflife) * float(normalization_multiplier))
    return max(int(halflife) * 9, int(round(normalization_halflife * 3.0)))


def _bloquear_warmup(mask: np.ndarray, warmup: int) -> None:
    if warmup > 0:
        mask[: min(int(warmup), mask.shape[0])] = False


def _precalentar_vwapcvd_jit() -> None:
    if njit is None:
        return

    global _JIT_PRECALENTADO
    if _JIT_PRECALENTADO:
        return

    with _JIT_WARMUP_LOCK:
        if _JIT_PRECALENTADO:
            return
        close, volume, buy, sell = _arrays_warmup(writeable=True)
        _calcular_vwap_cvd(close, volume, buy, sell, 5, 3.0, 2.5)
        close_ro, volume_ro, buy_ro, sell_ro = _arrays_warmup(writeable=False)
        _calcular_vwap_cvd(close_ro, volume_ro, buy_ro, sell_ro, 5, 3.0, 2.5)
        _JIT_PRECALENTADO = True


def _arrays_warmup(*, writeable: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.array([100.0, 100.6, 100.1, 101.2, 101.8, 101.1], dtype=np.float64)
    volume = np.array([10.0, 12.0, 9.0, 14.0, 11.0, 13.0], dtype=np.float64)
    buy = np.array([5.0, 8.0, 3.0, 10.0, 7.0, 4.0], dtype=np.float64)
    sell = volume - buy
    if not writeable:
        close.setflags(write=False)
        volume.setflags(write=False)
        buy.setflags(write=False)
        sell.setflags(write=False)
    return close, volume, buy, sell


@_jit_cache
def _calcular_vwap_cvd(
    close: np.ndarray,
    volume: np.ndarray,
    taker_buy: np.ndarray,
    taker_sell: np.ndarray,
    halflife_bars: int,
    normalization_multiplier: float,
    clip_sigmas: float,
) -> tuple[np.ndarray, ...]:
    n = int(close.shape[0])
    vwap = np.empty(n, dtype=np.float64)
    distance_raw = np.zeros(n, dtype=np.float64)
    distance_signal = np.zeros(n, dtype=np.float64)
    distance_z = np.zeros(n, dtype=np.float64)
    cvd_raw = np.zeros(n, dtype=np.float64)
    cvd_signal = np.zeros(n, dtype=np.float64)
    cvd_z = np.zeros(n, dtype=np.float64)
    if n == 0:
        return (
            vwap,
            distance_raw,
            distance_signal,
            distance_z,
            cvd_raw,
            cvd_signal,
            cvd_z,
        )

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

    cvd_signal_state = 0.0
    cvd_mean = 0.0
    cvd_mean_sq = 0.0
    cvd_final_mean = 0.0
    cvd_final_mean_sq = 0.0

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
        clip_lower = dist_mean - clip_sigmas * dist_std
        clip_upper = dist_mean + clip_sigmas * dist_std
        if dist_std <= 0.0:
            clipped_dist = dist_signal_state
        else:
            clipped_dist = _clip(dist_signal_state, clip_lower, clip_upper)

        if idx == 0:
            dist_final_mean = clipped_dist
            dist_final_mean_sq = clipped_dist * clipped_dist
        else:
            dist_final_mean = alpha_norm * clipped_dist + (1.0 - alpha_norm) * dist_final_mean
            dist_final_mean_sq = (
                alpha_norm * clipped_dist * clipped_dist
                + (1.0 - alpha_norm) * dist_final_mean_sq
            )
        dist_final_std = math.sqrt(max(0.0, dist_final_mean_sq - dist_final_mean * dist_final_mean))
        dist_z_i = (dist_signal_state - dist_final_mean) / dist_final_std if dist_final_std > 0.0 else 0.0
        distance_z[idx] = dist_z_i

        buy = max(float(taker_buy[idx]), 0.0)
        sell = max(float(taker_sell[idx]), 0.0)
        total_cvd = max(volumen, buy + sell)
        cvd_raw_i = (buy - sell) / total_cvd if total_cvd > VOLUME_EPSILON else 0.0
        cvd_raw_i = _clip(cvd_raw_i, -1.0, 1.0)
        cvd_raw[idx] = cvd_raw_i

        if idx == 0:
            cvd_signal_state = cvd_raw_i
            cvd_mean = cvd_signal_state
            cvd_mean_sq = cvd_signal_state * cvd_signal_state
        else:
            cvd_signal_state = alpha_fast * cvd_raw_i + (1.0 - alpha_fast) * cvd_signal_state
            cvd_mean = alpha_norm * cvd_signal_state + (1.0 - alpha_norm) * cvd_mean
            cvd_mean_sq = alpha_norm * cvd_signal_state * cvd_signal_state + (1.0 - alpha_norm) * cvd_mean_sq
        cvd_signal[idx] = cvd_signal_state

        cvd_std = math.sqrt(max(0.0, cvd_mean_sq - cvd_mean * cvd_mean))
        cvd_clip_lower = cvd_mean - clip_sigmas * cvd_std
        cvd_clip_upper = cvd_mean + clip_sigmas * cvd_std
        if cvd_std <= 0.0:
            clipped_cvd = cvd_signal_state
        else:
            clipped_cvd = _clip(cvd_signal_state, cvd_clip_lower, cvd_clip_upper)

        if idx == 0:
            cvd_final_mean = clipped_cvd
            cvd_final_mean_sq = clipped_cvd * clipped_cvd
        else:
            cvd_final_mean = alpha_norm * clipped_cvd + (1.0 - alpha_norm) * cvd_final_mean
            cvd_final_mean_sq = (
                alpha_norm * clipped_cvd * clipped_cvd
                + (1.0 - alpha_norm) * cvd_final_mean_sq
            )
        cvd_final_std = math.sqrt(max(0.0, cvd_final_mean_sq - cvd_final_mean * cvd_final_mean))
        cvd_z_i = (cvd_signal_state - cvd_final_mean) / cvd_final_std if cvd_final_std > 0.0 else 0.0
        cvd_z[idx] = cvd_z_i

    return (
        vwap,
        distance_raw,
        distance_signal,
        distance_z,
        cvd_raw,
        cvd_signal,
        cvd_z,
    )


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


def _columna_float(df: pl.DataFrame, nombre: str) -> np.ndarray:
    serie = df[nombre].cast(pl.Float64).fill_null(0.0).fill_nan(0.0)
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
    niveles: list[dict] | None = None,
) -> dict:
    ts_seg, vals = _puntos_finitos(df, valores, decimales=6)
    payload = {
        "nombre": nombre,
        "tipo": "pane",
        "color": color,
        "data": [{"t": int(t), "v": float(v)} for t, v in zip(ts_seg, vals)],
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

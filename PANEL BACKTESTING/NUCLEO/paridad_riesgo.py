"""Paridad de riesgo por volatilidad EWMA.

Este modulo contiene la matematica y el contrato interno de paridad. Los
valores editables por el usuario viven en `SALIDAS/paridad.py`, junto al resto
de parametros de salida.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from SALIDAS import paridad as paridad_cfg

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba es opcional
    njit = None


@dataclass(frozen=True)
class ParametrosParidadRiesgo:
    activa: bool
    riesgo_max_pct: float = paridad_cfg.RIESGO_MAXIMO_PCT
    vol_halflife: int = paridad_cfg.VOL_HALFLIFE
    sl_ewma_mult: float = paridad_cfg.SL_EWMA_MULT
    tp_ewma_mult: float = paridad_cfg.TP_EWMA_MULT
    trail_act_ewma_mult: float = paridad_cfg.TRAIL_ACT_EWMA_MULT
    trail_dist_ewma_mult: float = paridad_cfg.TRAIL_DIST_EWMA_MULT


def parametros_para_trial(
    trial: Any,
    exit_type: str,
    *,
    activa: bool,
    optimizar: bool,
) -> tuple[ParametrosParidadRiesgo, dict[str, float | int]]:
    """Devuelve parametros de paridad y el dict que se persistira por trial."""
    if not activa:
        return ParametrosParidadRiesgo(activa=False), {}

    if optimizar:
        riesgo = trial.suggest_float(
            "risk_max_pct",
            paridad_cfg.RIESGO_MAXIMO_MIN,
            paridad_cfg.RIESGO_MAXIMO_MAX,
            step=0.5,
        )
        halflife = trial.suggest_int(
            "risk_vol_halflife",
            paridad_cfg.VOL_HALFLIFE_MIN,
            paridad_cfg.VOL_HALFLIFE_MAX,
        )
        sl_mult = trial.suggest_float(
            "risk_sl_ewma_mult",
            paridad_cfg.SL_EWMA_MULT_MIN,
            paridad_cfg.SL_EWMA_MULT_MAX,
            step=0.1,
        )
        tp_mult = paridad_cfg.TP_EWMA_MULT
        trail_act = paridad_cfg.TRAIL_ACT_EWMA_MULT
        trail_dist = paridad_cfg.TRAIL_DIST_EWMA_MULT
        if exit_type == "FIXED":
            tp_mult = trial.suggest_float(
                "risk_tp_ewma_mult",
                paridad_cfg.TP_EWMA_MULT_MIN,
                paridad_cfg.TP_EWMA_MULT_MAX,
                step=0.1,
            )
        elif exit_type == "TRAILING":
            trail_act = trial.suggest_float(
                "risk_trail_act_ewma_mult",
                paridad_cfg.TRAIL_ACT_EWMA_MULT_MIN,
                paridad_cfg.TRAIL_ACT_EWMA_MULT_MAX,
                step=0.1,
            )
            trail_dist = trial.suggest_float(
                "risk_trail_dist_ewma_mult",
                paridad_cfg.TRAIL_DIST_EWMA_MULT_MIN,
                paridad_cfg.TRAIL_DIST_EWMA_MULT_MAX,
                step=0.1,
            )
            trail_act, trail_dist = normalizar_trailing_mult(trail_act, trail_dist)
    else:
        riesgo = paridad_cfg.RIESGO_MAXIMO_PCT
        halflife = paridad_cfg.VOL_HALFLIFE
        sl_mult = paridad_cfg.SL_EWMA_MULT
        tp_mult = paridad_cfg.TP_EWMA_MULT
        trail_act = paridad_cfg.TRAIL_ACT_EWMA_MULT
        trail_dist = paridad_cfg.TRAIL_DIST_EWMA_MULT

    params = ParametrosParidadRiesgo(
        activa=True,
        riesgo_max_pct=float(riesgo),
        vol_halflife=int(halflife),
        sl_ewma_mult=float(sl_mult),
        tp_ewma_mult=float(tp_mult),
        trail_act_ewma_mult=float(trail_act),
        trail_dist_ewma_mult=float(trail_dist),
    )
    return params, parametros_a_dict(params, exit_type)


def parametros_a_dict(
    params: ParametrosParidadRiesgo,
    exit_type: str,
) -> dict[str, float | int]:
    if not params.activa:
        return {}
    salida = {
        "risk_max_pct": float(params.riesgo_max_pct),
        "risk_vol_halflife": int(params.vol_halflife),
        "risk_sl_ewma_mult": float(params.sl_ewma_mult),
    }
    if exit_type == "FIXED":
        salida["risk_tp_ewma_mult"] = float(params.tp_ewma_mult)
    if exit_type == "TRAILING":
        salida["risk_trail_act_ewma_mult"] = float(params.trail_act_ewma_mult)
        salida["risk_trail_dist_ewma_mult"] = float(params.trail_dist_ewma_mult)
    return salida


def params_desde_dict(
    parametros: dict[str, Any],
    exit_type: str,
    *,
    activa: bool,
) -> ParametrosParidadRiesgo:
    if not activa:
        return ParametrosParidadRiesgo(activa=False)
    trail_act = float(
        parametros.get("risk_trail_act_ewma_mult", paridad_cfg.TRAIL_ACT_EWMA_MULT)
    )
    trail_dist = float(
        parametros.get("risk_trail_dist_ewma_mult", paridad_cfg.TRAIL_DIST_EWMA_MULT)
    )
    if exit_type == "TRAILING":
        trail_act, trail_dist = normalizar_trailing_mult(trail_act, trail_dist)
    return ParametrosParidadRiesgo(
        activa=True,
        riesgo_max_pct=float(parametros.get("risk_max_pct", paridad_cfg.RIESGO_MAXIMO_PCT)),
        vol_halflife=int(parametros.get("risk_vol_halflife", paridad_cfg.VOL_HALFLIFE)),
        sl_ewma_mult=float(parametros.get("risk_sl_ewma_mult", paridad_cfg.SL_EWMA_MULT)),
        tp_ewma_mult=float(parametros.get("risk_tp_ewma_mult", paridad_cfg.TP_EWMA_MULT)),
        trail_act_ewma_mult=trail_act,
        trail_dist_ewma_mult=trail_dist,
    )


def normalizar_trailing_mult(act_mult: float, dist_mult: float) -> tuple[float, float]:
    act = abs(float(act_mult)) if float(act_mult) != 0.0 else paridad_cfg.TRAIL_ACT_EWMA_MULT
    dist = abs(float(dist_mult)) if float(dist_mult) != 0.0 else paridad_cfg.TRAIL_DIST_EWMA_MULT
    if dist >= act:
        act, dist = dist, act
    if act == dist:
        act += 0.1
    return float(act), float(dist)


def calcular_volatilidad_ewma(df_o_close: pl.DataFrame | np.ndarray, halflife: int) -> np.ndarray:
    """Volatilidad EWMA causal de retornos logaritmicos.

    `vol[i]` usa `close[i]` y datos anteriores. Si una senal se confirma en la
    vela `i`, esta volatilidad ya es conocida al cierre de esa misma vela.
    """
    if int(halflife) <= 0:
        raise ValueError("[PARIDAD] halflife debe ser mayor que 0.")

    if isinstance(df_o_close, pl.DataFrame):
        if "close" not in df_o_close.columns:
            raise ValueError("[PARIDAD] Falta columna close para calcular volatilidad.")
        close = df_o_close["close"].cast(pl.Float64).to_numpy()
    else:
        close = np.asarray(df_o_close, dtype=np.float64)

    if close.dtype != np.float64 or not close.flags["C_CONTIGUOUS"]:
        close = np.ascontiguousarray(close, dtype=np.float64)
    return _calcular_volatilidad_ewma_close(close, int(halflife))


def proyectar_volatilidad_a_base(
    vol_tf: np.ndarray,
    tf_to_base_idx: np.ndarray,
    base_len: int,
) -> np.ndarray:
    """Proyecta volatilidad del TF de estrategia al cierre operativo base."""
    valores = np.asarray(vol_tf, dtype=np.float64)
    mapeo = np.asarray(tf_to_base_idx, dtype=np.int64)
    if valores.shape[0] != mapeo.shape[0]:
        raise ValueError(
            "[PARIDAD] Longitud de volatilidad y mapeo no coincide: "
            f"{valores.shape[0]:,} != {mapeo.shape[0]:,}."
        )
    arr = np.zeros(int(base_len), dtype=np.float64)
    if mapeo.shape[0] == 0:
        return arr
    validos = (mapeo >= 0) & (mapeo < int(base_len))
    valores_limpios = np.where(np.isfinite(valores), valores, 0.0)
    arr[mapeo[validos]] = valores_limpios[validos]
    return arr


def _jit_cache(func):
    if njit is None:
        return func
    return njit(cache=True)(func)


@_jit_cache
def _calcular_volatilidad_ewma_close(close: np.ndarray, halflife: int) -> np.ndarray:
    n = close.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 1.0 - math.exp(math.log(0.5) / float(halflife))
    var = 0.0
    for idx in range(1, n):
        prev = close[idx - 1]
        curr = close[idx]
        retorno = 0.0
        if prev > 0.0 and curr > 0.0 and math.isfinite(prev) and math.isfinite(curr):
            retorno = math.log(curr / prev)
        var = alpha * retorno * retorno + (1.0 - alpha) * var
        out[idx] = math.sqrt(var) if var > 0.0 else 0.0
    return out

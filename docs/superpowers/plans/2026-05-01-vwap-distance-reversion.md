# VWAP Distance Reversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new standalone VWAP distance mean-reversion strategy without modifying `vwapcvd.py`.

**Architecture:** The new strategy lives in its own module and owns its full VWAP distance calculation. Signal generation is split into a pure state-machine helper so threshold activation/reversion behavior can be tested directly.

**Tech Stack:** Python, NumPy, Polars, optional Numba JIT, `unittest`.

---

## File Structure

- Create: `PANEL BACKTESTING/ESTRATEGIAS/vwap_distance_reversion.py`
  - Defines `VWAPDistanceReversion`.
  - Calculates EWM VWAP and normalized VWAP distance internally.
  - Generates reversal entries through `_generar_senales_reversion()`.
  - Exposes chart indicators for VWAP and `VWAP DIST Z`.
- Create: `PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py`
  - Tests the distance calculation against a local reference implementation.
  - Tests the threshold state machine with explicit `distance_z` arrays.
  - Tests integration through `generar_señales()` and `indicadores_para_grafica()`.
- Do not modify: `PANEL BACKTESTING/ESTRATEGIAS/vwapcvd.py`.

---

### Task 1: Add Failing Tests

**Files:**
- Create: `PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py`

- [ ] **Step 1: Write the failing test file**

Create `PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py` with these tests:

```python
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ESTRATEGIAS.vwap_distance_reversion import (  # noqa: E402
    IDX_DISTANCE_Z,
    IDX_VWAP,
    VWAPDistanceReversion,
    _calcular_vwap_distance,
    _generar_senales_reversion,
    _warmup,
)
from NUCLEO.base_estrategia import CacheIndicadores  # noqa: E402
from NUCLEO.contexto import construir_arrays_motor  # noqa: E402


class VWAPDistanceReversionTest(unittest.TestCase):
    def test_calculo_distancia_vwap_coincide_con_referencia_manual(self) -> None:
        close = np.array([100.0, 100.8, 101.7, 100.9, 99.8, 98.9, 99.7, 100.6], dtype=np.float64)
        volume = np.array([20.0, 24.0, 22.0, 26.0, 21.0, 25.0, 23.0, 27.0], dtype=np.float64)

        esperado = _referencia_vwap_distance(close, volume, 3, 2.0, 2.5)
        valores = _calcular_vwap_distance(close, volume, 3, 2.0, 2.5)

        np.testing.assert_allclose(valores[IDX_VWAP], esperado["vwap"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(valores[IDX_DISTANCE_Z], esperado["distance_z"], rtol=1e-12, atol=1e-12)
        self.assertTrue(np.isfinite(valores[IDX_DISTANCE_Z]).all())
        self.assertEqual(valores[IDX_VWAP].shape, close.shape)

    def test_maquina_de_estados_entra_solo_al_revertir_despues_de_activarse(self) -> None:
        distance_z = np.array([
            0.00,
            0.44,
            0.56,
            0.62,
            0.46,
            0.20,
            -0.20,
            -0.46,
            -0.58,
            -0.64,
            -0.44,
            0.10,
            0.55,
            0.70,
            0.49,
        ], dtype=np.float64)
        finitos = np.isfinite(distance_z)

        senales = _generar_senales_reversion(distance_z, finitos, 0.5, 0)

        esperado = np.zeros(distance_z.shape[0], dtype=np.int8)
        esperado[4] = -1
        esperado[10] = 1
        esperado[14] = -1
        np.testing.assert_array_equal(senales, esperado)

    def test_no_entra_sin_activacion_previa_ni_a_traves_de_no_finitos(self) -> None:
        distance_z = np.array([0.49, 0.46, 0.20, np.nan, 0.62, 0.44], dtype=np.float64)
        finitos = np.isfinite(distance_z)

        senales = _generar_senales_reversion(distance_z, finitos, 0.5, 0)

        np.testing.assert_array_equal(senales, np.zeros(distance_z.shape[0], dtype=np.int8))

    def test_warmup_ignora_activaciones_tempranas(self) -> None:
        distance_z = np.array([0.00, 0.62, 0.44, 0.61, 0.43], dtype=np.float64)
        finitos = np.isfinite(distance_z)

        senales = _generar_senales_reversion(distance_z, finitos, 0.5, 3)

        esperado = np.zeros(distance_z.shape[0], dtype=np.int8)
        esperado[4] = -1
        np.testing.assert_array_equal(senales, esperado)

    def test_estrategia_no_requiere_cvd_y_expone_indicadores(self) -> None:
        df = _df_vwap_reversion(360)
        params = {
            "halflife_bars": 4,
            "normalization_multiplier": 1.0,
            "vwap_clip_sigmas": 2.5,
            "umbral_distance_z": 0.45,
        }

        estrategia = VWAPDistanceReversion()
        self.assertEqual(estrategia.COLUMNAS_REQUERIDAS, {"volume"})
        estrategia.bind(construir_arrays_motor(df), CacheIndicadores())
        try:
            senales = getattr(estrategia, "generar_se\u00f1ales")(df, params).to_numpy()
            indicadores = estrategia.indicadores_para_grafica(df, params)
            valores = _calcular_vwap_distance(
                df["close"].to_numpy(),
                df["volume"].to_numpy(),
                params["halflife_bars"],
                params["normalization_multiplier"],
                params["vwap_clip_sigmas"],
            )
        finally:
            estrategia.desvincular()

        finitos = (
            np.isfinite(df["close"].to_numpy())
            & np.isfinite(valores[IDX_VWAP])
            & np.isfinite(valores[IDX_DISTANCE_Z])
        )
        esperado = _generar_senales_reversion(
            valores[IDX_DISTANCE_Z],
            finitos,
            params["umbral_distance_z"],
            _warmup(params["halflife_bars"], params["normalization_multiplier"]),
        )

        np.testing.assert_array_equal(senales, esperado)
        self.assertEqual(set(np.unique(senales)).issubset({-1, 0, 1}), True)
        self.assertGreater(int((senales == 1).sum()), 0)
        self.assertGreater(int((senales == -1).sum()), 0)
        self.assertEqual([item["tipo"] for item in indicadores], ["overlay", "pane"])
        self.assertEqual(indicadores[1]["nombre"], "VWAP DIST Z")
        self.assertEqual([nivel["valor"] for nivel in indicadores[1]["niveles"]], [0.45, 0.0, -0.45])


def _referencia_vwap_distance(
    close: np.ndarray,
    volume: np.ndarray,
    halflife_bars: int,
    normalization_multiplier: float,
    clip_sigmas: float,
) -> dict[str, np.ndarray]:
    n = close.shape[0]
    vwap = np.empty(n, dtype=np.float64)
    distance_z = np.zeros(n, dtype=np.float64)
    distance_signal = np.zeros(n, dtype=np.float64)
    distance_raw = np.zeros(n, dtype=np.float64)
    if n == 0:
        return {"vwap": vwap, "distance_z": distance_z}

    alpha_fast = _alpha(float(halflife_bars))
    alpha_norm = _alpha(max(1.0, float(halflife_bars) * float(normalization_multiplier)))
    clip_sigmas = max(0.1, float(clip_sigmas))
    usar_volumen_real = float(np.maximum(volume, 0.0).sum()) > 1e-7

    precio0 = _precio_no_negativo(float(close[0]))
    volumen0 = max(float(volume[0]), 0.0) if usar_volumen_real else 1.0
    pv_ewm = precio0 * volumen0
    v_ewm = volumen0
    dist_signal_state = 0.0
    dist_mean = 0.0
    dist_mean_sq = 0.0
    dist_final_mean = 0.0
    dist_final_mean_sq = 0.0

    for idx in range(n):
        precio = _precio_no_negativo(float(close[idx]))
        volumen = max(float(volume[idx]), 0.0)
        volumen_eff = volumen if usar_volumen_real else 1.0
        pv_actual = precio * volumen_eff
        pv_ewm = alpha_fast * pv_actual + (1.0 - alpha_fast) * pv_ewm
        v_ewm = alpha_fast * volumen_eff + (1.0 - alpha_fast) * v_ewm
        vwap_i = pv_ewm / v_ewm if v_ewm >= 1e-7 else 0.0
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
            clipped_dist = min(max(dist_signal_state, dist_mean - clip_sigmas * dist_std), dist_mean + clip_sigmas * dist_std)

        if idx == 0:
            dist_final_mean = clipped_dist
            dist_final_mean_sq = clipped_dist * clipped_dist
        else:
            dist_final_mean = alpha_norm * clipped_dist + (1.0 - alpha_norm) * dist_final_mean
            dist_final_mean_sq = alpha_norm * clipped_dist * clipped_dist + (1.0 - alpha_norm) * dist_final_mean_sq

        dist_final_std = math.sqrt(max(0.0, dist_final_mean_sq - dist_final_mean * dist_final_mean))
        distance_z[idx] = (dist_signal_state - dist_final_mean) / dist_final_std if dist_final_std > 0.0 else 0.0

    return {"vwap": vwap, "distance_z": distance_z}


def _df_vwap_reversion(n: int) -> pl.DataFrame:
    idx = np.arange(n, dtype=np.float64)
    close = 100.0 + 1.8 * np.sin(idx / 5.0) + 0.7 * np.sin(idx / 13.0)
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    span = np.full(n, 0.35, dtype=np.float64)
    volume = 120.0 + 25.0 * (np.sin(idx / 11.0) + 1.0)
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=n - 1),
                interval="1m",
                eager=True,
            ),
            "open": open_,
            "high": np.maximum(open_, close) + span,
            "low": np.minimum(open_, close) - span,
            "close": close,
            "volume": volume,
        }
    )


def _alpha(halflife: float) -> float:
    return 1.0 - math.exp(-math.log(2.0) / max(1.0, float(halflife)))


def _precio_no_negativo(value: float) -> float:
    return value if math.isfinite(value) and value > 0.0 else 0.0


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
python -m unittest 'PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py'
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ESTRATEGIAS.vwap_distance_reversion'`.

---

### Task 2: Implement Standalone Strategy

**Files:**
- Create: `PANEL BACKTESTING/ESTRATEGIAS/vwap_distance_reversion.py`
- Test: `PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py`

- [ ] **Step 1: Add the strategy module**

Create `PANEL BACKTESTING/ESTRATEGIAS/vwap_distance_reversion.py` with:

```python
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
except ImportError:  # pragma: no cover
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
            "halflife_bars": trial.suggest_int("halflife_bars", 8, 80, step=1),
            "normalization_multiplier": trial.suggest_float("normalization_multiplier", 1.5, 5.0, step=0.5),
            "vwap_clip_sigmas": trial.suggest_float("vwap_clip_sigmas", 2.0, 3.5, step=0.1),
            "umbral_distance_z": trial.suggest_float("umbral_distance_z", 0.30, 1.50, step=0.05),
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
```

Then add the helpers shown in Task 2 Step 2.

- [ ] **Step 2: Add calculation, state machine, and chart helpers**

Append these helpers in the same file:

```python
def _bloquear_warmup(mask: np.ndarray, warmup: int) -> None:
    if warmup > 0:
        mask[: min(int(warmup), mask.shape[0])] = False


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
```

- [ ] **Step 3: Run the focused test to verify GREEN**

Run:

```bash
python -m unittest 'PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py'
```

Expected: PASS.

- [ ] **Step 4: Run related tests**

Run:

```bash
python -m unittest 'PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py' 'PANEL BACKTESTING/TESTS/test_vwapcvd.py'
```

Expected: PASS for both files. This also checks that `vwapcvd.py` was not needed for the new strategy.

---

### Task 3: Repository Verification

**Files:**
- Check: `PANEL BACKTESTING/ESTRATEGIAS/vwapcvd.py`
- Check: `PANEL BACKTESTING/ESTRATEGIAS/vwap_distance_reversion.py`
- Check: `PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py`

- [ ] **Step 1: Confirm `vwapcvd.py` is untouched by this implementation**

Run:

```bash
git diff -- 'PANEL BACKTESTING/ESTRATEGIAS/vwapcvd.py'
```

Expected: output may show pre-existing user changes, but no hunks from this implementation. If new hunks appear from this task, revert only those hunks manually.

- [ ] **Step 2: Run syntax and whitespace checks for touched files**

Run:

```bash
python -m py_compile 'PANEL BACKTESTING/ESTRATEGIAS/vwap_distance_reversion.py' 'PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py'
git diff --check -- 'PANEL BACKTESTING/ESTRATEGIAS/vwap_distance_reversion.py' 'PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py'
```

Expected: both commands exit 0.

- [ ] **Step 3: Review final diff**

Run:

```bash
git diff -- 'PANEL BACKTESTING/ESTRATEGIAS/vwap_distance_reversion.py' 'PANEL BACKTESTING/TESTS/test_vwap_distance_reversion.py'
```

Expected: diff contains only the new strategy and its test file.

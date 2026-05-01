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
        distance_z = np.array(
            [
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
            ],
            dtype=np.float64,
        )
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
            clipped_dist = min(
                max(dist_signal_state, dist_mean - clip_sigmas * dist_std),
                dist_mean + clip_sigmas * dist_std,
            )

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

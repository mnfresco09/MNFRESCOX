from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ESTRATEGIAS.vwapcvd import (  # noqa: E402
    IDX_CVD_Z,
    IDX_VWAP,
    VWAPCVD,
    _calcular_vwap_cvd,
    _warmup,
)
from NUCLEO.base_estrategia import CacheIndicadores  # noqa: E402
from NUCLEO.contexto import construir_arrays_motor  # noqa: E402


class VWAPCVDTest(unittest.TestCase):
    def test_entradas_usan_vwap_como_direccion_y_cruce_cvd_z_como_trigger(self) -> None:
        df = _df_vwapcvd(1200)
        params = {
            "halflife_bars": 8,
            "normalization_multiplier": 1.5,
            "vwap_clip_sigmas": 2.0,
            "umbral_cvd": 0.8,
        }

        estrategia = VWAPCVD()
        estrategia.bind(construir_arrays_motor(df), CacheIndicadores())
        try:
            senales = getattr(estrategia, "generar_se\u00f1ales")(df, params).to_numpy()
            indicadores = estrategia.indicadores_para_grafica(df, params)
        finally:
            estrategia.desvincular()

        valores = _calcular_vwap_cvd(
            df["close"].to_numpy(),
            df["volume"].to_numpy(),
            df["taker_buy_volume"].to_numpy(),
            df["taker_sell_volume"].to_numpy(),
            params["halflife_bars"],
            params["normalization_multiplier"],
            params["vwap_clip_sigmas"],
        )
        vwap = valores[IDX_VWAP]
        cvd_z = valores[IDX_CVD_Z]
        cvd_z_prev = _shift(cvd_z)
        close = df["close"].to_numpy()
        umbral = params["umbral_cvd"]

        esperado = np.zeros(df.height, dtype=np.int8)
        finitos = np.isfinite(close) & np.isfinite(vwap) & np.isfinite(cvd_z_prev) & np.isfinite(cvd_z)
        long_mask = finitos & (close > vwap) & (cvd_z_prev <= umbral) & (cvd_z > umbral)
        short_mask = finitos & (close < vwap) & (cvd_z_prev >= -umbral) & (cvd_z < -umbral)
        warmup = _warmup(params["halflife_bars"], params["normalization_multiplier"])
        long_mask[:warmup] = False
        short_mask[:warmup] = False
        esperado[long_mask] = 1
        esperado[short_mask] = -1

        np.testing.assert_array_equal(senales, esperado)
        self.assertEqual({valor: int((senales == valor).sum()) for valor in (-1, 0, 1)}, {-1: 14, 0: 1173, 1: 13})
        self.assertTrue((close[senales == 1] > vwap[senales == 1]).all())
        self.assertTrue((close[senales == -1] < vwap[senales == -1]).all())
        self.assertTrue((cvd_z[senales == 1] > umbral).all())
        self.assertTrue((cvd_z[senales == -1] < -umbral).all())
        self.assertEqual([item["tipo"] for item in indicadores], ["overlay", "pane", "pane"])
        self.assertEqual(indicadores[1]["nombre"], "CVD Z(8)")


def _shift(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = np.nan
    out[1:] = arr[:-1]
    return out


def _df_vwapcvd(n: int) -> pl.DataFrame:
    idx = np.arange(n, dtype=np.float64)
    ret = 0.0007 * np.sin(idx / 17.0) + 0.00045 * np.cos(idx / 31.0)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    span = close * (0.001 + 0.0004 * (np.sin(idx / 9.0) + 1.0))
    high = np.maximum(open_, close) + span
    low = np.minimum(open_, close) - span
    volume = 150.0 + 40.0 * (np.sin(idx / 23.0) + 1.0)
    buy_ratio = np.clip(
        0.5 + 0.35 * np.sin(idx / 5.0) + 0.12 * np.sign(np.sin(idx / 41.0)),
        0.02,
        0.98,
    )
    taker_buy = volume * buy_ratio
    taker_sell = volume - taker_buy
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=n - 1),
                interval="1m",
                eager=True,
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "taker_buy_volume": taker_buy,
            "taker_sell_volume": taker_sell,
        }
    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DATOS.perturbaciones import (  # noqa: E402
    ConfiguracionPerturbaciones,
    aplicar_perturbaciones,
    seed_para_trial,
    validar_kernel_numba,
)
from DATOS.resampleo import resamplear  # noqa: E402
from ESTRATEGIAS.vat_absorcion import (  # noqa: E402
    VwapAbsorptionTrend,
    _calcular_vat,
    _precalentar_vat_jit,
    _warmup,
)
from NUCLEO.base_estrategia import CacheIndicadores  # noqa: E402
from NUCLEO.contexto import construir_arrays_motor  # noqa: E402
from NUCLEO.proyeccion import construir_mapeo, proyectar_senales_a_base  # noqa: E402


class VATCalculosTest(unittest.TestCase):
    def test_calculo_vat_coincide_con_referencia_manual(self) -> None:
        close = np.array([100.0, 101.0, 100.5, 102.0, 101.8, 103.0, 102.7, 104.2])
        volume = np.array([100.0, 120.0, 80.0, 150.0, 90.0, 160.0, 110.0, 170.0])
        delta = np.array([0.0, 60.0, -40.0, 120.0, -10.0, 130.0, -55.0, 140.0])
        hl_vwap = 3
        hl_cvd = 2

        esperado = _referencia_vat(close, volume, delta, hl_vwap, hl_cvd)
        vwap, vat_z = _calcular_vat(close, volume, delta, hl_vwap, hl_cvd)

        np.testing.assert_allclose(vwap, esperado["vwap"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vat_z, esperado["vat_z"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(esperado["cvd"], delta / volume, rtol=1e-12, atol=1e-12)
        anchors = np.array([0, 1, 2, 3, 4, 7])
        np.testing.assert_allclose(
            vwap[anchors],
            np.array([
                100.0,
                100.2377498356,
                100.2814030641,
                100.7664645863,
                100.9482053272,
                102.4344310262,
            ]),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            esperado["cvd_z"][anchors],
            np.array([0.0, 2.8575855453, -2.1762043086, 2.1743021224, -0.5418812083, 1.4757455982]),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            esperado["ret_z"][anchors],
            np.array([0.0, 2.8575855453, -1.5405208943, 2.3082284020, -0.6527882513, 1.6308906656]),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            vat_z[anchors],
            np.array([
                0.0,
                2.8575853144974657,
                -2.1735421507467048,
                2.1742070534429896,
                -0.3629744132944967,
                1.4638606461749648,
            ]),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertAlmostEqual(esperado["retorno"][3], (102.0 - 100.5) / 100.5)
        self.assertTrue(np.isfinite(esperado["cvd_z"][4]))
        self.assertTrue(np.isfinite(esperado["ret_z"][4]))

        alpha_vwap = _alpha(hl_vwap)
        pv_0 = close[0] * volume[0]
        v_0 = volume[0]
        pv_1 = alpha_vwap * close[1] * volume[1] + (1.0 - alpha_vwap) * pv_0
        v_1 = alpha_vwap * volume[1] + (1.0 - alpha_vwap) * v_0
        self.assertAlmostEqual(vwap[1], pv_1 / v_1)

    def test_jit_precalentado_cubre_arrays_readonly_y_paralelo(self) -> None:
        if not hasattr(_calcular_vat, "signatures"):
            self.skipTest("Numba no disponible.")

        _precalentar_vat_jit()
        self.assertGreaterEqual(len(_calcular_vat.signatures), 2)

        idx = np.arange(96, dtype=np.float64)
        close = 100.0 + np.cumsum(0.02 * np.sin(idx / 5.0))
        volume = 100.0 + 5.0 * np.cos(idx / 7.0)
        delta = volume * 0.2 * np.sin(idx / 3.0)
        close.setflags(write=False)
        volume.setflags(write=False)
        delta.setflags(write=False)

        def calcular(offset: int) -> float:
            _, vat_z = _calcular_vat(close, volume, delta, 5 + offset % 3, 3 + offset % 2)
            return float(vat_z[-1])

        with ThreadPoolExecutor(max_workers=8) as executor:
            resultados = list(executor.map(calcular, range(24)))

        self.assertTrue(np.isfinite(resultados).all())

    def test_senales_salidas_y_warmup_usan_componentes_vat(self) -> None:
        df = _df_order_flow(1200)
        estrategia = VwapAbsorptionTrend()
        estrategia.bind(construir_arrays_motor(df), CacheIndicadores())
        params = {"hl_vwap": 12, "hl_cvd": 6, "umbral": 0.35}
        try:
            senales = getattr(estrategia, "generar_se\u00f1ales")(df, params).to_numpy()
            salidas = estrategia.generar_salidas(df, params).to_numpy()
            vwap, vat_z = _calcular_vat(
                df["close"].to_numpy(),
                df["volume"].to_numpy(),
                df["vol_delta"].to_numpy(),
                params["hl_vwap"],
                params["hl_cvd"],
            )
        finally:
            estrategia.desvincular()

        warmup = _warmup(params["hl_vwap"], params["hl_cvd"])
        long_esperado = (df["close"].to_numpy() > vwap) & (vat_z > params["umbral"])
        short_esperado = (df["close"].to_numpy() <= vwap) & (vat_z < -params["umbral"])
        long_esperado[:warmup] = False
        short_esperado[:warmup] = False
        esperado = np.zeros(df.height, dtype=np.int8)
        esperado[long_esperado] = 1
        esperado[short_esperado] = -1

        np.testing.assert_array_equal(senales, esperado)
        self.assertEqual({valor: int((senales == valor).sum()) for valor in (-1, 0, 1)}, {-1: 214, 0: 764, 1: 222})
        self.assertEqual({valor: int((salidas == valor).sum()) for valor in (-1, 0, 1)}, {-1: 567, 0: 54, 1: 579})
        self.assertEqual(int(np.abs(senales[:warmup]).sum()), 0)
        self.assertEqual(set(np.unique(senales)).issubset({-1, 0, 1}), True)
        self.assertEqual(set(np.unique(salidas)).issubset({-1, 0, 1}), True)


class ResampleoOrderFlowTest(unittest.TestCase):
    def test_resampleo_etiqueta_apertura_y_suma_order_flow(self) -> None:
        df = _df_resampleo_1m()
        out = resamplear(df, "5m")

        self.assertEqual(out.height, 3)
        self.assertEqual(out["timestamp"].dt.minute().to_list(), [0, 5, 10])
        np.testing.assert_allclose(out["volume"].to_numpy(), np.array([515.0, 552.5, 590.0]))
        np.testing.assert_allclose(out["taker_buy_volume"].to_numpy(), np.array([310.0, 335.0, 360.0]))
        np.testing.assert_allclose(out["taker_sell_volume"].to_numpy(), np.array([205.0, 217.5, 230.0]))
        np.testing.assert_allclose(out["vol_delta"].to_numpy(), np.array([105.0, 117.5, 130.0]))
        np.testing.assert_allclose(out["quote_volume"].to_numpy(), np.array([51709.0, 56026.5, 60419.0]))
        np.testing.assert_array_equal(out["num_trades"].to_numpy(), np.array([60, 85, 110]))

        primera = out.row(0, named=True)
        self.assertEqual(primera["open"], df["open"][0])
        self.assertEqual(primera["high"], max(df["high"][:5]))
        self.assertEqual(primera["low"], min(df["low"][:5]))
        self.assertEqual(primera["close"], df["close"][4])
        self.assertAlmostEqual(primera["volume"], float(df["volume"][:5].sum()))
        self.assertAlmostEqual(primera["taker_buy_volume"], float(df["taker_buy_volume"][:5].sum()))
        self.assertAlmostEqual(primera["taker_sell_volume"], float(df["taker_sell_volume"][:5].sum()))
        self.assertAlmostEqual(primera["vol_delta"], float(df["vol_delta"][:5].sum()))
        self.assertAlmostEqual(primera["quote_volume"], float(df["quote_volume"][:5].sum()))
        self.assertEqual(primera["num_trades"], int(df["num_trades"][:5].sum()))

        for row in out.iter_rows(named=True):
            self.assertAlmostEqual(
                row["taker_buy_volume"] + row["taker_sell_volume"],
                row["volume"],
                places=10,
            )
            self.assertAlmostEqual(
                row["taker_buy_volume"] - row["taker_sell_volume"],
                row["vol_delta"],
                places=10,
            )

    def test_proyeccion_usa_cierre_operativo_de_velas_resampleadas(self) -> None:
        df = _df_resampleo_1m()
        out = resamplear(df, "5m")
        mapeo = construir_mapeo(df, out, timeframe="5m")
        np.testing.assert_array_equal(mapeo, np.array([4, 9, 14], dtype=np.int64))

        senales_tf = pl.Series("senal", np.array([1, -1, 1], dtype=np.int8))
        senales_base = proyectar_senales_a_base(senales_tf, mapeo, df.height).to_numpy()
        esperado = np.zeros(df.height, dtype=np.int8)
        esperado[[4, 9, 14]] = [1, -1, 1]
        np.testing.assert_array_equal(senales_base, esperado)


class PerturbacionesVATTest(unittest.TestCase):
    def test_vat_resampleado_sigue_consistente_despues_de_perturbaciones(self) -> None:
        try:
            validar_kernel_numba()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        df = _df_order_flow(720)
        config = ConfiguracionPerturbaciones(
            activa=True,
            seed_global=123,
            granularidad_cubos=0.005,
            percentil_tabla=0.10,
        ).con_tabla_desde(df)
        seed = seed_para_trial(
            config,
            trial_numero=7,
            activo="BTC",
            timeframe="5m",
            estrategia_id=3,
            salida_tipo="FIXED",
        )
        self.assertEqual(seed, 271634011)
        perturbado = aplicar_perturbaciones(df, config, seed=seed)

        self.assertEqual(perturbado.height, df.height)
        self.assertTrue(perturbado["timestamp"].equals(df["timestamp"]))
        _assert_order_flow_consistente(self, perturbado)

        resampled = resamplear(perturbado, "5m")
        self.assertEqual(resampled.height, 144)
        _assert_order_flow_consistente(self, resampled)
        self.assertAlmostEqual(float(df["volume"].sum()), 104862.222473, places=6)
        self.assertAlmostEqual(float(perturbado["volume"].sum()), 105676.757542, places=6)
        self.assertAlmostEqual(float(resampled["volume"].sum()), 105676.757542, places=6)

        primera = resampled.row(0, named=True)
        self.assertAlmostEqual(primera["volume"], float(perturbado["volume"][:5].sum()))
        self.assertAlmostEqual(primera["taker_buy_volume"], float(perturbado["taker_buy_volume"][:5].sum()))
        self.assertAlmostEqual(primera["taker_sell_volume"], float(perturbado["taker_sell_volume"][:5].sum()))
        self.assertAlmostEqual(primera["vol_delta"], float(perturbado["vol_delta"][:5].sum()))
        self.assertAlmostEqual(primera["volume"], 718.033940, places=6)
        self.assertAlmostEqual(primera["taker_buy_volume"], 379.974226, places=6)
        self.assertAlmostEqual(primera["taker_sell_volume"], 338.059715, places=6)
        self.assertAlmostEqual(primera["vol_delta"], 41.914511, places=6)

        estrategia = VwapAbsorptionTrend()
        estrategia.bind(construir_arrays_motor(resampled), CacheIndicadores())
        params = {"hl_vwap": 20, "hl_cvd": 8, "umbral": 0.5}
        try:
            senales = getattr(estrategia, "generar_se\u00f1ales")(resampled, params).to_numpy()
            indicadores = estrategia.indicadores_para_grafica(resampled, params)
            vwap, vat_z = _calcular_vat(
                resampled["close"].to_numpy(),
                resampled["volume"].to_numpy(),
                resampled["vol_delta"].to_numpy(),
                params["hl_vwap"],
                params["hl_cvd"],
            )
        finally:
            estrategia.desvincular()

        referencia = _referencia_vat(
            resampled["close"].to_numpy(),
            resampled["volume"].to_numpy(),
            resampled["vol_delta"].to_numpy(),
            params["hl_vwap"],
            params["hl_cvd"],
        )
        np.testing.assert_allclose(vwap, referencia["vwap"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vat_z, referencia["vat_z"], rtol=1e-12, atol=1e-12)
        self.assertEqual(set(np.unique(senales)).issubset({-1, 0, 1}), True)
        self.assertEqual([item["tipo"] for item in indicadores], ["overlay", "pane"])


def _referencia_vat(
    close: np.ndarray,
    volume: np.ndarray,
    delta: np.ndarray,
    hl_vwap: int,
    hl_cvd: int,
) -> dict[str, np.ndarray]:
    n = close.shape[0]
    vwap = np.empty(n, dtype=np.float64)
    vat_z_arr = np.zeros(n, dtype=np.float64)
    cvd_arr = np.zeros(n, dtype=np.float64)
    retorno_arr = np.zeros(n, dtype=np.float64)
    cvd_z_arr = np.zeros(n, dtype=np.float64)
    ret_z_arr = np.zeros(n, dtype=np.float64)

    av = _alpha(hl_vwap)
    an = _alpha(hl_cvd * 3)

    pv_ewm = close[0] * volume[0]
    v_ewm = volume[0]
    cm = cms = rm = rms = 0.0
    for idx in range(n):
        pv_ewm = av * (close[idx] * volume[idx]) + (1.0 - av) * pv_ewm
        v_ewm = av * volume[idx] + (1.0 - av) * v_ewm
        vwap[idx] = pv_ewm / v_ewm if v_ewm > 1e-12 else close[idx]

        cvd = delta[idx] / volume[idx] if volume[idx] > 1e-12 else 0.0
        cvd = min(1.0, max(-1.0, cvd))
        retorno = (close[idx] - close[idx - 1]) / close[idx - 1] if idx > 0 else 0.0
        cvd_arr[idx] = cvd
        retorno_arr[idx] = retorno

        cm = an * cvd + (1.0 - an) * cm
        cms = an * cvd * cvd + (1.0 - an) * cms
        rm = an * retorno + (1.0 - an) * rm
        rms = an * retorno * retorno + (1.0 - an) * rms
        cvd_std = math.sqrt(max(0.0, cms - cm * cm))
        ret_std = math.sqrt(max(0.0, rms - rm * rm))
        cvd_z = (cvd - cm) / cvd_std if cvd_std > 1e-12 else 0.0
        ret_z = (retorno - rm) / ret_std if ret_std > 1e-12 else 0.0
        cvd_z_arr[idx] = cvd_z
        ret_z_arr[idx] = ret_z

        acuerdo = math.tanh(cvd_z * ret_z)
        vat_z_arr[idx] = cvd_z * (0.5 + 0.5 * acuerdo)

    return {
        "vwap": vwap,
        "vat_z": vat_z_arr,
        "cvd": cvd_arr,
        "retorno": retorno_arr,
        "cvd_z": cvd_z_arr,
        "ret_z": ret_z_arr,
    }


def _alpha(halflife: int) -> float:
    return 1.0 - math.exp(-math.log(2.0) / float(halflife))


def _df_order_flow(n: int) -> pl.DataFrame:
    idx = np.arange(n, dtype=np.float64)
    base_ret = 0.0008 * np.sin(idx / 11.0) + 0.00035 * np.cos(idx / 29.0)
    close = 100.0 * np.exp(np.cumsum(base_ret))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    rango = close * (0.0015 + 0.0005 * (np.sin(idx / 13.0) + 1.0))
    high = np.maximum(open_, close) + rango
    low = np.minimum(open_, close) - rango
    volume = 120.0 + 25.0 * (np.sin(idx / 17.0) + 1.0)
    sell_prop = np.clip(0.5 - base_ret * 180.0 + 0.10 * np.sin(idx / 7.0), 0.05, 0.95)
    taker_sell = volume * sell_prop
    taker_buy = volume - taker_sell
    vol_delta = taker_buy - taker_sell
    quote_volume = volume * (high + low) * 0.5
    taker_buy_quote_volume = taker_buy * (high + low) * 0.5
    num_trades = (80 + (volume * 1.5)).astype(np.int64)
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
            "quote_volume": quote_volume,
            "num_trades": num_trades,
            "taker_buy_volume": taker_buy,
            "taker_buy_quote_volume": taker_buy_quote_volume,
            "taker_sell_volume": taker_sell,
            "vol_delta": vol_delta,
        }
    )


def _df_resampleo_1m() -> pl.DataFrame:
    n = 15
    idx = np.arange(n, dtype=np.float64)
    taker_buy = 60.0 + idx
    taker_sell = 40.0 + idx * 0.5
    volume = taker_buy + taker_sell
    close = 100.0 + idx * 0.2
    open_ = close - 0.1
    high = close + (idx % 5) * 0.1
    low = open_ - (idx % 3) * 0.1
    quote_volume = volume * close
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
            "quote_volume": quote_volume,
            "num_trades": np.arange(10, 10 + n, dtype=np.int64),
            "taker_buy_volume": taker_buy,
            "taker_buy_quote_volume": taker_buy * close,
            "taker_sell_volume": taker_sell,
            "vol_delta": taker_buy - taker_sell,
        }
    )


def _assert_order_flow_consistente(test: unittest.TestCase, df: pl.DataFrame) -> None:
    buy = df["taker_buy_volume"].to_numpy()
    sell = df["taker_sell_volume"].to_numpy()
    volume = df["volume"].to_numpy()
    delta = df["vol_delta"].to_numpy()
    test.assertTrue(np.isfinite(volume).all())
    test.assertTrue((volume >= 0.0).all())
    np.testing.assert_allclose(buy + sell, volume, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(buy - sell, delta, rtol=1e-9, atol=1e-9)
    test.assertTrue((df["high"].to_numpy() >= df["low"].to_numpy()).all())
    test.assertTrue((df["open"].to_numpy() >= df["low"].to_numpy()).all())
    test.assertTrue((df["open"].to_numpy() <= df["high"].to_numpy()).all())
    test.assertTrue((df["close"].to_numpy() >= df["low"].to_numpy()).all())
    test.assertTrue((df["close"].to_numpy() <= df["high"].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()

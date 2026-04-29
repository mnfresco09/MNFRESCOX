from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MOTOR import MOTIVOS, simular_full  # noqa: E402
from NUCLEO.contexto import SimConfigMotor  # noqa: E402
from NUCLEO.paridad_riesgo import (  # noqa: E402
    calcular_volatilidad_ewma,
    proyectar_volatilidad_a_base,
)


class ParidadRiesgoTest(unittest.TestCase):
    def test_volatilidad_ewma_es_causal_y_reproduce_la_formula_manual(self) -> None:
        close = np.array([100.0, 101.0, 100.0, 102.0], dtype=np.float64)
        df = pl.DataFrame({"close": close})
        halflife = 2
        alpha = 1.0 - math.exp(math.log(0.5) / float(halflife))

        esperado = np.zeros(close.shape[0], dtype=np.float64)
        var = 0.0
        for idx in range(1, close.shape[0]):
            retorno = math.log(close[idx] / close[idx - 1])
            var = alpha * retorno * retorno + (1.0 - alpha) * var
            esperado[idx] = math.sqrt(var)

        calculado = calcular_volatilidad_ewma(df, halflife)

        np.testing.assert_allclose(calculado, esperado, rtol=0.0, atol=1e-12)
        self.assertEqual(float(calculado[0]), 0.0)

    def test_proyeccion_de_volatilidad_solo_coloca_valores_en_cierres_operativos(self) -> None:
        vol_tf = np.array([0.01, 0.02, 0.03], dtype=np.float64)
        mapeo = np.array([4, 9, 14], dtype=np.int64)

        proyectada = proyectar_volatilidad_a_base(vol_tf, mapeo, base_len=16)

        esperado = np.zeros(16, dtype=np.float64)
        esperado[[4, 9, 14]] = vol_tf
        np.testing.assert_array_equal(proyectada, esperado)

    def test_motor_usa_volatilidad_de_la_senal_para_apalancamiento_y_sl(self) -> None:
        arrays = SimpleNamespace(
            timestamps=np.array([1, 2, 3], dtype=np.int64),
            opens=np.array([100.0, 100.0, 100.0], dtype=np.float64),
            highs=np.array([100.0, 100.1, 100.2], dtype=np.float64),
            lows=np.array([100.0, 99.79, 99.5], dtype=np.float64),
            closes=np.array([100.0, 100.0, 99.6], dtype=np.float64),
            salidas_neutras=np.zeros(3, dtype=np.int8),
        )
        cfg = SimConfigMotor(
            saldo_inicial=10_000.0,
            saldo_por_trade=500.0,
            apalancamiento=35.0,
            saldo_minimo=1_000.0,
            comision_pct=0.0,
            comision_lados=2,
            exit_type="FIXED",
            exit_sl_pct=20.0,
            exit_tp_pct=40.0,
            exit_velas=0,
            paridad_riesgo=True,
            paridad_riesgo_max_pct=5.0,
            paridad_apalancamiento_min=1.0,
            paridad_apalancamiento_max=50.0,
            risk_vol_ewma=np.array([0.001, 0.0, 0.0], dtype=np.float64),
            exit_sl_ewma_mult=2.0,
            exit_tp_ewma_mult=4.0,
        )

        result = simular_full(arrays, np.array([1, 0, 0], dtype=np.int8), sim_cfg=cfg)
        trades = result.take_trades()

        self.assertEqual(MOTIVOS[int(trades["motivo_salida"][0])], "SL")
        self.assertAlmostEqual(float(trades["apalancamiento"][0]), 25.0)
        self.assertAlmostEqual(float(trades["risk_vol_ewma"][0]), 0.001)
        self.assertAlmostEqual(float(trades["risk_sl_dist_pct"][0]), 0.002)
        self.assertAlmostEqual(float(trades["precio_salida"][0]), 99.8)
        self.assertAlmostEqual(float(trades["roi"][0]), -0.05)


if __name__ == "__main__":
    unittest.main()

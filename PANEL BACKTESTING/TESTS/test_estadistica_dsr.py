"""Tests del Deflated Sharpe Ratio, MinBTL y utilidades (Fase 3).

Pura stdlib (math, statistics): no requieren Polars ni el motor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from COMUN import estadistica as est  # noqa: E402


class TestDSR(unittest.TestCase):
    def test_phi_inv_inversa_de_phi(self):
        for p in (0.1, 0.5, 0.9, 0.975):
            self.assertAlmostEqual(est.phi(est.phi_inv(p)), p, places=6)

    def test_sin_seleccion_dsr_igual_psr(self):
        # Con N<=1 no hay testing múltiple: SR_0 = 0 y el DSR colapsa al PSR.
        psr = est.probabilistic_sharpe_ratio(0.15, 200)
        dsr = est.deflated_sharpe_ratio(
            0.15, 200, n_configuraciones=1, varianza_sharpe_trials=0.01
        )
        self.assertAlmostEqual(psr, dsr, places=12)

    def test_dsr_menor_que_psr_con_muchos_trials(self):
        # Al deflactar contra muchas pruebas, el DSR debe ser <= PSR.
        psr = est.probabilistic_sharpe_ratio(0.15, 200)
        dsr = est.deflated_sharpe_ratio(
            0.15, 200, n_configuraciones=1000, varianza_sharpe_trials=0.01
        )
        self.assertLess(dsr, psr)
        self.assertGreaterEqual(dsr, 0.0)

    def test_maximo_esperado_crece_con_N(self):
        e10 = est.maximo_sharpe_esperado_estandarizado(10)
        e100 = est.maximo_sharpe_esperado_estandarizado(100)
        e1000 = est.maximo_sharpe_esperado_estandarizado(1000)
        self.assertLess(e10, e100)
        self.assertLess(e100, e1000)
        self.assertEqual(est.maximo_sharpe_esperado_estandarizado(1), 0.0)

    def test_sharpe_referencia_escala_con_varianza(self):
        sr_baja = est.sharpe_referencia_deflactado(0.01, 500)
        sr_alta = est.sharpe_referencia_deflactado(0.04, 500)
        self.assertAlmostEqual(sr_alta / sr_baja, 2.0, places=6)  # sqrt(0.04/0.01)=2

    def test_minbtl_crece_con_N_y_baja_con_sharpe(self):
        m_pocos = est.minimum_backtest_length(10, 1.0)
        m_muchos = est.minimum_backtest_length(10000, 1.0)
        self.assertLess(m_pocos, m_muchos)
        m_sr_bajo = est.minimum_backtest_length(1000, 0.5)
        m_sr_alto = est.minimum_backtest_length(1000, 2.0)
        self.assertGreater(m_sr_bajo, m_sr_alto)
        self.assertEqual(est.minimum_backtest_length(1000, 0.0), float("inf"))

    def test_dsr_en_rango(self):
        for n_cfg in (1, 10, 100, 1000):
            dsr = est.deflated_sharpe_ratio(
                0.2, 150, n_configuraciones=n_cfg, varianza_sharpe_trials=0.02
            )
            self.assertGreaterEqual(dsr, 0.0)
            self.assertLessEqual(dsr, 1.0)


if __name__ == "__main__":
    unittest.main()

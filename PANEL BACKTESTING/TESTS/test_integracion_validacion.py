"""Tests de la integración de validación OOS: métricas por ventana y la capa de
composición (con callbacks de prueba). NumPy + stdlib; no requieren el motor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from VALIDACION import integracion
from VALIDACION.metricas_subconjunto import (
    metricas_en_indices,
    metricas_subconjunto,
    retornos_por_trade,
)


def _trades():
    # 10 trades, entradas en índices 0..900, pnl alternando signo.
    idx = np.arange(0, 1000, 100, dtype=np.int64)
    pnl = np.array([100, -50, 80, -30, 120, -40, 90, -20, 110, -10], dtype=np.float64)
    return {"idx_entrada": idx, "pnl": pnl}


class TestMetricasSubconjunto(unittest.TestCase):
    def test_ventana_rango(self):
        m = metricas_subconjunto(_trades(), 0, 500, saldo_inicial=10_000, saldo_por_trade=500)
        self.assertEqual(m["total_trades"], 5)  # idx 0,100,200,300,400
        self.assertAlmostEqual(m["pnl_total"], 100 - 50 + 80 - 30 + 120)

    def test_ventana_vacia(self):
        m = metricas_subconjunto(_trades(), 10_000, 20_000, saldo_inicial=10_000, saldo_por_trade=500)
        self.assertEqual(m["total_trades"], 0)
        self.assertEqual(m["saldo_final"], 10_000)

    def test_membership(self):
        indices = np.array([0, 100, 200])  # 3 trades
        m = metricas_en_indices(_trades(), indices, saldo_inicial=10_000, saldo_por_trade=500)
        self.assertEqual(m["total_trades"], 3)
        self.assertAlmostEqual(m["pnl_total"], 100 - 50 + 80)

    def test_max_drawdown_en_rango(self):
        m = metricas_subconjunto(_trades(), 0, 1000, saldo_inicial=10_000, saldo_por_trade=500)
        self.assertGreaterEqual(m["max_drawdown"], 0.0)
        self.assertLessEqual(m["max_drawdown"], 1.0)

    def test_retornos_por_trade(self):
        r = retornos_por_trade(_trades(), saldo_por_trade=500)
        self.assertEqual(r.shape[0], 10)
        self.assertAlmostEqual(r[0], 100 / 500)


class TestComposicion(unittest.TestCase):
    def test_validacion_completa_con_fakes(self):
        # Callbacks de prueba: la "config" es un escalar de calidad; evaluar
        # devuelve un sharpe estable y positivo proporcional a esa calidad.
        def optimizar(indices_train):
            return {"calidad": 1.0 + indices_train.size / 10_000.0}

        def evaluar(config, indices):
            return {"sharpe_ratio": 0.12, "total_trades": int(indices.size // 5)}

        retornos = np.random.default_rng(0).normal(0.01, 0.05, size=200)
        res = integracion.ejecutar_validacion_completa(
            n_obs=1200,
            optimizar=optimizar,
            evaluar=evaluar,
            sharpe_hat=0.15,
            n_trades=150,
            n_configuraciones=500,
            varianza_sharpe_trials=0.02,
            retornos_mejor=retornos,
            capital_inicial=10_000.0,
            bootstrap_iter=1000,
            cabecera={"activo": "BTC"},
        )
        self.assertIn(res.veredicto.color, ("verde", "ambar", "rojo"))
        self.assertEqual(len(res.cpcv.valores_oos), 15)  # C(6,2)
        self.assertIsNotNone(res.wfa)
        self.assertGreaterEqual(res.dsr, 0.0)
        self.assertLessEqual(res.dsr, 1.0)
        self.assertIsNotNone(res.bootstrap)
        # El dict del informe trae las secciones esperadas y en orden lógico.
        d = res.datos_informe
        self.assertIn("oos", d)
        self.assertIn("veredicto", d)
        self.assertIn("robustez", d)

    def test_pbo_opcional(self):
        def optimizar(idx):
            return {}

        def evaluar(config, idx):
            return {"sharpe_ratio": 1.0}

        rng = np.random.default_rng(1)
        matriz = rng.standard_normal((600, 20))
        res = integracion.ejecutar_validacion_completa(
            n_obs=600, optimizar=optimizar, evaluar=evaluar,
            sharpe_hat=0.2, n_trades=120, n_configuraciones=300, varianza_sharpe_trials=0.01,
            matriz_pbo=matriz, pbo_s=10, capital_inicial=10_000.0,
        )
        self.assertIsNotNone(res.pbo)
        self.assertGreaterEqual(res.pbo, 0.0)
        self.assertLessEqual(res.pbo, 1.0)


if __name__ == "__main__":
    unittest.main()

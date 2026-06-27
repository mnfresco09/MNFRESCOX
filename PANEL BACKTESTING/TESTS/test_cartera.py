"""Tests de la Fase 6 (CARTERA: sizing + cartera). NumPy + stdlib."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from CARTERA import cartera, sizing  # noqa: E402


class TestSizing(unittest.TestCase):
    def test_vol_ewma_positiva(self):
        rng = np.random.default_rng(0)
        v = sizing.vol_ewma(rng.normal(0, 0.02, 500), lambda_=0.94)
        self.assertGreater(v, 0.0)

    def test_volatility_target(self):
        # vol objetivo 10%, realizada 20% → tamaño = 0.5 · capital
        t = sizing.volatility_target_size(10_000, 0.10, 0.20)
        self.assertAlmostEqual(t, 5_000.0)
        # vol realizada 0 → 0 (no se puede dimensionar)
        self.assertEqual(sizing.volatility_target_size(10_000, 0.10, 0.0), 0.0)
        # tope de apalancamiento
        t2 = sizing.volatility_target_size(10_000, 0.40, 0.10, apalancamiento_max=2.0)
        self.assertAlmostEqual(t2, 20_000.0)

    def test_kelly_fraccional(self):
        # f* = media/var = 0.1/0.04 = 2.5; con fraccion 0.5 se acota a 0.5.
        f = sizing.kelly_fraccional(0.1, 0.04, fraccion=0.5)
        self.assertAlmostEqual(f, 0.5)
        self.assertEqual(sizing.kelly_fraccional(0.1, 0.0), 0.0)


class TestCartera(unittest.TestCase):
    def test_correlacion_shape(self):
        rng = np.random.default_rng(1)
        M = rng.normal(0, 1, (200, 3))
        C = cartera.matriz_correlacion(M)
        self.assertEqual(C.shape, (3, 3))
        self.assertAlmostEqual(C[0, 0], 1.0, places=6)

    def test_descorrelacionadas_aportan(self):
        rng = np.random.default_rng(2)
        # Tres estrategias mediocres pero independientes con media positiva.
        M = rng.normal(0.05, 1.0, (1000, 3))
        contribs = cartera.contribucion_marginal_sharpe(M)
        self.assertEqual(len(contribs), 3)
        # Invariante válido en muestra finita: diversificar entre estrategias
        # independientes con media positiva da un Sharpe de cartera positivo y
        # no peor que el de la estrategia individual más floja.
        sharpe_port = cartera.sharpe_cartera(M)
        self.assertGreater(sharpe_port, 0.0)
        self.assertGreaterEqual(sharpe_port, min(c.sharpe_individual for c in contribs) - 1e-9)

    def test_redundante_correlacionada(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0.05, 1.0, 1000)
        # Dos columnas idénticas (correlación 1): una es redundante.
        M = np.column_stack([base, base.copy()])
        contribs = cartera.contribucion_marginal_sharpe(M)
        # Quitar una columna idéntica no cambia el Sharpe → contribución ~0.
        self.assertAlmostEqual(contribs[0].contribucion_marginal, 0.0, places=6)
        self.assertTrue(contribs[0].redundante)


if __name__ == "__main__":
    unittest.main()

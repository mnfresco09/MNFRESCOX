"""Tests del PBO (CSCV) — COMUN/sobreajuste.py (Fase 3).

Requiere NumPy (disponible). No requiere Polars ni el motor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from COMUN.sobreajuste import pbo_cscv  # noqa: E402


class TestPBO(unittest.TestCase):
    def test_ruido_puro_pbo_cercano_a_medio(self):
        # N configuraciones de puro ruido i.i.d.: la mejor IS no tiene por qué
        # serlo OOS, así que el PBO debe rondar 0.5.
        rng = np.random.default_rng(7)
        M = rng.standard_normal((2000, 50))
        res = pbo_cscv(M, s=10)
        # Ruido puro: la mejor IS no persiste OOS, el PBO se aleja de los
        # extremos. El umbral exacto fluctúa con la muestra; lo robusto es que
        # quede claramente por encima del de una estrategia con edge real.
        self.assertGreater(res.pbo, 0.2)
        self.assertLess(res.pbo, 0.8)
        self.assertEqual(res.n_configuraciones, 50)

    def test_estrategia_genuina_pbo_bajo(self):
        # Una columna con media claramente positiva (edge real) domina IS y OOS,
        # así que el PBO debe ser bajo.
        rng = np.random.default_rng(11)
        M = rng.standard_normal((2000, 30)) * 0.5
        M[:, 0] += 0.5  # edge persistente en la columna 0
        res = pbo_cscv(M, s=10)
        self.assertLess(res.pbo, 0.2)
        self.assertFalse(res.es_sobreajuste)

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            pbo_cscv(np.zeros((100, 1)), s=8)        # < 2 configuraciones
        with self.assertRaises(ValueError):
            pbo_cscv(np.zeros((100, 5)), s=7)        # s impar
        with self.assertRaises(ValueError):
            pbo_cscv(np.zeros((4, 5)), s=8)          # menos filas que bloques
        with self.assertRaises(ValueError):
            pbo_cscv(np.zeros((10,)), s=4)           # no 2D

    def test_numero_de_combinaciones(self):
        from math import comb

        rng = np.random.default_rng(3)
        M = rng.standard_normal((500, 12))
        res = pbo_cscv(M, s=8)
        self.assertEqual(res.n_combinaciones, comb(8, 4))


if __name__ == "__main__":
    unittest.main()

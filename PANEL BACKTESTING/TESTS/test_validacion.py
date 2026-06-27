"""Tests de la Fase 2 (CPCV + WFA + distribución + orquestador) y del veredicto
de la Fase 3/4. NumPy + stdlib; no requieren Polars ni el motor.
"""

from __future__ import annotations

import sys
import unittest
from math import comb
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from VALIDACION import cpcv, distribucion, orquestador, veredicto, wfa  # noqa: E402


class TestCPCV(unittest.TestCase):
    def test_trayectorias_y_combinaciones(self):
        self.assertEqual(cpcv.n_trayectorias(6, 2), 5)  # φ = 15·2/6
        folds = cpcv.generar_folds(600, n_grupos=6, k=2, embargo=0.01)
        self.assertEqual(len(folds), comb(6, 2))

    def test_purge_sin_solape_train_test(self):
        folds = cpcv.generar_folds(600, n_grupos=6, k=2, embargo=0.02, duracion_trade=3)
        for fold in folds:
            test_idx = set(cpcv.indices_de_rangos(fold.test_rangos).tolist())
            train_idx = set(fold.train_idx.tolist())
            self.assertEqual(train_idx & test_idx, set())  # purge: sin solape

    def test_embargo_recorta_train(self):
        sin = cpcv.generar_folds(600, n_grupos=6, k=2, embargo=0.0)
        con = cpcv.generar_folds(600, n_grupos=6, k=2, embargo=0.05)
        # Con embargo, el train tiene <= índices que sin embargo.
        self.assertLessEqual(con[0].train_idx.size, sin[0].train_idx.size)

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            cpcv.generar_folds(600, n_grupos=6, k=6)   # k >= n_grupos
        with self.assertRaises(ValueError):
            cpcv.generar_folds(600, n_grupos=1, k=1)   # n_grupos < 2


class TestWFA(unittest.TestCase):
    def test_ventanas_rolling_y_anchored(self):
        rolling = wfa.generar_ventanas(1000, n_ventanas=4, fraccion_test=0.15, anchored=False)
        anchored = wfa.generar_ventanas(1000, n_ventanas=4, fraccion_test=0.15, anchored=True)
        self.assertEqual(len(rolling), 4)
        # Anchored: el train siempre arranca en 0 y crece.
        self.assertTrue(all(v.train_idx[0] == 0 for v in anchored))
        self.assertLess(anchored[0].train_idx.size, anchored[-1].train_idx.size)

    def test_efficiency(self):
        self.assertAlmostEqual(wfa.wfa_efficiency(1.0, 2.0), 0.5)
        self.assertEqual(wfa.wfa_efficiency(1.0, 0.0), 0.0)  # IS no positivo


class TestDistribucion(unittest.TestCase):
    def test_resumen(self):
        d = distribucion.resumir([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(d.n, 4)
        self.assertAlmostEqual(d.mediana, 2.5)
        self.assertAlmostEqual(d.fraccion_positiva, 1.0)

    def test_vacio_lanza(self):
        with self.assertRaises(ValueError):
            distribucion.resumir([float("nan")])


class TestOrquestador(unittest.TestCase):
    def test_ejecutar_cpcv_con_callbacks(self):
        # Estrategia "real": el retorno OOS es positivo y estable.
        def optimizar(train_idx):
            return {"umbral": float(train_idx.size)}

        def evaluar(params, idx):
            return {"sharpe_ratio": 1.2}

        res = orquestador.ejecutar_cpcv(
            600, optimizar=optimizar, evaluar=evaluar, n_grupos=6, k=2
        )
        self.assertEqual(len(res.valores_oos), comb(6, 2))
        self.assertAlmostEqual(res.distribucion_oos.media, 1.2)
        self.assertAlmostEqual(res.ratio_oos_is, 1.0)

    def test_ejecutar_wfa_con_callbacks(self):
        def optimizar(train_idx):
            return {}

        def evaluar(params, idx):
            # OOS degrada a la mitad del IS según el tamaño del tramo.
            return {"sharpe_ratio": 2.0 if idx.size > 100 else 1.0}

        res = orquestador.ejecutar_wfa(1000, optimizar=optimizar, evaluar=evaluar, n_ventanas=3)
        self.assertGreaterEqual(res.efficiency, 0.0)


class TestVeredicto(unittest.TestCase):
    def test_verde_total(self):
        v = veredicto.evaluar_veredicto(
            dsr=0.97, pbo=0.10, ratio_oos_is=0.8, p25_sharpe_oos=0.2,
            n_trades=150, wfa_efficiency=0.7,
        )
        self.assertEqual(v.color, veredicto.VERDE)
        self.assertTrue(v.aprobada)

    def test_un_rojo_mata(self):
        v = veredicto.evaluar_veredicto(dsr=0.97, pbo=0.60, n_trades=150)  # PBO rojo
        self.assertEqual(v.color, veredicto.ROJO)
        self.assertIn("PBO", " ".join(v.motivos(veredicto.ROJO)))

    def test_ambar(self):
        v = veredicto.evaluar_veredicto(dsr=0.92, pbo=0.10, n_trades=150)  # DSR ámbar
        self.assertEqual(v.color, veredicto.AMBAR)

    def test_distribucion_oos(self):
        v = veredicto.evaluar_veredicto(mediana_sharpe_oos=0.3, p25_sharpe_oos=-0.1)
        # mediana>0 pero p25<0 → ámbar
        self.assertEqual(v.color, veredicto.AMBAR)

    def test_holdout(self):
        v = veredicto.evaluar_veredicto(holdout_metrica=-0.1)  # colapsa
        self.assertEqual(v.color, veredicto.ROJO)


if __name__ == "__main__":
    unittest.main()

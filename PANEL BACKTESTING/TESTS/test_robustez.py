"""Tests de la Fase 5 (ROBUSTEZ) y de los helpers de objetivo robusto (Fase 4).
NumPy + stdlib.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from ROBUSTEZ import bootstrap, nula, regimen, sensibilidad  # noqa: E402


def _cargar_modulo(ruta_rel: str, nombre: str):
    """Carga un módulo por ruta evitando ejecutar el __init__ de su paquete
    (necesario para OPTIMIZACION, cuyo __init__ importa Polars/motor)."""
    ruta = RAIZ / ruta_rel
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


robustez_objetivo = _cargar_modulo("OPTIMIZACION/robustez_objetivo.py", "robustez_objetivo")


class TestBootstrap(unittest.TestCase):
    def test_distribuciones(self):
        rng = np.random.default_rng(0)
        retornos = rng.normal(5.0, 50.0, size=300)  # PnL por trade
        res = bootstrap.bootstrap_trades(retornos, n_iter=2000, saldo_inicial=10_000, seed=1)
        self.assertEqual(res.equity_final.shape[0], 2000)
        # max drawdown como fracción en [0, 1]
        self.assertTrue(np.all(res.max_drawdown >= 0.0))
        self.assertTrue(np.all(res.max_drawdown <= 1.0))
        p = res.percentiles_equity()
        self.assertLessEqual(p[5], p[50])
        self.assertLessEqual(p[50], p[95])

    def test_block_bootstrap(self):
        rng = np.random.default_rng(2)
        retornos = rng.normal(0.0, 0.01, size=200)
        res = bootstrap.bootstrap_trades(
            retornos, n_iter=500, tam_bloque=5, compuesto=True, seed=3
        )
        self.assertEqual(res.n_trades, 200)
        self.assertEqual(res.sharpe.shape[0], 500)

    def test_validaciones(self):
        with self.assertRaises(ValueError):
            bootstrap.bootstrap_trades([], n_iter=10)


class TestRegimen(unittest.TestCase):
    def test_agrupa(self):
        retornos = [1.0, -2.0, 3.0, -1.0]
        etiquetas = ["alcista", "bajista", "alcista", "lateral"]
        res = regimen.rendimiento_por_regimen(retornos, etiquetas)
        self.assertEqual(res["alcista"].n_trades, 2)
        self.assertAlmostEqual(res["alcista"].retorno_total, 4.0)

    def test_longitud_incoherente(self):
        with self.assertRaises(ValueError):
            regimen.rendimiento_por_regimen([1, 2, 3], ["a", "b"])

    def test_etiquetas_macd(self):
        # Tendencia alcista clara → predomina 'alcista'; longitud preservada.
        close = np.linspace(100, 200, 300)
        etq = regimen.etiquetas_macd(close)
        self.assertEqual(etq.shape[0], close.shape[0])
        self.assertIn("alcista", set(etq.tolist()))


class TestNula(unittest.TestCase):
    def test_barajar_conserva_multiset(self):
        r = [1.0, 2.0, 3.0, 4.0]
        b = nula.barajar(r, seed=5)
        self.assertEqual(sorted(b.tolist()), r)

    def test_contraste(self):
        dist = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        c = nula.contrastar(0.5, dist, nivel=0.95)
        self.assertTrue(c.supera)
        self.assertEqual(c.p_valor, 0.0)
        c2 = nula.contrastar(0.0, dist)
        self.assertFalse(c2.supera)

    def test_distribucion_nula(self):
        d = nula.distribucion_nula(100, lambda rng: rng.normal(), seed=1)
        self.assertEqual(d.shape[0], 100)


class TestSensibilidad(unittest.TestCase):
    def test_estable(self):
        s = sensibilidad.sensibilidad_fecha_inicio([0, 5, 10], lambda o: 1.0, umbral_cv=0.25)
        self.assertFalse(s.fragil)
        self.assertAlmostEqual(s.media, 1.0)

    def test_fragil(self):
        s = sensibilidad.sensibilidad_fecha_inicio([0, 1, 2], lambda o: float(o), umbral_cv=0.25)
        self.assertTrue(s.fragil)


class TestObjetivoRobusto(unittest.TestCase):
    def test_vector_pareto(self):
        v = robustez_objetivo.vector_pareto(
            {"psr": 0.9, "max_drawdown": -0.12, "trades_por_dia": 3.0}
        )
        self.assertEqual(v, (0.9, 0.12, 3.0))  # drawdown en valor absoluto
        self.assertEqual(len(robustez_objetivo.DIRECCIONES_PARETO), 3)

    def test_seleccionar_meseta_prefiere_region_estable(self):
        from types import SimpleNamespace

        def t(p, psr):
            return SimpleNamespace(parametros={"x": p}, metricas={"psr": psr}, score=psr)

        # Pico FRÁGIL en x=100 (psr altísimo pero rodeado de configuraciones
        # malas) vs MESETA estable en x≈10 (vecindario uniformemente bueno).
        universo = [
            t(100, 0.99), t(98, 0.40), t(102, 0.41),     # pico solitario y frágil
            t(9, 0.84), t(10, 0.85), t(11, 0.86), t(12, 0.85),  # meseta estable
        ]
        candidatos = [universo[0], universo[4]]  # pico (x=100) vs meseta (x=10)
        elegido = robustez_objetivo.seleccionar_meseta(
            candidatos, universo,
            valor=lambda x: x.metricas["psr"],
            parametros=lambda x: x.parametros,
            k=3,
        )
        self.assertEqual(elegido.parametros["x"], 10)  # gana la meseta, no el pico

    def test_seleccionar_meseta_un_candidato(self):
        from types import SimpleNamespace

        c = SimpleNamespace(parametros={"x": 1}, metricas={"psr": 0.5}, score=0.5)
        self.assertIs(
            robustez_objetivo.seleccionar_meseta(
                [c], [c], valor=lambda x: 0.5, parametros=lambda x: x.parametros
            ),
            c,
        )

    def test_penalizaciones(self):
        s = robustez_objetivo.penalizacion_turnover(1.0, 200, trades_objetivo=100, factor=0.5)
        self.assertAlmostEqual(s, 1.0 - 0.5 * (100 / 100))
        self.assertEqual(robustez_objetivo.penalizacion_turnover(1.0, 50, factor=0.0), 1.0)
        self.assertAlmostEqual(robustez_objetivo.penalizacion_complejidad(1.0, 4, factor=0.1), 0.6)

if __name__ == "__main__":
    unittest.main()

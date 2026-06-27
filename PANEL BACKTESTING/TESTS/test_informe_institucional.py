"""Tests del informe institucional unificado (Fase 7).

Se carga por ruta para evitar el __init__ de REPORTES (que importa Polars).
Verifica que las secciones aparecen en el ORDEN narrativo correcto y que el
holdout queda marcado.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def _cargar(ruta_rel: str, nombre: str):
    ruta = RAIZ / ruta_rel
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


informe = _cargar("REPORTES/informe_institucional.py", "informe_institucional")


def _datos():
    return {
        "cabecera": {"estrategia": "RSI", "activo": "BTC", "timeframe": "1h", "modo": "investigacion"},
        "preregistro": {"ineficiencia": "reversión a la media intradía"},
        "oos": {
            "distribucion": {"media": 1.3, "desviacion": 0.6, "p25": 0.7, "mediana": 1.2, "n": 5},
            "ratio_oos_is": 0.72,
            "wfa_efficiency": 0.65,
        },
        "veredicto": {
            "color": "verde",
            "dsr": 0.96,
            "pbo": 0.12,
            "minbtl": 1.8,
            "criterios": [{"nombre": "DSR", "valor": 0.96, "color": "verde", "detalle": "0.96 ≥ 0.95"}],
        },
        "robustez": {"bootstrap": {"p5_equity": 10500.0}, "nula": "SUPERA la nula"},
        "is": {"mejor_score": 0.9},
        "equity": {"valores": [10000, 10100, 10250, 10200, 10400], "indice_holdout": 3},
    }


class TestInforme(unittest.TestCase):
    def test_genera_html_con_secciones_en_orden(self):
        h = informe.generar_informe_institucional(_datos())
        self.assertIn("<!DOCTYPE html>", h)
        # Las secciones aparecen en el orden narrativo correcto.
        posiciones = [h.index(f"id='{anchor}'") for anchor in informe.ORDEN_SECCIONES if f"id='{anchor}'" in h]
        self.assertEqual(posiciones, sorted(posiciones))
        # OOS aparece ANTES que la optimización IS (titular correcto).
        self.assertLess(h.index("resultados-oos"), h.index("optimizacion-is"))

    def test_marca_holdout(self):
        h = informe.generar_informe_institucional(_datos())
        self.assertIn("HOLDOUT BLOQUEADO", h)
        self.assertIn("plotly", h.lower())

    def test_veredicto_color(self):
        h = informe.generar_informe_institucional(_datos())
        self.assertIn("VERDE", h)

    def test_escribe_fichero(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "informe.html"
            informe.generar_informe_institucional(_datos(), ruta_salida=ruta)
            self.assertTrue(ruta.exists())
            self.assertIn("<!DOCTYPE html>", ruta.read_text(encoding="utf-8"))

    def test_secciones_opcionales(self):
        # Con datos mínimos no debe romper; solo renderiza lo presente.
        h = informe.generar_informe_institucional({"cabecera": {"activo": "BTC"}})
        self.assertIn("<!DOCTYPE html>", h)


if __name__ == "__main__":
    unittest.main()

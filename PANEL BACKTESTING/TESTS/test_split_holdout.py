"""Tests del split temporal de tres bloques y la exclusión física del holdout
(Fase 0) — DATOS/cargador.py.

Se omite automáticamente si Polars no está instalado (el módulo lo importa).
La lógica de fechas es pura y se valida sin tocar ningún fichero de datos.
"""

from __future__ import annotations

import sys
import types
import unittest
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

try:
    import polars  # noqa: F401
    from DATOS import cargador
    _POLARS = True
except Exception:  # pragma: no cover - entorno sin polars
    _POLARS = False


def _cfg(modo="investigacion"):
    return types.SimpleNamespace(
        FECHA_INICIO="2021-01-01",
        FECHA_FIN="2024-12-31",
        HOLDOUT_INICIO="2024-01-01",
        MODO=modo,
    )


@unittest.skipUnless(_POLARS, "Polars no disponible en este entorno")
class TestSplit(unittest.TestCase):
    def test_limites_split(self):
        inicio, holdout, fin_excl = cargador.limites_split(_cfg())
        self.assertEqual(inicio, date(2021, 1, 1))
        self.assertEqual(holdout, date(2024, 1, 1))
        self.assertEqual(fin_excl, date(2025, 1, 1))  # FECHA_FIN + 1 día

    def test_investigacion_excluye_holdout(self):
        # En modo investigación, "auto" devuelve el tramo TRAIN/VALIDATION:
        # [inicio, holdout). El holdout queda físicamente fuera.
        inicio, fin_excl = cargador._rango_para_tramo(_cfg("investigacion"), "auto")
        self.assertEqual(inicio, date(2021, 1, 1))
        self.assertEqual(fin_excl, date(2024, 1, 1))

    def test_veredicto_final_incluye_todo(self):
        inicio, fin_excl = cargador._rango_para_tramo(_cfg("veredicto_final"), "auto")
        self.assertEqual(inicio, date(2021, 1, 1))
        self.assertEqual(fin_excl, date(2025, 1, 1))

    def test_tramo_holdout_explicito(self):
        inicio, fin_excl = cargador._rango_para_tramo(_cfg(), "holdout")
        self.assertEqual(inicio, date(2024, 1, 1))
        self.assertEqual(fin_excl, date(2025, 1, 1))

    def test_holdout_invalido_lanza(self):
        cfg = _cfg()
        cfg.HOLDOUT_INICIO = "2025-06-01"  # posterior a FECHA_FIN
        with self.assertRaises(ValueError):
            cargador.limites_split(cfg)


if __name__ == "__main__":
    unittest.main()

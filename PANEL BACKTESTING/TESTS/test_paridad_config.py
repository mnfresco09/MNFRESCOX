from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CONFIGURACION import config  # noqa: E402
from CONFIGURACION.validador_config import _validar_paridad_riesgo  # noqa: E402
from NUCLEO import paridad_riesgo  # noqa: E402
from SALIDAS import paridad  # noqa: E402


class ParidadConfigTest(unittest.TestCase):
    def test_paridad_riesgo_lee_defaults_desde_salidas_paridad(self) -> None:
        params = paridad_riesgo.ParametrosParidadRiesgo(activa=True)

        self.assertEqual(params.riesgo_max_pct, paridad.RIESGO_MAXIMO_PCT)
        self.assertEqual(params.vol_halflife, paridad.VOL_HALFLIFE)
        self.assertEqual(params.sl_ewma_mult, paridad.SL_EWMA_MULT)
        self.assertEqual(params.tp_ewma_mult, paridad.TP_EWMA_MULT)
        self.assertEqual(params.trail_act_ewma_mult, paridad.TRAIL_ACT_EWMA_MULT)
        self.assertEqual(params.trail_dist_ewma_mult, paridad.TRAIL_DIST_EWMA_MULT)

    def test_validador_paridad_usa_limites_de_salidas_no_de_config(self) -> None:
        cfg = SimpleNamespace(
            USAR_PARIDAD_RIESGO=True,
            OPTIMIZAR_PARIDAD_RIESGO=True,
        )
        errores: list[str] = []

        _validar_paridad_riesgo(cfg, errores)

        self.assertEqual(errores, [])

    def test_config_general_no_expone_parametros_operativos_de_paridad(self) -> None:
        self.assertFalse(hasattr(config, "PARIDAD_APALANCAMIENTO_MIN"))
        self.assertFalse(hasattr(config, "PARIDAD_APALANCAMIENTO_MAX"))
        self.assertTrue(hasattr(config, "USAR_PARIDAD_RIESGO"))
        self.assertTrue(hasattr(config, "OPTIMIZAR_PARIDAD_RIESGO"))


if __name__ == "__main__":
    unittest.main()

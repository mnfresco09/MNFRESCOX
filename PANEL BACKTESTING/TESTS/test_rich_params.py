from __future__ import annotations

import sys
import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from REPORTES.rich import _panel_parametros  # noqa: E402


class RichParametrosTest(unittest.TestCase):
    def test_paridad_mueve_risk_al_bloque_exit_y_oculta_porcentajes_legacy(self) -> None:
        params = {
            "__exit_type": "TRAILING",
            "__exit_sl_pct": 25.0,
            "__exit_trail_act_pct": 30.0,
            "__exit_trail_dist_pct": 6.0,
            "__paridad_riesgo": True,
            "risk_max_pct": 4.5,
            "risk_sl_ewma_mult": 1.5,
            "risk_trail_act_ewma_mult": 10.3,
            "risk_trail_dist_ewma_mult": 5.55,
            "risk_vol_halflife": 235,
            "halflife_bars": 25,
            "normalization_multiplier": 3.5,
            "umbral_cvd": 1.5,
        }

        text = _render_text(_panel_parametros(params, "TRAILING"))

        self.assertIn("PARIDAD", text)
        self.assertIn("RIESGO", text)
        self.assertIn("SL EWMA", text)
        self.assertIn("TRAIL ACT", text)
        self.assertIn("TRAIL DIST", text)
        self.assertIn("VOL HL", text)
        self.assertIn("HALFLIFE BARS", text)
        self.assertNotIn("25.0%", text)
        self.assertNotIn("30.0%", text)
        self.assertNotIn("6.0%", text)
        self.assertNotIn("RISK MAX PCT", text)


def _render_text(renderable) -> str:
    console = Console(record=True, width=100, color_system=None, file=StringIO())
    console.print(renderable)
    return console.export_text()


if __name__ == "__main__":
    unittest.main()

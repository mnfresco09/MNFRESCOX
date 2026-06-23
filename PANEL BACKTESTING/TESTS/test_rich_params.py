from __future__ import annotations

import sys
import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from REPORTES.rich import _panel_parametros  # noqa: E402


class RichParametrosTest(unittest.TestCase):
    def test_trailing_muestra_sl_activa_dist_y_params_estrategia(self) -> None:
        params = {
            "__exit_type": "TRAILING",
            "__exit_sl_pct": 25.0,
            "__exit_trail_act_pct": 30.0,
            "__exit_trail_dist_pct": 6.0,
            "halflife_bars": 25,
            "normalization_multiplier": 3.5,
            "umbral_cvd": 1.5,
        }

        text = _render_text(_panel_parametros(params, "TRAILING"))

        self.assertIn("TRAILING", text)
        self.assertIn("SL", text)
        self.assertIn("ACTIVA", text)
        self.assertIn("DIST", text)
        self.assertNotIn("PARIDAD", text)
        self.assertNotIn("EWMA", text)


def _render_text(renderable) -> str:
    console = Console(record=True, width=100, color_system=None, file=StringIO())
    console.print(renderable)
    return console.export_text()


if __name__ == "__main__":
    unittest.main()

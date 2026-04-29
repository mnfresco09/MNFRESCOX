from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from REPORTES.informe import _crear_payload, _render_html  # noqa: E402


class InformeRobustezTest(unittest.TestCase):
    def test_payload_incluye_labels_profesionales_y_metricas_derivadas(self) -> None:
        payload = _payload_minimo()

        self.assertEqual(payload["field_labels"]["risk_sl_ewma_mult"], "SL xVOL")
        self.assertEqual(payload["field_labels"]["risk_trail_act_ewma_mult"], "TRAIL ACT xVOL")
        self.assertEqual(payload["field_labels"]["risk_vol_halflife"], "VOL HL")
        self.assertIn("return_dd_ratio", payload["derived_keys"])
        self.assertIn("pnl_por_trade", payload["derived_keys"])
        self.assertAlmostEqual(payload["trials"][0]["derived"]["return_dd_ratio"], 4.0)
        self.assertAlmostEqual(payload["trials"][0]["derived"]["pnl_por_trade"], 10.0)

    def test_template_terminal_tiene_vistas_avanzadas_y_rebuild_de_tab_limpio(self) -> None:
        html = _render_html(_payload_minimo())

        self.assertIn("MNFRESCOX TERMINAL", html)
        self.assertIn("ROBUSTEZ REPORT - V2", html)
        for view_id in (
            "ranking",
            "heatmap",
            "topzone",
            "scatter2d",
            "scatter3d",
            "pareto",
            "correlation",
            "convergence",
            "distribution",
            "sensitivity",
            "parallel",
        ):
            self.assertIn(f"id:'{view_id}'", html)
        self.assertIn("function switchView", html)
        self.assertIn("function renderCurrent", html)
        self.assertIn("buildControls();\n  readControls();", html)


def _payload_minimo() -> dict:
    trials = [
        SimpleNamespace(
            numero=1,
            score=10.0,
            parametros={
                "risk_sl_ewma_mult": 2.0,
                "risk_trail_act_ewma_mult": 4.0,
                "risk_trail_dist_ewma_mult": 1.0,
                "risk_vol_halflife": 50,
                "umbral_cvd": 1.5,
            },
            metricas={
                "roi_total": 0.20,
                "max_drawdown": 0.05,
                "pnl_total": 100.0,
                "total_trades": 10,
                "profit_factor": 1.4,
                "sharpe_ratio": 0.2,
            },
        ),
        SimpleNamespace(
            numero=2,
            score=6.0,
            parametros={
                "risk_sl_ewma_mult": 3.0,
                "risk_trail_act_ewma_mult": 5.0,
                "risk_trail_dist_ewma_mult": 1.5,
                "risk_vol_halflife": 80,
                "umbral_cvd": 1.0,
            },
            metricas={
                "roi_total": 0.10,
                "max_drawdown": 0.04,
                "pnl_total": 40.0,
                "total_trades": 8,
                "profit_factor": 1.2,
                "sharpe_ratio": 0.1,
            },
        ),
    ]
    return _crear_payload(
        trials=trials,
        estrategia=SimpleNamespace(NOMBRE="VWAP-CVD"),
        activo="BTC",
        timeframe="1H",
        salida_tipo="TRAILING",
    )


if __name__ == "__main__":
    unittest.main()

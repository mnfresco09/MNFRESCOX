from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from REPORTES.html import _render_html  # noqa: E402


class HtmlTooltipTest(unittest.TestCase):
    def test_tooltip_estandar_sin_rastro_de_riesgo(self) -> None:
        html = _render_html(_payload_minimo(), "")

        self.assertIn("function tooltipStandardRows", html)
        self.assertIn("PNL BRUTO", html)
        self.assertIn("PNL NETO", html)
        # La antigua gestión de riesgo (paridad) ya no deja rastro en el report.
        self.assertNotIn("riskMode", html)
        self.assertNotIn("tooltipRiskRows", html)
        self.assertNotIn("risk_tp_ewma_mult", html)

    def test_equity_usa_baseline_unica_sobre_saldo_inicial(self) -> None:
        payload = _payload_minimo()
        payload["equity_curve"] = [
            {"time": 1, "saldo": 1000.0, "equity_pct": 0.0},
            {"time": 2, "saldo": 1040.0, "equity_pct": 4.0},
            {"time": 3, "saldo": 960.0, "equity_pct": -4.0},
        ]

        html = _render_html(payload, "")

        self.assertIn("function addEquityPane", html)
        self.assertIn("const initialEquity=num(eqArr[0]?.saldo)", html)
        self.assertIn("baseValue:{type:'price',price:initialEquity}", html)
        self.assertIn("bottomLineColor:T.equityNegativeLine", html)
        self.assertIn("value:num(p.saldo)", html)
        self.assertNotIn("drawdownSeries", html)
        self.assertNotIn("EQUITY / DRAWDOWN", html)


def _payload_minimo() -> dict:
    return {
        "titulo": "BTC 1H | TEST | FIXED",
        "trial": 1,
        "score": 1.0,
        "timeframe_ejecucion": "1m",
        "salida": {"tipo": "FIXED", "sl_pct": 20, "tp_pct": 40, "velas": 0},
        "metricas": {},
        "parametros": {"halflife": 120, "umbral": 0.85},
        "conteo_senales": {},
        "rango": {"velas": 0},
        "resumen_trades": {},
        "analitica": {"secciones": [], "veredicto": {}},
        "distribuciones": {},
        "candles": [],
        "markers": [],
        "trades": [],
        "equity_curve": [],
        "indicadores": [],
    }


if __name__ == "__main__":
    unittest.main()

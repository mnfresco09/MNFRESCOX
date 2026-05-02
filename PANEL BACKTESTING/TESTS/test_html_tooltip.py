from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from REPORTES.html import _render_html  # noqa: E402


class HtmlTooltipTest(unittest.TestCase):
    def test_tooltip_tiene_modo_paridad_sin_pnl_bruto(self) -> None:
        html = _render_html(_payload_minimo_paridad(), "")

        self.assertIn("const riskMode", html)
        self.assertIn("function tooltipRiskRows", html)
        self.assertIn("function tooltipStandardRows", html)
        self.assertIn("TP %", html)
        self.assertIn("SL %", html)
        self.assertIn("APALANC.", html)

    def test_equity_usa_baseline_unica_sobre_saldo_inicial(self) -> None:
        payload = _payload_minimo_paridad()
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


def _payload_minimo_paridad() -> dict:
    return {
        "titulo": "BTC 1H | TEST | FIXED",
        "trial": 1,
        "score": 1.0,
        "timeframe_ejecucion": "1m",
        "salida": {"tipo": "FIXED", "sl_pct": 20, "tp_pct": 40, "velas": 0},
        "metricas": {},
        "parametros": {
            "risk_max_pct": 5.0,
            "risk_sl_ewma_mult": 2.0,
            "risk_tp_ewma_mult": 4.0,
        },
        "conteo_senales": {},
        "rango": {"velas": 0},
        "candles": [],
        "markers": [],
        "trades": [],
        "equity_curve": [],
        "indicadores": [],
    }


if __name__ == "__main__":
    unittest.main()

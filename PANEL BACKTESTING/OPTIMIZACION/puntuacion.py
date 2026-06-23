"""Función de score que Optuna maximiza.

Se elige con `FUNCION_SCORE` en config. Todas las opciones tienen base
estadística (definiciones en `OPTIMIZACION.estadistica`):

  - "PSR"    Probabilistic Sharpe Ratio. En [0, 1]: probabilidad de que el
             Sharpe real sea positivo dado el observado y el tamaño de muestra.
             Recomendado: premia estrategias creíbles y castiga las casualidades.
  - "SHARPE" Sharpe anualizado. Rentabilidad ajustada por volatilidad.
  - "CALMAR" CAGR / Max Drawdown. Rentabilidad anual frente al peor desplome.
  - "ROI"    Retorno total simple sobre el saldo inicial (sin ajuste de riesgo).

Reglas comunes:
  - 0 operaciones → score 0.
  - Por debajo de `MIN_TRADES_SCORE` operaciones → score 0 (muestra insuficiente).
  - Se devuelve el valor crudo de la métrica (no se reescala a 0..100): Optuna
    solo necesita ordenar, y conservar la escala real facilita interpretarlo.
"""

from __future__ import annotations

from math import isfinite

from CONFIGURACION import config as cfg

_FUNCIONES = {"PSR", "SHARPE", "CALMAR", "ROI"}


def calcular_score(metricas: dict) -> float:
    n = int(metricas.get("total_trades", 0))
    if n == 0:
        return 0.0
    if n < _min_trades():
        return 0.0

    modo = _modo()
    if modo == "PSR":
        valor = _num(metricas.get("psr"))
    elif modo == "SHARPE":
        valor = _num(metricas.get("sharpe_anualizado"))
    elif modo == "CALMAR":
        valor = _num(metricas.get("calmar"))
    else:  # ROI
        valor = _num(metricas.get("roi_total"))

    return round(valor, 6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _modo() -> str:
    modo = str(getattr(cfg, "FUNCION_SCORE", "PSR")).upper()
    return modo if modo in _FUNCIONES else "PSR"


def _min_trades() -> int:
    try:
        return max(0, int(getattr(cfg, "MIN_TRADES_SCORE", 0)))
    except (TypeError, ValueError):
        return 0


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if isfinite(number) else 0.0

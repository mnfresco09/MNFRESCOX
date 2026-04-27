from __future__ import annotations

from math import exp, isfinite, isinf, log


def calcular_score(metricas: dict) -> float:
    if int(metricas.get("total_trades", 0)) == 0:
        return 0.0

    f1 = _factor_expectancy(_num(metricas.get("expectancy")))
    f2 = _factor_profit_factor(_num(metricas.get("profit_factor"), allow_inf=True))
    f3 = _factor_drawdown(_num(metricas.get("max_drawdown")))
    f4 = _factor_trades_dia(_num(metricas.get("trades_por_dia")))

    score = f1 * f2 * f3 * f4 * 100.0
    return round(_clamp(score, 0.0, 100.0), 6)


def _factor_expectancy(expectancy: float) -> float:
    expectancy_pct = expectancy * 100.0
    if expectancy_pct < -10.0:
        return 0.0

    k = 0.4
    umbral = 3.0
    return 1.0 / (1.0 + exp(-k * (expectancy_pct - umbral)))


def _factor_profit_factor(profit_factor: float) -> float:
    if profit_factor <= 1.0:
        return 0.0

    pf_efectivo = 10.0 if isinf(profit_factor) else profit_factor
    return log(pf_efectivo) / log(pf_efectivo + 1.0)


def _factor_drawdown(max_drawdown: float) -> float:
    dd = max(0.0, max_drawdown)
    if dd <= 0.25:
        return 1.0

    exceso = (dd - 0.25) / 0.10
    return max(0.0, 1.0 - exceso**2)


def _factor_trades_dia(trades_por_dia: float) -> float:
    tpd = max(0.0, trades_por_dia)
    if tpd >= 0.2:
        return 1.0
    return tpd / 0.2


def _num(value, *, allow_inf: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if isfinite(number) or (allow_inf and isinf(number) and number > 0):
        return number
    return 0.0


def _clamp(value: float, min_value: float, max_value: float) -> float:
    if not isfinite(value):
        return min_value
    return max(min_value, min(max_value, value))

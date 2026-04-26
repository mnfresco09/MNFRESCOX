from __future__ import annotations

from math import sqrt


def calcular_metricas(resultado) -> dict[str, float | int | bool]:
    trades = list(resultado.trades)
    pnls = [float(t.pnl) for t in trades]
    rois = [float(t.roi) for t in trades]
    duraciones = [int(t.duracion_velas) for t in trades]

    ganancias = sum(p for p in pnls if p > 0)
    perdidas = abs(sum(p for p in pnls if p < 0))
    profit_factor = _ratio_seguro(ganancias, perdidas)

    return {
        "saldo_inicial": float(resultado.saldo_inicial),
        "saldo_final": float(resultado.saldo_final),
        "total_trades": int(resultado.total_trades),
        "trades_ganadores": int(resultado.trades_ganadores),
        "trades_perdedores": int(resultado.trades_perdedores),
        "win_rate": float(resultado.win_rate),
        "roi_total": float(resultado.roi_total),
        "pnl_total": float(resultado.pnl_total),
        "pnl_promedio": float(resultado.pnl_promedio),
        "max_drawdown": float(resultado.max_drawdown),
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(_sharpe_simple(rois)),
        "duracion_media_velas": float(sum(duraciones) / len(duraciones)) if duraciones else 0.0,
        "parado_por_saldo": bool(resultado.parado_por_saldo),
    }


def _ratio_seguro(numerador: float, denominador: float) -> float:
    if denominador == 0:
        return float("inf") if numerador > 0 else 0.0
    return numerador / denominador


def _sharpe_simple(retornos: list[float]) -> float:
    if len(retornos) < 2:
        return 0.0

    media = sum(retornos) / len(retornos)
    varianza = sum((r - media) ** 2 for r in retornos) / (len(retornos) - 1)
    desviacion = sqrt(varianza)
    if desviacion == 0:
        return 0.0

    return (media / desviacion) * sqrt(len(retornos))

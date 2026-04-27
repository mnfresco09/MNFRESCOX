from __future__ import annotations

from datetime import date
from math import isfinite, sqrt


def calcular_metricas(resultado, fecha_inicio: date, fecha_fin: date) -> dict[str, float | int | bool]:
    trades = list(resultado.trades)
    total_trades = len(trades)

    if total_trades == 0:
        return _metricas_sin_trades()

    pnls = [_safe_float(t.pnl) for t in trades]
    retornos = [_retorno_trade(t) for t in trades]
    duraciones_velas = [int(t.duracion_velas) for t in trades]
    comisiones = [_safe_float(t.comision_total) for t in trades]
    equity = [_safe_float(v) for v in resultado.equity_curve]
    if not equity:
        equity = [_safe_float(resultado.saldo_inicial)]

    trades_long = sum(1 for t in trades if int(t.direccion) == 1)
    trades_short = total_trades - trades_long
    trades_ganadores = sum(1 for pnl in pnls if pnl > 0)
    trades_perdedores = sum(1 for pnl in pnls if pnl < 0)
    trades_neutros = total_trades - trades_ganadores - trades_perdedores
    win_rate = trades_ganadores / total_trades if total_trades else 0.0
    pnl_neto_total = sum(pnls)
    comisiones_total = sum(comisiones)
    pnl_bruto_total = pnl_neto_total + comisiones_total
    pnl_promedio = pnl_neto_total / total_trades if total_trades else 0.0
    saldo_inicial = _safe_float(resultado.saldo_inicial)
    saldo_final = _safe_float(resultado.saldo_final)
    roi_total = pnl_neto_total / saldo_inicial if saldo_inicial else 0.0
    expectancy = sum(retornos) / total_trades
    ganancias = sum(p for p in pnls if p > 0)
    perdidas = abs(sum(p for p in pnls if p < 0))
    profit_factor = _ratio_seguro(ganancias, perdidas)
    trades_por_dia = total_trades / _dias_periodo(fecha_inicio, fecha_fin)

    # Duración media en segundos (ts en microsegundos)
    dur_seg_lista = [
        max(0, int(t.ts_salida - t.ts_entrada)) / 1_000_000
        for t in trades
    ]
    duracion_media_seg = sum(dur_seg_lista) / len(dur_seg_lista)
    duracion_media_velas = sum(duraciones_velas) / len(duraciones_velas)

    return {
        "saldo_inicial": saldo_inicial,
        "saldo_final": saldo_final,
        "total_trades": int(total_trades),
        "trades_long": int(trades_long),
        "trades_short": int(trades_short),
        "trades_ganadores": int(trades_ganadores),
        "trades_perdedores": int(trades_perdedores),
        "trades_neutros": int(trades_neutros),
        "win_rate": float(win_rate),
        "roi_total": float(roi_total),
        "expectancy": float(expectancy),
        "trades_por_dia": float(trades_por_dia),
        "pnl_bruto_total": float(pnl_bruto_total),
        "pnl_total": float(pnl_neto_total),
        "pnl_promedio": float(pnl_promedio),
        "max_drawdown": float(_max_drawdown(equity)),
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(_sharpe_simple(retornos)),
        "duracion_media_seg": float(duracion_media_seg),
        "duracion_media_velas": float(duracion_media_velas),
        "parado_por_saldo": bool(resultado.parado_por_saldo),
    }


def _metricas_sin_trades() -> dict[str, float | int | bool]:
    return {
        "saldo_inicial": 0.0,
        "saldo_final": 0.0,
        "total_trades": 0,
        "trades_long": 0,
        "trades_short": 0,
        "trades_ganadores": 0,
        "trades_perdedores": 0,
        "trades_neutros": 0,
        "win_rate": 0.0,
        "roi_total": 0.0,
        "expectancy": 0.0,
        "trades_por_dia": 0.0,
        "pnl_bruto_total": 0.0,
        "pnl_total": 0.0,
        "pnl_promedio": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "duracion_media_seg": 0.0,
        "duracion_media_velas": 0.0,
        "parado_por_saldo": False,
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

    return media / desviacion


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0

    max_equity = equity[0]
    max_dd = 0.0
    for valor in equity:
        if valor > max_equity:
            max_equity = valor
        if max_equity <= 0:
            continue
        dd = (max_equity - valor) / max_equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _retorno_trade(trade) -> float:
    colateral = _safe_float(trade.colateral)
    if colateral == 0:
        return 0.0
    return _safe_float(trade.pnl) / colateral


def _dias_periodo(fecha_inicio: date, fecha_fin: date) -> int:
    dias = (fecha_fin - fecha_inicio).days + 1
    if dias < 1:
        raise ValueError("[METRICAS] fecha_fin no puede ser anterior a fecha_inicio.")
    return dias


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default

from __future__ import annotations

from math import sqrt


def calcular_metricas(resultado) -> dict[str, float | int | bool]:
    trades = list(resultado.trades)
    pnls = [float(t.pnl) for t in trades]
    rois = [float(t.roi) for t in trades]
    duraciones_velas = [int(t.duracion_velas) for t in trades]
    comisiones = [float(t.comision_total) for t in trades]
    equity = [float(v) for v in resultado.equity_curve]

    total_trades = len(trades)
    trades_long = sum(1 for t in trades if int(t.direccion) == 1)
    trades_short = total_trades - trades_long
    trades_ganadores = sum(1 for pnl in pnls if pnl > 0)
    trades_perdedores = total_trades - trades_ganadores
    win_rate = trades_ganadores / total_trades if total_trades else 0.0
    pnl_neto_total = sum(pnls)
    comisiones_total = sum(comisiones)
    pnl_bruto_total = pnl_neto_total + comisiones_total
    pnl_promedio = pnl_neto_total / total_trades if total_trades else 0.0
    saldo_inicial = float(resultado.saldo_inicial)
    saldo_final = float(resultado.saldo_final)
    roi_total = pnl_neto_total / saldo_inicial if saldo_inicial else 0.0
    ganancias = sum(p for p in pnls if p > 0)
    perdidas = abs(sum(p for p in pnls if p < 0))
    profit_factor = _ratio_seguro(ganancias, perdidas)

    # Duración media en segundos (ts en microsegundos)
    if trades:
        dur_seg_lista = [
            max(0, int(t.ts_salida - t.ts_entrada)) / 1_000_000
            for t in trades
        ]
        duracion_media_seg = sum(dur_seg_lista) / len(dur_seg_lista)
        duracion_media_velas = sum(duraciones_velas) / len(duraciones_velas)
    else:
        duracion_media_seg = 0.0
        duracion_media_velas = 0.0

    return {
        "saldo_inicial": saldo_inicial,
        "saldo_final": saldo_final,
        "total_trades": int(total_trades),
        "trades_long": int(trades_long),
        "trades_short": int(trades_short),
        "trades_ganadores": int(trades_ganadores),
        "trades_perdedores": int(trades_perdedores),
        "win_rate": float(win_rate),
        "roi_total": float(roi_total),
        "pnl_bruto_total": float(pnl_bruto_total),
        "pnl_total": float(pnl_neto_total),
        "pnl_promedio": float(pnl_promedio),
        "max_drawdown": float(_max_drawdown(equity)),
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(_sharpe_simple(rois)),
        "duracion_media_seg": float(duracion_media_seg),
        "duracion_media_velas": float(duracion_media_velas),
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

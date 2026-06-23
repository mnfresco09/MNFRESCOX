"""Conversión Metricas (Rust) → dict + cálculo de `trades_por_dia`.

Las métricas pesadas (max_dd, sharpe, expectancy, conteos…) ya están
calculadas dentro del motor Rust. Aquí sólo añadimos lo que requiere
información externa (las fechas de la ventana del backtest).
"""

from __future__ import annotations

from datetime import date
from math import isfinite

from COMUN import estadistica as est


def calcular_metricas(metricas, fecha_inicio: date, fecha_fin: date) -> dict[str, float | int | bool]:
    total_trades = int(metricas.total_trades)
    if total_trades == 0:
        return _metricas_sin_trades(metricas)

    dias = _dias_periodo(fecha_inicio, fecha_fin)
    anios = est.anios_desde_dias(dias)
    saldo_inicial = _safe_float(metricas.saldo_inicial)
    saldo_final = _safe_float(metricas.saldo_final)
    sharpe = _safe_float(metricas.sharpe_ratio)
    max_dd = _safe_float(metricas.max_drawdown)
    trades_por_dia = total_trades / dias

    # Métricas derivadas, ajustadas por riesgo y por muestra (ver estadistica.py).
    # Aquí el PSR usa la aproximación normal (skew=0, curtosis=3): es la ruta
    # caliente de Optuna y no dispone de los momentos por trade. El reporte
    # recalcula un PSR más fino con asimetría/curtosis reales en analitica.py.
    cagr = est.cagr(saldo_inicial, saldo_final, anios)
    calmar = est.calmar(cagr, max_dd)
    sharpe_anualizado = est.sharpe_anualizado(sharpe, est.trades_por_anio(trades_por_dia))
    psr = est.probabilistic_sharpe_ratio(sharpe, total_trades)

    return {
        "saldo_inicial":        saldo_inicial,
        "saldo_final":          saldo_final,
        "total_trades":         total_trades,
        "trades_long":          int(metricas.trades_long),
        "trades_short":         int(metricas.trades_short),
        "trades_ganadores":     int(metricas.trades_ganadores),
        "trades_perdedores":    int(metricas.trades_perdedores),
        "trades_neutros":       int(metricas.trades_neutros),
        "win_rate":             _safe_float(metricas.win_rate),
        "roi_total":            _safe_float(metricas.roi_total),
        "expectancy":           _safe_float(metricas.expectancy),
        "trades_por_dia":       trades_por_dia,
        "pnl_bruto_total":      _safe_float(metricas.pnl_bruto_total),
        "pnl_total":            _safe_float(metricas.pnl_total),
        "pnl_promedio":         _safe_float(metricas.pnl_promedio),
        "max_drawdown":         max_dd,
        "profit_factor":        _profit_factor(metricas.profit_factor),
        "sharpe_ratio":         sharpe,
        "sharpe_anualizado":    sharpe_anualizado,
        "cagr":                 cagr,
        "calmar":               calmar,
        "psr":                  psr,
        "duracion_media_seg":   _safe_float(metricas.duracion_media_seg),
        "duracion_media_velas": _safe_float(metricas.duracion_media_velas),
        "parado_por_saldo":     bool(metricas.parado_por_saldo),
    }


def _metricas_sin_trades(metricas) -> dict[str, float | int | bool]:
    return {
        "saldo_inicial":        _safe_float(metricas.saldo_inicial),
        "saldo_final":          _safe_float(metricas.saldo_final),
        "total_trades":         0,
        "trades_long":          0,
        "trades_short":         0,
        "trades_ganadores":     0,
        "trades_perdedores":    0,
        "trades_neutros":       0,
        "win_rate":             0.0,
        "roi_total":            0.0,
        "expectancy":           0.0,
        "trades_por_dia":       0.0,
        "pnl_bruto_total":      0.0,
        "pnl_total":            0.0,
        "pnl_promedio":         0.0,
        "max_drawdown":         0.0,
        "profit_factor":        0.0,
        "sharpe_ratio":         0.0,
        "sharpe_anualizado":    0.0,
        "cagr":                 0.0,
        "calmar":               0.0,
        "psr":                  0.0,
        "duracion_media_seg":   0.0,
        "duracion_media_velas": 0.0,
        "parado_por_saldo":     bool(metricas.parado_por_saldo),
    }


def _profit_factor(value: float) -> float:
    """Conserva +inf cuando no hay pérdidas y hay ganancias (consumido por puntuacion)."""
    f = float(value)
    if f != f:  # NaN
        return 0.0
    return f


def _dias_periodo(fecha_inicio: date, fecha_fin: date) -> int:
    dias = (fecha_fin - fecha_inicio).days + 1
    if dias < 1:
        raise ValueError("[METRICAS] fecha_fin no puede ser anterior a fecha_inicio.")
    return dias


def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if isfinite(f) else default

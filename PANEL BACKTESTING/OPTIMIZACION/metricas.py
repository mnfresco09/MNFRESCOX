"""Conversión Metricas (Rust) → dict + cálculo de `trades_por_dia`.

Las métricas pesadas (max_dd, sharpe, expectancy, conteos…) ya están
calculadas dentro del motor Rust. Aquí sólo añadimos lo que requiere
información externa (las fechas de la ventana del backtest).
"""

from __future__ import annotations

from datetime import date
from math import isfinite


def calcular_metricas(metricas, fecha_inicio: date, fecha_fin: date) -> dict[str, float | int | bool]:
    total_trades = int(metricas.total_trades)
    if total_trades == 0:
        return _metricas_sin_trades(metricas)

    dias = _dias_periodo(fecha_inicio, fecha_fin)
    return {
        "saldo_inicial":        _safe_float(metricas.saldo_inicial),
        "saldo_final":          _safe_float(metricas.saldo_final),
        "total_trades":         total_trades,
        "trades_long":          int(metricas.trades_long),
        "trades_short":         int(metricas.trades_short),
        "trades_ganadores":     int(metricas.trades_ganadores),
        "trades_perdedores":    int(metricas.trades_perdedores),
        "trades_neutros":       int(metricas.trades_neutros),
        "win_rate":             _safe_float(metricas.win_rate),
        "roi_total":            _safe_float(metricas.roi_total),
        "expectancy":           _safe_float(metricas.expectancy),
        "trades_por_dia":       total_trades / dias,
        "pnl_bruto_total":      _safe_float(metricas.pnl_bruto_total),
        "pnl_total":            _safe_float(metricas.pnl_total),
        "pnl_promedio":         _safe_float(metricas.pnl_promedio),
        "max_drawdown":         _safe_float(metricas.max_drawdown),
        "profit_factor":        _profit_factor(metricas.profit_factor),
        "sharpe_ratio":         _safe_float(metricas.sharpe_ratio),
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

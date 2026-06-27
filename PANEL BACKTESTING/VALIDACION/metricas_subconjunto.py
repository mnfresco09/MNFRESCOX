"""Recomputación de métricas sobre una VENTANA de índices (Fase 2).

La pieza que permite hacer CPCV/WFA sin tocar el motor Rust: se simula una vez
sobre la serie completa (los indicadores conservan todo su histórico, sin cortes
artificiales en los bordes) y luego se recortan los trades a la ventana temporal
del fold, recomputando las métricas SOLO con los trades cuya entrada cae dentro
de la ventana.

Convención de retorno por trade: `pnl / saldo_por_trade` (PnL sobre el colateral
comprometido). El Sharpe es la media/desviación típica de esos retornos por
operación, coherente con la convención del resto del sistema. Como CPCV compara
OOS vs IS recomputando AMBOS con esta misma fórmula, cualquier sesgo de convención
se cancela en el ratio OOS/IS.

Módulo puro (NumPy + estadística): verificable de forma aislada.
"""

from __future__ import annotations

import numpy as np

from COMUN import estadistica as est


def metricas_subconjunto(
    trades: dict,
    idx_min: int,
    idx_max: int,
    *,
    saldo_inicial: float,
    saldo_por_trade: float,
) -> dict:
    """Métricas recomputadas con los trades que entran en [idx_min, idx_max).

    Parameters
    ----------
    trades:
        Dict columnar de trades (como el de `SimResult.take_trades()`), con al
        menos `idx_entrada` (int) y `pnl` (float). Otras claves se ignoran.
    idx_min, idx_max:
        Ventana semiabierta de índices de ENTRADA del trade (en el espacio de
        índices de ejecución del motor).
    saldo_inicial, saldo_por_trade:
        Capital inicial de la ventana y colateral por operación (para el retorno
        por trade).

    Returns
    -------
    dict
        Claves: total_trades, sharpe_ratio, psr, roi_total, max_drawdown,
        pnl_total, saldo_final.
    """
    idx_entrada = np.asarray(trades.get("idx_entrada", []), dtype=np.int64)
    pnl = np.asarray(trades.get("pnl", []), dtype=np.float64)
    if idx_entrada.shape[0] != pnl.shape[0]:
        raise ValueError("idx_entrada y pnl deben tener la misma longitud.")

    a, b = int(idx_min), int(idx_max)
    if b <= a:
        return _vacio(saldo_inicial)

    mask = (idx_entrada >= a) & (idx_entrada < b)
    return _resumen(idx_entrada, pnl, mask, saldo_inicial, saldo_por_trade)


def metricas_en_indices(
    trades: dict,
    indices,
    *,
    saldo_inicial: float,
    saldo_por_trade: float,
) -> dict:
    """Igual que `metricas_subconjunto` pero restringe a un CONJUNTO de índices.

    Necesario para CPCV: el train purgado y el test combinatorio NO son rangos
    contiguos, sino uniones de bloques. Un trade entra en la métrica si su índice
    de entrada pertenece al conjunto `indices`.
    """
    idx_entrada = np.asarray(trades.get("idx_entrada", []), dtype=np.int64)
    pnl = np.asarray(trades.get("pnl", []), dtype=np.float64)
    if idx_entrada.shape[0] != pnl.shape[0]:
        raise ValueError("idx_entrada y pnl deben tener la misma longitud.")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return _vacio(saldo_inicial)
    mask = np.isin(idx_entrada, indices)
    return _resumen(idx_entrada, pnl, mask, saldo_inicial, saldo_por_trade)


def _resumen(
    idx_entrada: np.ndarray,
    pnl: np.ndarray,
    mask: np.ndarray,
    saldo_inicial: float,
    saldo_por_trade: float,
) -> dict:
    if not mask.any():
        return _vacio(saldo_inicial)

    # Orden cronológico por índice de entrada (path-dependiente para el max DD).
    orden = np.argsort(idx_entrada[mask], kind="stable")
    pnl_win = pnl[mask][orden]
    pnl_win = pnl_win[np.isfinite(pnl_win)]
    n = int(pnl_win.size)
    if n == 0:
        return _vacio(saldo_inicial)

    spt = float(saldo_por_trade) if saldo_por_trade else 1.0
    retornos = pnl_win / spt
    sd = float(retornos.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(retornos.mean() / sd) if sd > 0.0 else 0.0
    psr = est.probabilistic_sharpe_ratio(sharpe, n)

    equity = float(saldo_inicial) + np.concatenate([[0.0], np.cumsum(pnl_win)])
    max_dd = _max_drawdown(equity)
    saldo_final = float(equity[-1])
    pnl_total = float(pnl_win.sum())
    roi = pnl_total / float(saldo_inicial) if saldo_inicial else 0.0

    return {
        "total_trades": n,
        "sharpe_ratio": sharpe,
        "psr": float(psr),
        "roi_total": float(roi),
        "max_drawdown": float(max_dd),
        "pnl_total": pnl_total,
        "saldo_final": saldo_final,
    }


def retornos_por_trade(trades: dict, *, saldo_por_trade: float) -> np.ndarray:
    """Retornos por trade (pnl/colateral) en orden cronológico — para el bootstrap."""
    idx_entrada = np.asarray(trades.get("idx_entrada", []), dtype=np.int64)
    pnl = np.asarray(trades.get("pnl", []), dtype=np.float64)
    if idx_entrada.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    orden = np.argsort(idx_entrada, kind="stable")
    spt = float(saldo_por_trade) if saldo_por_trade else 1.0
    r = pnl[orden] / spt
    return r[np.isfinite(r)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vacio(saldo_inicial: float) -> dict:
    return {
        "total_trades": 0,
        "sharpe_ratio": 0.0,
        "psr": 0.0,
        "roi_total": 0.0,
        "max_drawdown": 0.0,
        "pnl_total": 0.0,
        "saldo_final": float(saldo_inicial),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    pico = np.maximum.accumulate(equity)
    pico = np.where(pico <= 0.0, np.nan, pico)
    dd = (pico - equity) / pico
    dd = dd[np.isfinite(dd)]
    return float(dd.max()) if dd.size else 0.0

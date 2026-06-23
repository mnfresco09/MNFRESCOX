"""Analitica avanzada de un trial a partir del replay (trades + equity).

El motor Rust solo calcula las metricas basicas (sharpe, max_dd, win_rate,
profit_factor, etc.). Aqui se derivan las metricas avanzadas que necesitan la
serie de trades y la curva de equity completas: Sortino, Calmar, CAGR,
recovery factor, exposicion, VaR/CVaR, asimetria, curtosis, payoff, rachas y
percentiles. Tambien se construyen las distribuciones para los graficos.

Es una capa de solo-lectura sobre `ReplayTrial`: no toca el motor ni los datos.
Tanto el report HTML como el Excel la consumen, asi que los numeros son siempre
identicos entre ambos.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from COMUN import estadistica as est


def analitica_avanzada(
    *,
    metricas: dict,
    trades: dict[str, np.ndarray],
    equity_curve: np.ndarray,
    fecha_inicio: date,
    fecha_fin: date,
) -> dict[str, float | int]:
    """Devuelve un dict de metricas avanzadas. Tolera 0 trades."""
    pnl = np.asarray(trades.get("pnl", np.empty(0)), dtype=np.float64)
    roi = np.asarray(trades.get("roi", np.empty(0)), dtype=np.float64)
    n = int(pnl.shape[0])
    eq = np.asarray(equity_curve, dtype=np.float64)

    dias = max(1, (fecha_fin - fecha_inicio).days + 1)
    anios = dias / 365.25

    if n == 0 or eq.shape[0] < 2:
        return _vacia()

    saldo_inicial = float(eq[0])
    saldo_final = float(eq[-1])

    # Drawdown en dinero y CAGR sobre la curva de equity.
    picos = np.maximum.accumulate(eq)
    dd_money = float(np.max(picos - eq))
    max_dd_frac = float(metricas.get("max_drawdown", 0.0))
    pnl_total = float(metricas.get("pnl_total", float(np.sum(pnl))))

    cagr = est.cagr(saldo_inicial, saldo_final, anios)
    calmar = est.calmar(cagr, max_dd_frac)
    recovery = pnl_total / dd_money if dd_money > 0 else 0.0

    # Sortino por trade: misma escala que el sharpe del motor (media/desv por
    # trade, sin anualizar), pero usando solo la desviacion bajista (MAR=0).
    media_roi = float(np.mean(roi))
    bajistas = np.minimum(roi, 0.0)
    desv_bajista = float(math.sqrt(float(np.mean(bajistas * bajistas))))
    sortino = est.sortino_por_trade(media_roi, desv_bajista)

    # Exposicion: fraccion del periodo con posicion abierta.
    if "ts_entrada" in trades and "ts_salida" in trades:
        dur_us = (
            np.asarray(trades["ts_salida"], dtype=np.int64)
            - np.asarray(trades["ts_entrada"], dtype=np.int64)
        )
        en_mercado_seg = float(np.sum(np.clip(dur_us, 0, None))) / 1_000_000.0
    else:
        en_mercado_seg = 0.0
    exposure = en_mercado_seg / (dias * 86_400.0) if dias > 0 else 0.0
    exposure = min(1.0, max(0.0, exposure))

    ganadores = pnl[pnl > 0]
    perdedores = pnl[pnl < 0]
    avg_win = float(np.mean(ganadores)) if ganadores.size else 0.0
    avg_loss = float(np.mean(perdedores)) if perdedores.size else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    var95 = float(np.percentile(pnl, 5))
    cola = pnl[pnl <= var95]
    cvar95 = float(np.mean(cola)) if cola.size else var95

    skew, kurt = _skew_kurtosis(pnl)
    perc = {
        int(q): float(np.percentile(pnl, q)) for q in (5, 25, 50, 75, 95)
    }

    # Versiones anualizadas (comparables entre estrategias de distinta frecuencia)
    # y PSR fino con asimetria/curtosis reales (mas preciso que el del run, que
    # usa la aproximacion normal por velocidad).
    sharpe = float(metricas.get("sharpe_ratio", 0.0))
    tpa = est.trades_por_anio(n / dias if dias > 0 else 0.0)
    sharpe_anualizado = est.sharpe_anualizado(sharpe, tpa)
    sortino_anualizado = est.sharpe_anualizado(sortino, tpa)
    psr = est.probabilistic_sharpe_ratio(sharpe, n, asimetria=skew, curtosis=kurt + 3.0)

    return {
        "sortino_ratio": _fin(sortino),
        "sharpe_anualizado": _fin(sharpe_anualizado),
        "sortino_anualizado": _fin(sortino_anualizado),
        "psr": _fin(psr),
        "calmar_ratio": _fin(calmar),
        "cagr": _fin(cagr),
        "recovery_factor": _fin(recovery),
        "max_drawdown_money": _fin(dd_money),
        "exposure": _fin(exposure),
        "payoff_ratio": _fin(payoff),
        "avg_win": _fin(avg_win),
        "avg_loss": _fin(avg_loss),
        "best_trade": _fin(float(np.max(pnl))),
        "worst_trade": _fin(float(np.min(pnl))),
        "var95": _fin(var95),
        "cvar95": _fin(cvar95),
        "skew": _fin(skew),
        "kurtosis": _fin(kurt),
        "max_win_streak": _racha(pnl, ganadora=True),
        "max_loss_streak": _racha(pnl, ganadora=False),
        "percentil_5": _fin(perc[5]),
        "percentil_25": _fin(perc[25]),
        "percentil_50": _fin(perc[50]),
        "percentil_75": _fin(perc[75]),
        "percentil_95": _fin(perc[95]),
    }


def distribuciones(trades: dict[str, np.ndarray], *, bins_pnl: int = 16, bins_dur: int = 14) -> dict:
    """Histogramas y conteos para los graficos de distribucion."""
    pnl = np.asarray(trades.get("pnl", np.empty(0)), dtype=np.float64)
    dur = np.asarray(trades.get("duracion_velas", np.empty(0)), dtype=np.float64)
    motivos = np.asarray(trades.get("motivo_salida", np.empty(0)), dtype=np.int64)
    from MOTOR.wrapper import MOTIVOS

    conteo_motivo: dict[str, int] = {}
    for codigo in motivos.tolist():
        nombre = MOTIVOS[int(codigo)] if 0 <= int(codigo) < len(MOTIVOS) else str(codigo)
        conteo_motivo[nombre] = conteo_motivo.get(nombre, 0) + 1

    return {
        "pnl": _histograma(pnl, bins_pnl),
        "duracion": _histograma(dur, bins_dur),
        "motivo": conteo_motivo,
    }


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _histograma(data: np.ndarray, bins: int) -> dict:
    if data.size == 0:
        return {"edges": [], "counts": []}
    lo = float(np.min(data))
    hi = float(np.max(data))
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(data, bins=bins, range=(lo, hi))
    return {
        "edges": [round(float(e), 2) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
    }


def _skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    n = x.size
    if n < 3:
        return 0.0, 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 0:
        return 0.0, 0.0
    z = (x - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    return skew, kurt


def _racha(pnl: np.ndarray, *, ganadora: bool) -> int:
    maxima = actual = 0
    for v in pnl.tolist():
        es = (v > 0) if ganadora else (v < 0)
        if es:
            actual += 1
            maxima = max(maxima, actual)
        else:
            actual = 0
    return int(maxima)


def _fin(v: float) -> float:
    f = float(v)
    return f if math.isfinite(f) else 0.0


def _vacia() -> dict[str, float | int]:
    claves = [
        "sortino_ratio", "sharpe_anualizado", "sortino_anualizado", "psr",
        "calmar_ratio", "cagr", "recovery_factor",
        "max_drawdown_money", "exposure", "payoff_ratio", "avg_win", "avg_loss",
        "best_trade", "worst_trade", "var95", "cvar95", "skew", "kurtosis",
        "percentil_5", "percentil_25", "percentil_50", "percentil_75", "percentil_95",
    ]
    salida: dict[str, float | int] = {k: 0.0 for k in claves}
    salida["max_win_streak"] = 0
    salida["max_loss_streak"] = 0
    return salida

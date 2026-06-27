"""Rendimiento por régimen de mercado (Fase 5).

Una estrategia que solo gana en un régimen específico es una apuesta direccional
disfrazada, no un edge robusto. Este módulo separa el rendimiento de los trades
por la etiqueta de régimen vigente (alcista / bajista / lateral, derivada del
detector MACD existente) y reporta las métricas de cada uno.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricasRegimen:
    regimen: str
    n_trades: int
    retorno_total: float
    retorno_medio: float
    sharpe: float
    win_rate: float


def rendimiento_por_regimen(retornos_trade, etiquetas_regimen) -> dict[str, MetricasRegimen]:
    """Agrupa los retornos por trade según su etiqueta de régimen.

    `retornos_trade` y `etiquetas_regimen` deben tener la misma longitud (una
    entrada por trade). Devuelve un dict {regimen: MetricasRegimen}.
    """
    r = np.asarray(retornos_trade, dtype=np.float64)
    etiquetas = np.asarray(list(etiquetas_regimen))
    if r.shape[0] != etiquetas.shape[0]:
        raise ValueError(
            f"retornos_trade ({r.shape[0]}) y etiquetas_regimen ({etiquetas.shape[0]}) "
            "deben tener la misma longitud."
        )

    salida: dict[str, MetricasRegimen] = {}
    for etiqueta in sorted(set(etiquetas.tolist())):
        mask = etiquetas == etiqueta
        sub = r[mask]
        sub = sub[np.isfinite(sub)]
        if sub.size == 0:
            continue
        sd = float(sub.std(ddof=1)) if sub.size > 1 else 0.0
        sharpe = float(sub.mean() / sd) if sd > 0.0 else 0.0
        salida[str(etiqueta)] = MetricasRegimen(
            regimen=str(etiqueta),
            n_trades=int(sub.size),
            retorno_total=float(sub.sum()),
            retorno_medio=float(sub.mean()),
            sharpe=sharpe,
            win_rate=float((sub > 0.0).mean()),
        )
    return salida

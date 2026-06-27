"""Rendimiento por régimen de mercado (Fase 5).

Una estrategia que solo gana en un régimen específico es una apuesta direccional
disfrazada, no un edge robusto. Este módulo separa el rendimiento de los trades
por la etiqueta de régimen vigente (alcista / bajista / lateral, derivada del
detector MACD existente) y reporta las métricas de cada uno.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def etiquetas_macd(
    close,
    *,
    rapida: int = 12,
    lenta: int = 26,
    senal: int = 9,
    umbral_lateral: float = 0.1,
) -> np.ndarray:
    """Etiqueta cada vela como alcista / bajista / lateral según el MACD.

    Régimen = signo del histograma MACD (macd − señal). Para no clasificar como
    direccional el ruido cerca de cero, se define una banda lateral proporcional
    a la desviación típica del histograma (`umbral_lateral` · std).

    Devuelve un array de strings de la misma longitud que `close`.
    """
    c = np.asarray(close, dtype=np.float64)
    if c.size == 0:
        return np.empty(0, dtype=object)
    macd = _ema(c, rapida) - _ema(c, lenta)
    histograma = macd - _ema(macd, senal)
    sd = float(np.std(histograma)) or 1.0
    banda = float(umbral_lateral) * sd
    return np.where(
        histograma > banda, "alcista",
        np.where(histograma < -banda, "bajista", "lateral"),
    )


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Media móvil exponencial (mismo criterio de span que pandas.ewm)."""
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.shape[0]):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


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

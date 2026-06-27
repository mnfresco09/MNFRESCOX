"""Bootstrap de la secuencia de trades (Fase 5).

Remuestrea los retornos de los trades con reemplazo (o por bloques si hay
autocorrelación) y genera miles de curvas de equity alternativas, para obtener la
DISTRIBUCIÓN de equity final, max drawdown y Sharpe. Responde a "¿mi drawdown del
8% fue representativo o tuve suerte con el orden?". Un único max drawdown es casi
inútil sin su distribución.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResultadoBootstrap:
    n_iter: int
    n_trades: int
    saldo_inicial: float
    equity_final: np.ndarray
    max_drawdown: np.ndarray
    sharpe: np.ndarray

    def percentiles_equity(self, qs=(5, 25, 50, 75, 95)) -> dict[int, float]:
        return {int(q): float(np.percentile(self.equity_final, q)) for q in qs}

    @property
    def p5_equity_final(self) -> float:
        return float(np.percentile(self.equity_final, 5))

    def percentiles_max_drawdown(self, qs=(50, 95, 99)) -> dict[int, float]:
        return {int(q): float(np.percentile(self.max_drawdown, q)) for q in qs}


def bootstrap_trades(
    retornos_trade,
    *,
    n_iter: int = 10000,
    tam_bloque: int = 1,
    saldo_inicial: float = 10_000.0,
    compuesto: bool = False,
    seed: int | None = None,
) -> ResultadoBootstrap:
    """Bootstrap de la secuencia de trades.

    Parameters
    ----------
    retornos_trade:
        Retornos POR TRADE. En modo aditivo (`compuesto=False`) se interpretan
        como PnL en divisa que se suma al saldo; en modo compuesto, como
        fracciones que multiplican el saldo (1 + r).
    n_iter:
        Nº de curvas de equity remuestreadas.
    tam_bloque:
        1 = bootstrap i.i.d. (independiente). > 1 = block bootstrap (preserva
        autocorrelación de bloques de trades consecutivos).
    """
    r = np.asarray(retornos_trade, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        raise ValueError("retornos_trade no contiene valores finitos.")
    if n_iter < 1:
        raise ValueError("n_iter debe ser >= 1.")
    if tam_bloque < 1:
        raise ValueError("tam_bloque debe ser >= 1.")

    rng = np.random.default_rng(seed)
    muestras = _remuestrear(r, n_iter=n_iter, n=n, tam_bloque=tam_bloque, rng=rng)  # (n_iter, n)

    equity = _equity(muestras, saldo_inicial=saldo_inicial, compuesto=compuesto)  # (n_iter, n+1)
    equity_final = equity[:, -1]
    max_dd = _max_drawdown(equity)
    sd = muestras.std(axis=1, ddof=1)
    media = muestras.mean(axis=1)
    sharpe = np.where(sd > 0.0, media / sd, 0.0)

    return ResultadoBootstrap(
        n_iter=int(n_iter),
        n_trades=int(n),
        saldo_inicial=float(saldo_inicial),
        equity_final=equity_final,
        max_drawdown=max_dd,
        sharpe=sharpe.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remuestrear(r: np.ndarray, *, n_iter: int, n: int, tam_bloque: int, rng) -> np.ndarray:
    if tam_bloque == 1:
        idx = rng.integers(0, n, size=(n_iter, n))
        return r[idx]
    # Block bootstrap: se concatenan bloques contiguos hasta cubrir n.
    n_bloques = int(np.ceil(n / tam_bloque))
    inicios = rng.integers(0, n, size=(n_iter, n_bloques))
    desplaz = np.arange(tam_bloque)
    # (n_iter, n_bloques, tam_bloque) -> índices circulares -> recortar a n.
    idx = (inicios[:, :, None] + desplaz[None, None, :]) % n
    idx = idx.reshape(n_iter, n_bloques * tam_bloque)[:, :n]
    return r[idx]


def _equity(muestras: np.ndarray, *, saldo_inicial: float, compuesto: bool) -> np.ndarray:
    n_iter = muestras.shape[0]
    inicial = np.full((n_iter, 1), float(saldo_inicial))
    if compuesto:
        factores = np.cumprod(1.0 + muestras, axis=1)
        return np.hstack([inicial, saldo_inicial * factores])
    acum = np.cumsum(muestras, axis=1)
    return np.hstack([inicial, saldo_inicial + acum])


def _max_drawdown(equity: np.ndarray) -> np.ndarray:
    """Max drawdown (fracción positiva) por fila de una matriz de equity."""
    pico = np.maximum.accumulate(equity, axis=1)
    dd = (pico - equity) / pico
    return dd.max(axis=1).astype(np.float64)

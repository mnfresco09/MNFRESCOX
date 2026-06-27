"""Evaluación a nivel de cartera (Fase 6).

En vez de optimizar cada activo aislado, evalúa la CONTRIBUCIÓN MARGINAL al
Sharpe de la cartera considerando correlaciones entre estrategias. Dos
estrategias mediocres descorrelacionadas pueden ser una cartera excelente; dos
buenas correlacionadas son redundantes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContribucionEstrategia:
    indice: int
    sharpe_individual: float
    contribucion_marginal: float   # ΔSharpe de la cartera al quitar la estrategia
    redundante: bool               # quitarla apenas baja (o sube) el Sharpe


def _validar_matriz(retornos: np.ndarray) -> np.ndarray:
    M = np.asarray(retornos, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"retornos debe ser 2D (T, M); ndim={M.ndim}.")
    if M.shape[1] < 1:
        raise ValueError("Se necesita al menos una estrategia (columna).")
    return M


def matriz_correlacion(retornos) -> np.ndarray:
    """Matriz de correlación entre estrategias (columnas)."""
    M = _validar_matriz(retornos)
    if M.shape[0] < 2:
        return np.eye(M.shape[1])
    return np.corrcoef(M, rowvar=False)


def sharpe_cartera(retornos, pesos=None) -> float:
    """Sharpe (por observación) de la cartera con los pesos dados (iguales por defecto)."""
    M = _validar_matriz(retornos)
    n_estrategias = M.shape[1]
    if pesos is None:
        pesos = np.full(n_estrategias, 1.0 / n_estrategias)
    pesos = np.asarray(pesos, dtype=np.float64)
    if pesos.shape[0] != n_estrategias:
        raise ValueError("pesos debe tener una entrada por estrategia.")
    serie = M @ pesos
    sd = float(serie.std(ddof=1)) if serie.size > 1 else 0.0
    return float(serie.mean() / sd) if sd > 0.0 else 0.0


def contribucion_marginal_sharpe(retornos) -> list[ContribucionEstrategia]:
    """Contribución marginal de cada estrategia al Sharpe de la cartera.

    Se mide por leave-one-out: ΔSharpe = Sharpe(cartera completa) −
    Sharpe(cartera sin la estrategia i). Positivo = añade valor; ≈0 o negativo =
    redundante o perjudicial (probablemente muy correlacionada con el resto).
    """
    M = _validar_matriz(retornos)
    n = M.shape[1]
    sharpe_total = sharpe_cartera(M)
    salida: list[ContribucionEstrategia] = []
    for i in range(n):
        col = M[:, i]
        sd_i = float(col.std(ddof=1)) if col.size > 1 else 0.0
        sharpe_ind = float(col.mean() / sd_i) if sd_i > 0.0 else 0.0
        if n == 1:
            contrib = sharpe_total
        else:
            resto = np.delete(M, i, axis=1)
            contrib = sharpe_total - sharpe_cartera(resto)
        salida.append(
            ContribucionEstrategia(
                indice=i,
                sharpe_individual=sharpe_ind,
                contribucion_marginal=float(contrib),
                redundante=bool(contrib <= 0.0),
            )
        )
    return salida

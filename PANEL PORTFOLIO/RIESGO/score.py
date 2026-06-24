"""Score final de cartera y elección de la recomendada (pregunta 3 → decisión).

`calcular_score_cartera` combina recompensas (Sharpe/K-ratio) con penalizaciones
por métricas de cola (CVaR/Expected Shortfall FHS, CDaR, VaR, concentración y
turnover). Para que sean comparables, cada métrica se ESTANDARIZA en z-score
TRANSVERSAL sobre el conjunto de candidatos y se combina con los pesos de
`_tecnico.PESOS_SCORE`.

    score = w_sharpe·z(sharpe) + w_k·z(k_ratio)
            − w_cvar·z(|CVaR99|) − w_cdar·z(|CDaR|)
            − w_var·z(|VaR99|) − w_hhi·z(HHI) − w_turn·z(turnover)

Mayor score = mejor binomio rentabilidad/riesgo bajo el régimen actual.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from CONFIGURACION import _tecnico
from CONTRATOS.modelos import Configuracion, PortfolioCandidate


def _z(valores: np.ndarray) -> np.ndarray:
    mu = float(np.mean(valores))
    sd = float(np.std(valores))
    if sd < 1e-12:
        return np.zeros_like(valores)
    return (valores - mu) / sd


def _turnover(pesos: pd.Series, previa: pd.Series | None) -> float:
    if previa is None:
        return 0.0
    comun = pesos.reindex(previa.index).fillna(0.0)
    return float(np.abs(comun.to_numpy() - previa.to_numpy()).sum())


def calcular_score_cartera(
    candidatos: tuple[PortfolioCandidate, ...],
    cfg: Configuracion,
    cartera_previa: pd.Series | None = None,
) -> tuple[PortfolioCandidate, ...]:
    """Devuelve los candidatos con `score` y `detalle_score` rellenos."""
    pesos = _tecnico.PESOS_SCORE
    sharpe = np.array([c.sharpe for c in candidatos], dtype=float)
    k_ratio = np.array([c.k_ratio if c.k_ratio is not None else 0.0 for c in candidatos], dtype=float)
    var99 = np.array([abs(c.forecast.var_fhs_99) if c.forecast else 0.0 for c in candidatos])
    cvar99 = np.array([abs(c.forecast.cvar_fhs_99) if c.forecast else 0.0 for c in candidatos])
    cdar = np.array([abs(c.simulacion.cdar_30d) if c.simulacion else 0.0 for c in candidatos])
    hhi = np.array([c.descomposicion.concentracion_hhi for c in candidatos], dtype=float)
    turn = np.array([_turnover(c.pesos, cartera_previa) for c in candidatos], dtype=float)
    falta_linealidad = np.array([
        1.0 - c.r2_curva_capital if c.r2_curva_capital is not None else 1.0
        for c in candidatos
    ], dtype=float)

    z_sharpe, z_k, z_var, z_cvar, z_cdar, z_hhi, z_turn, z_linealidad = (
        _z(x) for x in (sharpe, k_ratio, var99, cvar99, cdar, hhi, turn, falta_linealidad)
    )

    salida: list[PortfolioCandidate] = []
    for i, c in enumerate(candidatos):
        componentes = (
            ("sharpe", pesos["sharpe"] * float(z_sharpe[i])),
            ("k_ratio", pesos["k_ratio"] * float(z_k[i])),
            ("var", -pesos["var"] * float(z_var[i])),
            ("cvar", -pesos["cvar"] * float(z_cvar[i])),
            ("cdar", -pesos["cdar"] * float(z_cdar[i])),
            ("concentracion", -pesos["concentracion"] * float(z_hhi[i])),
            ("turnover", -pesos["turnover"] * float(z_turn[i])),
            ("linealidad", -pesos["linealidad"] * float(z_linealidad[i])),
        )
        score = float(sum(v for _, v in componentes))
        salida.append(replace(c, score=score, detalle_score=componentes))
    return tuple(salida)

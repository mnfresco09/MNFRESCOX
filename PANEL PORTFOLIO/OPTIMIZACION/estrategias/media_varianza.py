"""Estrategia Champion: Media-Varianza restringida."""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.modelos import (
    Configuracion,
    MomentsResult,
    PortfolioCandidate,
    PortfolioInput,
    ResultadoFrontera,
)
from OPTIMIZACION.frontera import construir_frontera

from .base import OptimizadorBase, construir_candidate_comun

PERCENTILES = (("bajo", 0.20), ("medio", 0.50), ("alto", 0.80))


class OptimizadorMediaVarianza(OptimizadorBase):
    """Wrapper Strategy del motor Markowitz actual."""

    nombre = "MARKOWITZ"

    def optimizar(
        self,
        entrada: PortfolioInput,
        momentos: MomentsResult,
        cfg: Configuracion,
        frontera: ResultadoFrontera | None = None,
    ) -> tuple[PortfolioCandidate, ...]:
        fr = frontera or construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
        puntos = fr.puntos.sort_values("volatilidad").reset_index(drop=True)
        candidatos: list[PortfolioCandidate] = []
        for nivel, percentil in PERCENTILES:
            fila = _fila_en_percentil_riesgo(puntos, percentil)
            candidatos.append(self._candidate_desde_fila(nivel, fila, entrada, momentos, cfg))
        fila_max_sharpe = puntos.iloc[int(puntos["sharpe"].to_numpy(dtype=float).argmax())]
        candidatos.append(self._candidate_desde_fila("max_sharpe", fila_max_sharpe, entrada, momentos, cfg))
        return tuple(candidatos)

    def _candidate_desde_fila(
        self,
        nivel: str,
        fila: pd.Series,
        entrada: PortfolioInput,
        momentos: MomentsResult,
        cfg: Configuracion,
    ) -> PortfolioCandidate:
        activos = list(momentos.cov_estructural.index)
        pesos = pd.Series([float(fila[f"peso·{a}"]) for a in activos], index=activos)
        return construir_candidate_comun(
            nivel=nivel,
            motor=self.nombre,
            pesos=pesos,
            entrada=entrada,
            momentos=momentos,
            cfg=cfg,
        )


def _fila_en_percentil_riesgo(puntos: pd.DataFrame, percentil: float) -> pd.Series:
    objetivo = float(np.quantile(puntos["volatilidad"].to_numpy(dtype=float), percentil))
    idx = int((puntos["volatilidad"] - objetivo).abs().argmin())
    return puntos.iloc[idx]

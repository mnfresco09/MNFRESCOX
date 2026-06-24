"""Estrategia Champion: Media-Varianza restringida."""

from __future__ import annotations

from CONTRATOS.modelos import (
    Configuracion,
    MomentsResult,
    PortfolioCandidate,
    PortfolioInput,
    ResultadoFrontera,
)
from OPTIMIZACION.frontera import construir_frontera

from .base import OptimizadorBase, construir_candidate_comun


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
        candidato = construir_candidate_comun(
            nivel="markowitz",
            motor=self.nombre,
            pesos=fr.maximo_sharpe_pesos,
            entrada=entrada,
            momentos=momentos,
            cfg=cfg,
        )
        return (candidato,)

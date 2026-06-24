"""Fachada de ANALISIS — ensambla `MomentsResult` (preguntas 1 y 2).

Estadística individual por activo + doble lente de covarianza (Ledoit-Wolf
estructural + EWMA táctica) + retorno ajustado (shrinkage/Black-Litterman) +
correlación media y de cola. No optimiza ni reporta: solo describe.
"""

from __future__ import annotations

import pandas as pd

from CONTRATOS.modelos import (
    Configuracion,
    EstadisticaActivo,
    MomentsResult,
    PortfolioInput,
)

from .correlacion import correlacion_cola, correlacion_media
from .momentos import (
    covarianza_ewma,
    covarianza_ledoit_wolf,
    estadistica_por_activo,
    retornos_medios,
    volatilidades,
)
from .retorno_esperado import estimar_retorno_esperado

PERCENTIL_COLA = 0.10


def calcular_momentos(entrada: PortfolioInput, cfg: Configuracion) -> MomentsResult:
    log_retornos = entrada.log_retornos

    # Doble lente de covarianza.
    cov_estructural, shrink_cov = covarianza_ledoit_wolf(log_retornos, cfg.dias_anio)
    cov_tactica = covarianza_ewma(log_retornos, cfg.dias_anio, cfg.lambda_ewma)

    vols = volatilidades(cov_estructural)
    vols_tac = volatilidades(cov_tactica)

    # Retornos: histórico crudo + ajustado (shrinkage o Black-Litterman).
    mu_medio = retornos_medios(log_retornos, cfg.dias_anio)
    mu_ajustado, fuente_mu = estimar_retorno_esperado(log_retornos, cov_estructural, cfg)

    corr = correlacion_media(log_retornos)
    corr_cola, _obs = correlacion_cola(log_retornos, cfg.activo_referencia, PERCENTIL_COLA)

    stats = estadistica_por_activo(log_retornos, mu_medio, mu_ajustado, vols, vols_tac)
    estadisticas = tuple(
        EstadisticaActivo(ticker=a, **stats[a]) for a in log_retornos.columns
    )

    return MomentsResult(
        activos=entrada.activos,
        retornos_ajustados=mu_ajustado,
        retornos_medios=mu_medio,
        cov_estructural=cov_estructural,
        cov_tactica=cov_tactica,
        volatilidades=vols,
        volatilidades_tacticas=vols_tac,
        correlacion=corr,
        correlacion_cola=corr_cola,
        shrinkage_cov=shrink_cov,
        shrinkage_retorno=cfg.shrinkage_retorno if fuente_mu == "shrinkage" else 0.0,
        estadisticas=estadisticas,
        fuente_tactica="EWMA",
    )

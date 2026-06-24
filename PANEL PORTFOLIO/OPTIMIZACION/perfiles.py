"""Selección automática de carteras por nivel de riesgo (pregunta 3).

Perfiles DINÁMICOS: Bajo / Medio / Alto NO son volatilidades absolutas fijas;
se derivan de los percentiles (P20 / P50 / P80 por defecto) de la distribución
de volatilidad de la PROPIA frontera eficiente del universo actual. Se añade la
cartera de Máximo Sharpe. Cada candidato trae su descomposición de riesgo (MCR).

La volatilidad y el retorno son in-sample (estructural). El riesgo táctico
(vol T+1, VaR, CDaR, score) lo añaden después las capas RIESGO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import (
    Configuracion,
    MomentsResult,
    PortfolioCandidate,
    ResultadoFrontera,
)
from RIESGO.mcr import descomponer_riesgo


def _pesos_en_volatilidad(puntos: pd.DataFrame, activos: list[str], vol_objetivo: float) -> pd.Series:
    """Punto eficiente cuya volatilidad es la más cercana por encima del objetivo."""
    cols = [f"peso·{a}" for a in activos]
    vol = puntos["volatilidad"].to_numpy()
    candidatos = puntos[vol >= vol_objetivo - 1e-12]
    fila = candidatos.iloc[0] if not candidatos.empty else puntos.iloc[int(vol.argmax())]
    return pd.Series([float(fila[c]) for c in cols], index=activos)


def _candidato(
    nivel: str,
    pesos: pd.Series,
    momentos: MomentsResult,
    cfg: Configuracion,
) -> PortfolioCandidate:
    activos = list(momentos.cov_estructural.index)
    w = pesos.reindex(activos).to_numpy(dtype=float)
    mu = momentos.retornos_ajustados.reindex(activos).to_numpy(dtype=float)
    cov_e = momentos.cov_estructural.to_numpy(dtype=float)
    cov_t = momentos.cov_tactica.to_numpy(dtype=float)
    rf = cfg.tasa_libre_riesgo_anual

    ret = float(w @ mu)
    vol_e = float(np.sqrt(max(w @ cov_e @ w, 0.0)))
    vol_t = float(np.sqrt(max(w @ cov_t @ w, 0.0)))
    sharpe = (ret - rf) / vol_e if vol_e > 0 else 0.0
    descomp = descomponer_riesgo(pesos.reindex(activos), momentos.cov_tactica)
    return PortfolioCandidate(
        nivel=nivel,
        pesos=pesos.reindex(activos),
        retorno_esperado=ret,
        volatilidad_estructural=vol_e,
        volatilidad_tactica=vol_t,
        sharpe=float(sharpe),
        descomposicion=descomp,
    )


def seleccionar_perfiles(
    frontera: ResultadoFrontera,
    momentos: MomentsResult,
    cfg: Configuracion,
) -> tuple[PortfolioCandidate, ...]:
    puntos = frontera.puntos
    if puntos.empty:
        raise ErrorOptimizacion("OPTIMIZACION", "Frontera vacía: no hay perfiles que seleccionar.")
    activos = list(momentos.cov_estructural.index)
    vol = puntos["volatilidad"].to_numpy()

    candidatos: list[PortfolioCandidate] = []
    for nivel, percentil in cfg.percentiles_perfil:
        vol_obj = float(np.quantile(vol, percentil))
        pesos = _pesos_en_volatilidad(puntos, activos, vol_obj)
        candidatos.append(_candidato(nivel, pesos, momentos, cfg))

    candidatos.append(_candidato("max_sharpe", frontera.maximo_sharpe_pesos, momentos, cfg))
    return tuple(candidatos)


def curva_top_sharpe(frontera: ResultadoFrontera, ventana: int = 15) -> pd.DataFrame:
    """Estabilidad de pesos: evolución de la composición alrededor del máximo
    Sharpe de la frontera (las `ventana` carteras de mayor Sharpe, ordenadas por
    volatilidad). Sirve para juzgar si la cartera óptima es robusta o frágil."""
    puntos = frontera.puntos
    if puntos.empty:
        return pd.DataFrame()
    top = puntos.sort_values("sharpe", ascending=False).head(ventana)
    return top.sort_values("volatilidad").reset_index(drop=True)

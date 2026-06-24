"""Ensamblado del leaderboard multi-criterio con riesgo PRECISO en la shortlist.

Estrategia de rendimiento: la frontera (120+ carteras) se criba con métricas
paramétricas instantáneas; solo la SHORTLIST (unión de los Top-5 por criterio +
los candidatos principales) se confirma con FHS + Monte Carlo precisos. Así el
informe muestra cifras exactas donde importa y sigue siendo rápido aunque el
motor Rust caiga a fallback.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from CONTRATOS.modelos import (
    Configuracion,
    CriterioRanking,
    MomentsResult,
    PortfolioCandidate,
    PortfolioInput,
    ResultadoFrontera,
)
from OPTIMIZACION.exploracion import (
    CRITERIOS,
    construir_candidato,
    tabla_exploracion,
    top_por_criterio,
)
from RIESGO.forecast import calcular_forecast, calcular_simulacion
from RIESGO.score import calcular_score_cartera


def _firma(pesos: pd.Series) -> tuple:
    return tuple(np.round(pesos.to_numpy(dtype=float), 4))


def construir_exploracion(
    frontera: ResultadoFrontera,
    momentos: MomentsResult,
    entrada: PortfolioInput,
    cfg: Configuracion,
    perfiles: tuple[PortfolioCandidate, ...],
) -> dict:
    activos = list(momentos.cov_estructural.index)
    tabla, (b1, b2), vol_ms = tabla_exploracion(frontera, momentos, cfg)
    tops = top_por_criterio(tabla, n=5)

    # Candidatos baratos por cada fila Top-5 (con su clase), indexados por firma.
    cols_peso = [f"peso·{a}" for a in activos]
    por_firma: dict[tuple, PortfolioCandidate] = {}
    tops_candidatos: dict[str, list[PortfolioCandidate]] = {}
    for clave, _n, _d, _col, _s in CRITERIOS:
        lista = []
        for _, fila in tops[clave].iterrows():
            pesos = pd.Series([float(fila[c]) for c in cols_peso], index=activos)
            cand = construir_candidato(clave, pesos, momentos, cfg, str(fila["clase"]))
            firma = _firma(pesos)
            por_firma.setdefault(firma, cand)
            lista.append((firma, cand))
        tops_candidatos[clave] = lista

    # Shortlist = firmas de los Top-5 + las de los candidatos principales.
    for p in perfiles:
        por_firma.setdefault(_firma(p.pesos), p)

    # Riesgo PRECISO (FHS + Monte Carlo) solo en la shortlist.
    enriquecidos: dict[tuple, PortfolioCandidate] = {}
    for firma, cand in por_firma.items():
        fc = calcular_forecast(cand.pesos, entrada.log_retornos, momentos, cfg)
        sim = calcular_simulacion(cand.pesos, entrada.log_retornos, cfg)
        enriquecidos[firma] = replace(cand, forecast=fc, simulacion=sim)

    # Score multifactor preciso sobre la shortlist completa.
    firmas = list(enriquecidos)
    puntuados = calcular_score_cartera(tuple(enriquecidos[f] for f in firmas), cfg)
    enriquecidos = {f: c for f, c in zip(firmas, puntuados)}

    # Clave de reordenación por la métrica PRECISA (FHS/MC) dentro de cada Top-5.
    claves_precisas = {
        "sharpe": lambda c: c.sharpe,
        "score": lambda c: (c.score if c.score is not None else -1e18),
        "var99": lambda c: c.forecast.var_fhs_99,        # menos negativo = mejor
        "cdar": lambda c: c.simulacion.cdar_30d,          # menos negativo = mejor
        "starr": lambda c: (c.starr or 0.0),
        "diversificacion": lambda c: (c.diversificacion or 0.0),
    }

    # Leaderboard final: Top-5 por criterio, reordenado por la métrica precisa.
    leaderboard: list[CriterioRanking] = []
    for clave, nombre, desc, _col, sentido in CRITERIOS:
        cands = [enriquecidos[firma] for firma, _ in tops_candidatos[clave]]
        cands.sort(key=claves_precisas[clave], reverse=True)
        leaderboard.append(CriterioRanking(clave, nombre, desc, sentido, tuple(cands)))

    # Clasificación de la nube por banda de volatilidad (estructural).
    nube = frontera.nube_factible.copy()
    nube["clase"] = pd.cut(nube["volatilidad"], bins=[-np.inf, b1, b2, np.inf],
                           labels=["bajo", "medio", "alto"]).astype(str)

    cols_clasif = (["vol_struct", "retorno", "sharpe", "clase", "diversificacion", "starr", "var99"]
                   + [f"peso·{a}" for a in activos])
    clasif_frontera = tabla[cols_clasif].rename(columns={"vol_struct": "volatilidad"})

    vmin = float(frontera.puntos["volatilidad"].min())
    vmax = float(frontera.puntos["volatilidad"].max())
    anclas = (("Mínima varianza", vmin), ("Máx Sharpe", vol_ms), ("Máx retorno", vmax))

    return {
        "clasificacion_frontera": clasif_frontera,
        "clasificacion_nube": nube,
        "anclas": anclas,
        "leaderboard": tuple(leaderboard),
        "bandas": (b1, b2),
    }

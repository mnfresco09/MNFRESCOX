"""Estrategia Challenger: Nested Clustered Optimization."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import (
    Configuracion,
    MomentsResult,
    PortfolioCandidate,
    PortfolioInput,
    Restricciones,
    ResultadoFrontera,
)
from OPTIMIZACION import optimizador as opt

from .base import OptimizadorBase, construir_candidate_comun, proyectar_a_restricciones


class OptimizadorNCO(OptimizadorBase):
    """NCO jerárquico: optimiza dentro de clústeres y luego entre clústeres."""

    nombre = "NCO"

    def optimizar(
        self,
        entrada: PortfolioInput,
        momentos: MomentsResult,
        cfg: Configuracion,
        frontera: ResultadoFrontera | None = None,
    ) -> tuple[PortfolioCandidate, ...]:
        pesos = self._resolver_nco(entrada, momentos, cfg)
        return (
            construir_candidate_comun(
                nivel="nco",
                motor=self.nombre,
                pesos=pesos,
                entrada=entrada,
                momentos=momentos,
                cfg=cfg,
            ),
        )

    def _resolver_nco(
        self,
        entrada: PortfolioInput,
        momentos: MomentsResult,
        cfg: Configuracion,
    ) -> pd.Series:
        activos = list(momentos.cov_estructural.index)
        if len(activos) <= 2:
            return proyectar_a_restricciones(_pesos_equiponderados(activos), cfg)

        clusters = _clusters_por_correlacion(momentos.correlacion.reindex(index=activos, columns=activos))
        intra: dict[int, pd.Series] = {}
        retornos_cluster = {}
        simples = np.expm1(entrada.log_retornos.reindex(columns=activos))

        restr_intra = Restricciones(solo_largos=True, peso_maximo=1.0, peso_minimo=0.0)
        for cluster_id, miembros in clusters.items():
            cov_sub = momentos.cov_estructural.loc[miembros, miembros]
            try:
                w_sub = opt.minima_varianza(cov_sub, restr_intra)
            except (ErrorOptimizacion, ValueError, np.linalg.LinAlgError):
                w_sub = pd.Series(1.0 / len(miembros), index=miembros)
            intra[cluster_id] = w_sub
            retornos_cluster[f"C{cluster_id}"] = simples[miembros].to_numpy(dtype=float) @ w_sub.to_numpy(dtype=float)

        matriz_cluster = pd.DataFrame(retornos_cluster, index=entrada.log_retornos.index)
        if matriz_cluster.shape[1] == 1:
            unico = next(iter(intra.values()))
            return proyectar_a_restricciones(unico.reindex(activos).fillna(0.0), cfg)

        mu_cluster = matriz_cluster.mean() * float(cfg.dias_anio)
        cov_cluster = pd.DataFrame(
            np.cov(matriz_cluster.to_numpy(dtype=float), rowvar=False) * float(cfg.dias_anio),
            index=matriz_cluster.columns,
            columns=matriz_cluster.columns,
        )
        restr_cluster = Restricciones(solo_largos=True, peso_maximo=1.0, peso_minimo=0.0)
        try:
            w_cluster = opt.maximo_sharpe(mu_cluster, cov_cluster, cfg.tasa_libre_riesgo_anual, restr_cluster)
        except (ErrorOptimizacion, ValueError, np.linalg.LinAlgError):
            w_cluster = opt.minima_varianza(cov_cluster, restr_cluster)

        pesos = pd.Series(0.0, index=activos)
        for cluster_id, w_sub in intra.items():
            escala = float(w_cluster[f"C{cluster_id}"])
            pesos.loc[w_sub.index] = escala * w_sub
        return proyectar_a_restricciones(pesos, cfg)


def _pesos_equiponderados(activos: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(activos), index=activos)


def _clusters_por_correlacion(correlacion: pd.DataFrame) -> dict[int, list[str]]:
    activos = list(correlacion.index)
    corr = correlacion.to_numpy(dtype=float)
    corr = np.nan_to_num((corr + corr.T) / 2.0, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    distancia = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    z = linkage(squareform(distancia, checks=False), method="average")
    n_clusters = min(len(activos), max(2, int(round(math.sqrt(len(activos))))))
    etiquetas = fcluster(z, t=n_clusters, criterion="maxclust")
    clusters: dict[int, list[str]] = {}
    for activo, etiqueta in zip(activos, etiquetas):
        clusters.setdefault(int(etiqueta), []).append(activo)
    return clusters

"""HRP — Hierarchical Risk Parity (López de Prado, 2016).

Tres pasos, SIN invertir la matriz de covarianza (su gran ventaja frente a
Markowitz: evita amplificar el ruido de estimación):

  1. Clustering jerárquico de los activos por distancia de correlación.
  2. Cuasi-diagonalización: reordena la matriz para juntar los activos similares.
  3. Bisección recursiva: reparte el peso entre los dos sub-clústeres en proporción
     inversa a su varianza, bajando por el árbol.

Devuelve pesos solo-largos que suman 1. El tope por activo se aplica fuera
(en la fachada) renormalizando, porque HRP no lo incorpora de forma nativa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from CONTRATOS.errores import ErrorOptimizacion


def _correlacion_desde_covarianza(covarianza: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(covarianza.to_numpy(dtype=float)))
    if (d <= 0).any():
        raise ErrorOptimizacion("OPTIMIZACION", "HRP: volatilidad no positiva en la covarianza.")
    inv = np.outer(1.0 / d, 1.0 / d)
    corr = covarianza.to_numpy(dtype=float) * inv
    return pd.DataFrame(np.clip(corr, -1.0, 1.0), index=covarianza.index, columns=covarianza.index)


def _orden_cuasidiagonal(enlace: np.ndarray) -> list[int]:
    """Devuelve las hojas del dendrograma en orden (cuasi-diagonalización)."""
    enlace = enlace.astype(int)
    n = enlace.shape[0] + 1
    orden = [enlace[-1, 0], enlace[-1, 1]]
    while max(orden) >= n:
        nuevo: list[int] = []
        for idx in orden:
            if idx < n:
                nuevo.append(idx)
            else:
                fila = enlace[idx - n]
                nuevo.extend([int(fila[0]), int(fila[1])])
        orden = nuevo
    return orden


def _varianza_cluster(covarianza: np.ndarray, indices: list[int]) -> float:
    sub = covarianza[np.ix_(indices, indices)]
    inv_var = 1.0 / np.diag(sub)
    w = inv_var / inv_var.sum()             # cartera inversa-varianza dentro del clúster
    return float(w @ sub @ w)


def _biseccion_recursiva(covarianza: np.ndarray, orden: list[int]) -> np.ndarray:
    pesos = np.ones(len(orden))
    clusteres = [orden]
    while clusteres:
        nuevos: list[list[int]] = []
        for cluster in clusteres:
            if len(cluster) <= 1:
                continue
            mitad = len(cluster) // 2
            izquierda, derecha = cluster[:mitad], cluster[mitad:]
            var_izq = _varianza_cluster(covarianza, izquierda)
            var_der = _varianza_cluster(covarianza, derecha)
            alfa = 1.0 - var_izq / (var_izq + var_der)   # más peso al de menor varianza
            for i in izquierda:
                pesos[orden.index(i)] *= alfa
            for i in derecha:
                pesos[orden.index(i)] *= 1.0 - alfa
            nuevos.extend([izquierda, derecha])
        clusteres = nuevos
    return pesos


def hrp(covarianza: pd.DataFrame) -> pd.Series:
    activos = list(covarianza.index)
    if len(activos) < 2:
        raise ErrorOptimizacion("OPTIMIZACION", "HRP requiere al menos 2 activos.")
    corr = _correlacion_desde_covarianza(covarianza).to_numpy()
    distancia = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))   # distancia de correlación
    np.fill_diagonal(distancia, 0.0)
    enlace = linkage(squareform(distancia, checks=False), method="single")
    orden = _orden_cuasidiagonal(enlace)
    pesos = _biseccion_recursiva(covarianza.to_numpy(dtype=float), orden)
    serie = pd.Series(0.0, index=activos)
    for posicion, indice in enumerate(orden):
        serie.iloc[indice] = pesos[posicion]
    return serie / serie.sum()

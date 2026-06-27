"""CPCV — Combinatorial Purged Cross-Validation (López de Prado).

Por qué CPCV y no Walk-Forward simple: WFA da UNA trayectoria OOS; CPCV da una
**distribución** de Sharpes OOS a partir de múltiples particiones combinatorias.
Eso convierte "mi Sharpe es 2.1" en "mi Sharpe OOS es 1.3 ± 0.6", que es la única
afirmación honesta.

Adaptación a este sistema (no es un modelo ML supervisado, es una optimización de
parámetros): los datos se dividen en `N` grupos temporales; se eligen `k` grupos
como test; cada combinación optimiza con Optuna sobre los grupos de train y
evalúa SIN reoptimizar sobre los de test. Concatenando los tramos de test según
la estructura combinatoria se obtienen `φ = C(N,k)·k/N` trayectorias OOS
completas y, por tanto, una distribución de cada métrica.

Este módulo es puramente combinatorio (índices enteros): construye los grupos,
las particiones, aplica purge + embargo y describe las trayectorias. La ejecución
real (optimizar/evaluar) la hace `orquestador.py` con callbacks inyectados.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np


@dataclass(frozen=True)
class Grupo:
    """Bloque temporal contiguo [inicio, fin) de índices de observación."""

    indice: int
    inicio: int
    fin: int

    @property
    def rango(self) -> tuple[int, int]:
        return (self.inicio, self.fin)

    def __len__(self) -> int:
        return self.fin - self.inicio


@dataclass(frozen=True)
class FoldCPCV:
    """Una partición CPCV: qué grupos son test y qué índices quedan en train.

    `train_idx` ya viene PURGADO y con EMBARGO aplicado. `test_grupos` son los
    índices de grupo usados como test (para reconstruir las trayectorias OOS).
    """

    combinacion: int
    test_grupos: tuple[int, ...]
    train_idx: np.ndarray  # int64, ordenado, sin solapes con test
    test_rangos: tuple[tuple[int, int], ...]


def n_trayectorias(n_grupos: int, k: int) -> int:
    """Número de trayectorias OOS completas: φ = C(N,k)·k/N."""
    _validar_n_k(n_grupos, k)
    return comb(n_grupos, k) * k // n_grupos


def construir_grupos(n_obs: int, n_grupos: int) -> list[Grupo]:
    """Divide [0, n_obs) en `n_grupos` bloques contiguos casi iguales."""
    if n_obs < n_grupos:
        raise ValueError(f"n_obs ({n_obs}) debe ser >= n_grupos ({n_grupos}).")
    cortes = np.linspace(0, n_obs, n_grupos + 1, dtype=int)
    return [Grupo(i, int(cortes[i]), int(cortes[i + 1])) for i in range(n_grupos)]


def generar_folds(
    n_obs: int,
    *,
    n_grupos: int = 6,
    k: int = 2,
    embargo: float = 0.01,
    duracion_trade: int = 1,
) -> list[FoldCPCV]:
    """Genera todas las particiones CPCV con purge + embargo aplicados.

    Parameters
    ----------
    n_obs:
        Número total de observaciones (velas) del tramo de TRAIN/VALIDATION.
    n_grupos, k:
        N grupos temporales, k de ellos como test. C(N,k) combinaciones.
    embargo:
        Fracción de `n_obs` que se descarta del train inmediatamente DESPUÉS de
        cada bloque de test, para cortar la fuga por autocorrelación serial.
    duracion_trade:
        Nº máximo de velas que puede abarcar un trade. Se usa para el PURGE: se
        elimina del train toda vela cuyo trade pudiera solaparse con el test
        (un trade que entra en train y sale dentro del test filtra información).

    Returns
    -------
    list[FoldCPCV]
        Una por combinación, con `train_idx` ya purgado y con embargo.
    """
    _validar_n_k(n_grupos, k)
    if not (0.0 <= embargo < 1.0):
        raise ValueError("embargo debe estar en [0, 1).")
    if duracion_trade < 1:
        raise ValueError("duracion_trade debe ser >= 1.")

    grupos = construir_grupos(n_obs, n_grupos)
    embargo_velas = int(round(embargo * n_obs))
    folds: list[FoldCPCV] = []

    for c, combo in enumerate(combinations(range(n_grupos), k)):
        test_rangos = tuple(grupos[g].rango for g in combo)
        train_idx = _train_purgado(
            n_obs=n_obs,
            test_rangos=test_rangos,
            embargo_velas=embargo_velas,
            duracion_trade=duracion_trade,
        )
        folds.append(
            FoldCPCV(
                combinacion=c,
                test_grupos=tuple(combo),
                train_idx=train_idx,
                test_rangos=test_rangos,
            )
        )
    return folds


def indices_de_rangos(rangos: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Aplana una tupla de rangos [ini, fin) a un array de índices ordenado."""
    if not rangos:
        return np.empty(0, dtype=np.int64)
    partes = [np.arange(ini, fin, dtype=np.int64) for (ini, fin) in rangos]
    return np.concatenate(partes)


# ---------------------------------------------------------------------------
# Purge + embargo
# ---------------------------------------------------------------------------

def _train_purgado(
    *,
    n_obs: int,
    test_rangos: tuple[tuple[int, int], ...],
    embargo_velas: int,
    duracion_trade: int,
) -> np.ndarray:
    """Índices de train tras eliminar solapes (purge) y buffer posterior (embargo).

    - PURGE: una vela de train se descarta si un trade abierto en ella podría
      cerrarse dentro de un bloque de test. Como un trade abarca como mucho
      `duracion_trade` velas, se purga la ventana [ini_test − duracion_trade, fin_test).
    - EMBARGO: además se descartan `embargo_velas` velas justo DESPUÉS de cada
      bloque de test.
    """
    en_train = np.ones(n_obs, dtype=bool)
    for (ini, fin) in test_rangos:
        purge_ini = max(0, ini - (duracion_trade - 1))
        en_train[purge_ini:fin] = False
        if embargo_velas > 0:
            embargo_fin = min(n_obs, fin + embargo_velas)
            en_train[fin:embargo_fin] = False
    return np.nonzero(en_train)[0].astype(np.int64)


def _validar_n_k(n_grupos: int, k: int) -> None:
    if n_grupos < 2:
        raise ValueError("n_grupos debe ser >= 2.")
    if not (1 <= k < n_grupos):
        raise ValueError(f"k debe cumplir 1 <= k < n_grupos ({n_grupos}); recibido k={k}.")

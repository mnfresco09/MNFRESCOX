"""Función objetivo robusta y multiobjetivo (Fase 4).

Maximizar cualquier métrica in-sample hasta su pico es la definición de
sobreajuste. Este módulo aporta tres herramientas para combatirlo:

  1. Penalizaciones del score escalar
     - `penalizacion_turnover`    → penaliza el exceso de operaciones (coste).
     - `penalizacion_complejidad` → penaliza el nº de parámetros libres.
     Se aplican en la función objetivo del runner, con factor 0 por defecto
     (sin efecto salvo que se activen en config).

  2. Frente de Pareto (NSGA-II), cuando `OPTUNA_MULTIOBJETIVO=True`
     - `vector_pareto`       → objetivos (PSR ↑, max drawdown ↓, turnover ↓).
       Se usa el PSR y no el Sharpe crudo por fiabilidad estadística: corrige
       tamaño de muestra, asimetría y curtosis.
     - `DIRECCIONES_PARETO`  → direcciones para `create_study(directions=...)`.

  3. Selección por MESETA (no por pico)
     - `seleccionar_meseta`  → entre los puntos del frente de Pareto, elige el
       que vive en la región más estable del espacio de parámetros (mediana del
       rendimiento de sus vecinos), NO el de score máximo. Un edge robusto vive
       en una meseta plana; un artefacto de sobreajuste, en un pico estrecho.

Funciones puras (stdlib + NumPy): verificables de forma aislada.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# Objetivos del frente de Pareto y sus direcciones (orden alineado con vector_pareto).
NOMBRES_PARETO = ("psr", "max_drawdown", "turnover")
DIRECCIONES_PARETO = ("maximize", "minimize", "minimize")


# ---------------------------------------------------------------------------
# Penalizaciones del score escalar
# ---------------------------------------------------------------------------

def penalizacion_turnover(
    score: float,
    n_trades: int,
    *,
    trades_objetivo: int = 100,
    factor: float = 0.0,
) -> float:
    """Resta al score una penalización proporcional al exceso de operaciones."""
    if factor <= 0.0 or trades_objetivo <= 0:
        return float(score)
    exceso = max(0, int(n_trades) - int(trades_objetivo))
    return float(score) - factor * (exceso / float(trades_objetivo))


def penalizacion_complejidad(score: float, n_parametros: int, *, factor: float = 0.0) -> float:
    """Resta una penalización lineal por número de parámetros libres."""
    if factor <= 0.0:
        return float(score)
    return float(score) - factor * max(0, int(n_parametros))


# ---------------------------------------------------------------------------
# Frente de Pareto (multiobjetivo)
# ---------------------------------------------------------------------------

def vector_pareto(metricas: dict) -> tuple[float, float, float]:
    """Vector objetivo (PSR ↑, max_drawdown ↓, turnover ↓) desde el dict de métricas.

    PSR (Probabilistic Sharpe Ratio) como rentabilidad ajustada por fiabilidad
    estadística; max drawdown como riesgo; turnover (operaciones/día, o total si
    no hay frecuencia) como proxy de coste y de grados de libertad.
    """
    psr = _num(metricas.get("psr"))
    drawdown = abs(_num(metricas.get("max_drawdown")))
    turnover = _num(metricas.get("trades_por_dia", metricas.get("total_trades")))
    return (psr, drawdown, turnover)


# ---------------------------------------------------------------------------
# Selección por meseta sobre el frente de Pareto
# ---------------------------------------------------------------------------

def seleccionar_meseta(
    candidatos: Sequence[Any],
    universo: Sequence[Any],
    *,
    valor: Callable[[Any], float],
    parametros: Callable[[Any], dict],
    k: int = 7,
):
    """Elige, entre `candidatos` (frente de Pareto), el de meseta más robusta.

    Para cada candidato se mide la MEDIANA del `valor` (p. ej. PSR) de sus `k`
    vecinos más cercanos en el espacio de parámetros normalizado del `universo`
    completo de trials. El candidato cuya vecindad rinde mejor de forma estable
    (meseta) gana; así se evita el pico estrecho que solo funciona en un punto.

    Devuelve el objeto candidato elegido. Si no hay parámetros numéricos o muy
    pocos trials, cae con elegancia al candidato de mayor `valor`.
    """
    candidatos = list(candidatos)
    universo = list(universo)
    if not candidatos:
        raise ValueError("seleccionar_meseta: no hay candidatos.")
    if len(candidatos) == 1:
        return candidatos[0]

    claves = _claves_numericas(universo, parametros)
    if not claves or len(universo) < 2:
        return max(candidatos, key=valor)

    matriz_univ = _matriz_parametros(universo, parametros, claves)
    matriz_univ = _normalizar(matriz_univ)
    valores_univ = np.array([_num(valor(t)) for t in universo], dtype=np.float64)

    indice_univ = {id(t): i for i, t in enumerate(universo)}
    k_efec = max(1, min(int(k), len(universo)))

    mejor_obj = None
    mejor_meseta = -np.inf
    for cand in candidatos:
        i = indice_univ.get(id(cand))
        if i is None:
            # Candidato no presente en el universo: usa su propio valor.
            meseta = _num(valor(cand))
        else:
            distancias = np.sqrt(((matriz_univ - matriz_univ[i]) ** 2).sum(axis=1))
            vecinos = np.argsort(distancias, kind="stable")[:k_efec]
            meseta = float(np.median(valores_univ[vecinos]))
        if meseta > mejor_meseta:
            mejor_meseta = meseta
            mejor_obj = cand
    return mejor_obj if mejor_obj is not None else max(candidatos, key=valor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claves_numericas(trials: Sequence[Any], parametros: Callable[[Any], dict]) -> list[str]:
    claves: set[str] = set()
    for t in trials:
        for k, v in (parametros(t) or {}).items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                claves.add(k)
    return sorted(claves)


def _matriz_parametros(
    trials: Sequence[Any], parametros: Callable[[Any], dict], claves: list[str]
) -> np.ndarray:
    filas = []
    for t in trials:
        p = parametros(t) or {}
        filas.append([_num(p.get(c)) for c in claves])
    return np.asarray(filas, dtype=np.float64)


def _normalizar(matriz: np.ndarray) -> np.ndarray:
    if matriz.size == 0:
        return matriz
    minimos = matriz.min(axis=0)
    maximos = matriz.max(axis=0)
    rango = maximos - minimos
    rango[rango == 0.0] = 1.0  # columnas constantes no aportan distancia
    return (matriz - minimos) / rango


def _num(valor) -> float:
    try:
        f = float(valor)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0

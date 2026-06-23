"""Markowitz sobre covarianza Ledoit-Wolf: mínima varianza, máximo Sharpe y
retorno-objetivo, más los puntos de la frontera eficiente.

Se usa la covarianza encogida (Ledoit-Wolf), nunca la muestral cruda. La
optimización es por SLSQP (scipy) con restricción de suma 1 y cotas por activo,
de modo que solo-largos y peso máximo se respetan dentro del propio optimizador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import Restricciones

from .comun import limites


def _resolver(
    objetivo,
    n: int,
    cotas: list[tuple[float, float]],
    restricciones_extra: list[dict],
) -> np.ndarray:
    inicial = np.full(n, 1.0 / n)
    restricciones = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *restricciones_extra]
    resultado = minimize(
        objetivo,
        inicial,
        method="SLSQP",
        bounds=cotas,
        constraints=restricciones,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not resultado.success:
        raise ErrorOptimizacion("OPTIMIZACION", f"Markowitz no convergió: {resultado.message}")
    return resultado.x


def minima_varianza(covarianza: pd.DataFrame, restricciones: Restricciones) -> pd.Series:
    cov = covarianza.to_numpy(dtype=float)
    n = cov.shape[0]
    w = _resolver(lambda w: w @ cov @ w, n, limites(restricciones, n), [])
    return pd.Series(w, index=covarianza.index)


def maximo_sharpe(
    retornos_esperados: pd.Series,
    covarianza: pd.DataFrame,
    tasa_libre_riesgo: float,
    restricciones: Restricciones,
) -> pd.Series:
    cov = covarianza.to_numpy(dtype=float)
    mu = retornos_esperados.reindex(covarianza.index).to_numpy(dtype=float)
    n = cov.shape[0]

    def negativo_sharpe(w):
        vol = np.sqrt(max(w @ cov @ w, 1e-18))
        return -((w @ mu - tasa_libre_riesgo) / vol)

    w = _resolver(negativo_sharpe, n, limites(restricciones, n), [])
    return pd.Series(w, index=covarianza.index)


def retorno_objetivo(
    retornos_esperados: pd.Series,
    covarianza: pd.DataFrame,
    objetivo_anual: float,
    restricciones: Restricciones,
) -> pd.Series:
    cov = covarianza.to_numpy(dtype=float)
    mu = retornos_esperados.reindex(covarianza.index).to_numpy(dtype=float)
    n = cov.shape[0]
    cotas = limites(restricciones, n)

    # Comprueba que el objetivo es alcanzable con estas cotas antes de optimizar.
    ret_min, ret_max = _rango_retorno_alcanzable(mu, cotas)
    if not ret_min - 1e-9 <= objetivo_anual <= ret_max + 1e-9:
        raise ErrorOptimizacion(
            "OPTIMIZACION",
            f"El retorno objetivo {objetivo_anual:.3f} es inalcanzable con las "
            f"restricciones (rango factible [{ret_min:.3f}, {ret_max:.3f}]).",
        )
    restr = [{"type": "eq", "fun": lambda w: w @ mu - objetivo_anual}]
    w = _resolver(lambda w: w @ cov @ w, n, cotas, restr)
    return pd.Series(w, index=covarianza.index)


def puntos_frontera(
    retornos_esperados: pd.Series,
    covarianza: pd.DataFrame,
    restricciones: Restricciones,
    n_puntos: int = 40,
) -> pd.DataFrame:
    """Frontera eficiente: para una rejilla de retornos objetivo, la mínima
    volatilidad alcanzable. Devuelve columnas [retorno, volatilidad] y un peso
    por activo (columna 'peso·<activo>'), para poder mostrar la composición de
    cada punto al hacer clic en el informe."""
    activos = list(covarianza.index)
    cov = covarianza.to_numpy(dtype=float)
    mu = retornos_esperados.reindex(covarianza.index).to_numpy(dtype=float)
    n = cov.shape[0]
    cotas = limites(restricciones, n)

    ret_mv = float(minima_varianza(covarianza, restricciones) @ mu)
    _, ret_max = _rango_retorno_alcanzable(mu, cotas)
    objetivos = np.linspace(ret_mv, ret_max, n_puntos)

    filas = []
    for objetivo in objetivos:
        try:
            restr = [{"type": "eq", "fun": (lambda obj: (lambda w: w @ mu - obj))(objetivo)}]
            w = _resolver(lambda w: w @ cov @ w, n, cotas, restr)
            fila = {"retorno": float(w @ mu), "volatilidad": float(np.sqrt(w @ cov @ w))}
            fila.update({f"peso·{activo}": float(peso) for activo, peso in zip(activos, w)})
            filas.append(fila)
        except ErrorOptimizacion:
            continue
    if not filas:
        raise ErrorOptimizacion("OPTIMIZACION", "No se pudo construir la frontera eficiente.")
    return pd.DataFrame(filas)


def _rango_retorno_alcanzable(mu: np.ndarray, cotas: list[tuple[float, float]]) -> tuple[float, float]:
    """Retorno mínimo y máximo de una cartera que suma 1 dentro de las cotas
    (problema lineal resuelto por orden de mu, suficiente con cotas idénticas)."""
    inf = np.array([c[0] for c in cotas])
    sup = np.array([c[1] for c in cotas])

    def extremo(maximizar: bool) -> float:
        orden = np.argsort(mu)[::-1] if maximizar else np.argsort(mu)
        w = inf.copy()
        restante = 1.0 - inf.sum()
        for i in orden:
            paso = min(sup[i] - inf[i], restante)
            w[i] += paso
            restante -= paso
            if restante <= 1e-15:
                break
        return float(w @ mu)

    return extremo(False), extremo(True)

"""Utilidades compartidas por los asignadores: restricciones, métricas y validación.

No contiene ningún método de asignación; solo el andamiaje común para que todos
respeten las mismas restricciones (solo-largos, peso máximo) y se midan igual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import MetricasEstimadas, Restricciones, ResultadoAsignacion

TOLERANCIA_PESOS = 1e-6


def limites(restricciones: Restricciones, n_activos: int) -> list[tuple[float, float]]:
    """Cotas por activo según las restricciones."""
    inferior = 0.0 if restricciones.solo_largos else -1.0
    superior = restricciones.peso_maximo if restricciones.peso_maximo is not None else 1.0
    return [(inferior, superior)] * n_activos


def validar_pesos(pesos: pd.Series, restricciones: Restricciones, metodo: str) -> None:
    """Comprueba que los pesos suman 1 y respetan las restricciones. Si no, detiene."""
    w = pesos.to_numpy(dtype=float)
    if not np.isfinite(w).all():
        raise ErrorOptimizacion("OPTIMIZACION", f"{metodo}: pesos no finitos.")
    if abs(w.sum() - 1.0) > 1e-4:
        raise ErrorOptimizacion("OPTIMIZACION", f"{metodo}: los pesos no suman 1 (suma={w.sum():.6f}).")
    if restricciones.solo_largos and (w < -TOLERANCIA_PESOS).any():
        raise ErrorOptimizacion("OPTIMIZACION", f"{metodo}: hay pesos negativos con solo-largos activo.")
    if restricciones.peso_maximo is not None and (w > restricciones.peso_maximo + 1e-4).any():
        raise ErrorOptimizacion(
            "OPTIMIZACION",
            f"{metodo}: algún peso supera el máximo {restricciones.peso_maximo}.",
        )


def metricas_estimadas(
    pesos: pd.Series,
    retornos_esperados: pd.Series,
    covarianza: pd.DataFrame,
    tasa_libre_riesgo: float,
) -> MetricasEstimadas:
    """Retorno, volatilidad y Sharpe anualizados que la cartera tendría según
    los momentos estimados (in-sample). El juicio honesto out-of-sample lo da
    el walk-forward de la capa RIESGO."""
    w = pesos.reindex(covarianza.index).to_numpy(dtype=float)
    mu = retornos_esperados.reindex(covarianza.index).to_numpy(dtype=float)
    cov = covarianza.to_numpy(dtype=float)
    retorno = float(w @ mu)
    varianza = float(w @ cov @ w)
    volatilidad = float(np.sqrt(max(varianza, 0.0)))
    sharpe = (retorno - tasa_libre_riesgo) / volatilidad if volatilidad > 0 else 0.0
    return MetricasEstimadas(retorno_anual=retorno, volatilidad_anual=volatilidad, sharpe=float(sharpe))


def aplicar_tope_y_renormalizar(
    pesos: pd.Series,
    restricciones: Restricciones,
    metodo: str,
) -> tuple[pd.Series, tuple[str, ...]]:
    """Aplica el tope por activo a pesos long-only (p. ej. HRP) y renormaliza.

    Reparte iterativamente el exceso de los activos topados entre los no topados.
    Devuelve (pesos, advertencias). Solo para métodos que no incorporan el tope
    en su propia optimización.
    """
    advertencias: list[str] = []
    if restricciones.peso_maximo is None:
        return pesos / pesos.sum(), ()
    tope = restricciones.peso_maximo
    w = pesos.clip(lower=0.0).to_numpy(dtype=float)
    w = w / w.sum()
    for _ in range(1000):
        exceso = np.clip(w - tope, 0.0, None)
        if exceso.sum() <= 1e-12:
            break
        w = np.minimum(w, tope)
        libres = w < tope - 1e-12
        if not libres.any():
            raise ErrorOptimizacion(
                "OPTIMIZACION", f"{metodo}: el tope {tope} hace imposible sumar 1."
            )
        w[libres] += exceso.sum() * (w[libres] / w[libres].sum())
    if (pesos.to_numpy() > tope + 1e-9).any():
        advertencias.append(f"{metodo}: pesos topados al máximo {tope} y renormalizados.")
    return pd.Series(w, index=pesos.index), tuple(advertencias)


def asignacion(
    nombre: str,
    pesos: pd.Series,
    retornos_esperados: pd.Series,
    covarianza: pd.DataFrame,
    tasa_libre_riesgo: float,
    restricciones: Restricciones,
    estado_solver: str,
    diagnostico: str,
    advertencias: tuple[str, ...] = (),
) -> ResultadoAsignacion:
    """Empaqueta y VALIDA una asignación antes de devolverla."""
    pesos = pesos.reindex(covarianza.index).astype(float)
    validar_pesos(pesos, restricciones, nombre)
    return ResultadoAsignacion(
        nombre=nombre,
        pesos=pesos,
        metricas=metricas_estimadas(pesos, retornos_esperados, covarianza, tasa_libre_riesgo),
        estado_solver=estado_solver,
        diagnostico=diagnostico,
        advertencias=advertencias,
    )

"""Nube Monte Carlo de carteras aleatorias factibles, para el reporte.

Genera miles de carteras al azar que respetan las restricciones y calcula su
retorno/volatilidad/Sharpe. Sirve de telón de fondo para ver dónde caen los 6
métodos respecto al universo de combinaciones posibles y la frontera eficiente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import Restricciones, ResultadoMonteCarlo


def nube_montecarlo(
    retornos_esperados: pd.Series,
    covarianza: pd.DataFrame,
    restricciones: Restricciones,
    n_carteras: int,
    semilla: int,
    tasa_libre_riesgo: float,
) -> ResultadoMonteCarlo:
    activos = list(covarianza.index)
    n = len(activos)
    rng = np.random.default_rng(semilla)
    mu = retornos_esperados.reindex(activos).to_numpy(dtype=float)
    cov = covarianza.to_numpy(dtype=float)
    tope = restricciones.peso_maximo

    if restricciones.solo_largos:
        # Dirichlet → pesos no negativos que suman 1. Si hay tope, rechazo simple.
        muestras = rng.dirichlet(np.ones(n), size=n_carteras * 3)
        if tope is not None:
            muestras = muestras[(muestras <= tope + 1e-12).all(axis=1)]
        if muestras.shape[0] == 0:
            raise ErrorOptimizacion("OPTIMIZACION", "Monte Carlo no generó carteras factibles.")
        muestras = muestras[:n_carteras]
    else:
        bruto = rng.normal(size=(n_carteras, n))
        muestras = bruto / bruto.sum(axis=1, keepdims=True)

    retornos = muestras @ mu
    varianzas = np.einsum("ij,jk,ik->i", muestras, cov, muestras)
    volatilidades = np.sqrt(np.clip(varianzas, 0.0, None))
    sharpe = np.where(volatilidades > 0, (retornos - tasa_libre_riesgo) / volatilidades, 0.0)

    pesos_df = pd.DataFrame(muestras, columns=activos)
    metricas_df = pd.DataFrame(
        {"retorno": retornos, "volatilidad": volatilidades, "sharpe": sharpe}
    )
    return ResultadoMonteCarlo(pesos=pesos_df, metricas=metricas_df)

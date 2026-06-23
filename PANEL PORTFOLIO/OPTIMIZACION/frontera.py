"""Frontera eficiente de Markowitz y carteras diagnósticas."""

from __future__ import annotations

import pandas as pd

from CONTRATOS.modelos import Configuracion, ResultadoFrontera

from . import markowitz
from .comun import asignacion


def construir_frontera(
    retornos_esperados: pd.Series,
    covarianza: pd.DataFrame,
    configuracion: Configuracion,
) -> ResultadoFrontera:
    restr = configuracion.restricciones
    rf = configuracion.tasa_libre_riesgo_anual

    mv = asignacion(
        "Mínima varianza",
        markowitz.minima_varianza(covarianza, restr),
        retornos_esperados, covarianza, rf, restr,
        "SLSQP", "Cartera de mínima varianza global.",
    )
    ms = asignacion(
        "Markowitz (máx Sharpe)",
        markowitz.maximo_sharpe(retornos_esperados, covarianza, rf, restr),
        retornos_esperados, covarianza, rf, restr,
        "SLSQP", "Cartera tangente de máximo Sharpe sobre Ledoit-Wolf.",
    )
    puntos = markowitz.puntos_frontera(retornos_esperados, covarianza, restr)
    fila_max = puntos.loc[puntos["retorno"].idxmax()]
    pesos_max = pd.Series(
        {activo: float(fila_max[f"peso·{activo}"]) for activo in covarianza.index}
    )
    mr = asignacion(
        "Markowitz (máx retorno factible)",
        pesos_max,
        retornos_esperados, covarianza, rf, restr,
        "frontera",
        "Cartera diagnóstica de máximo retorno factible dentro de las restricciones actuales.",
    )
    return ResultadoFrontera(puntos=puntos, minima_varianza=mv, maximo_sharpe=ms, maximo_retorno_factible=mr)

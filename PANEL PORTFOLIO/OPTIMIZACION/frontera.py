"""Frontera eficiente de Markowitz y sus 3 carteras objetivo."""

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
    ro = asignacion(
        "Markowitz (retorno objetivo)",
        markowitz.retorno_objetivo(retornos_esperados, covarianza, configuracion.retorno_objetivo_anual, restr),
        retornos_esperados, covarianza, rf, restr,
        "SLSQP", f"Mínima varianza para un retorno objetivo de {configuracion.retorno_objetivo_anual:.1%}.",
    )
    puntos = markowitz.puntos_frontera(retornos_esperados, covarianza, restr)
    return ResultadoFrontera(puntos=puntos, minima_varianza=mv, maximo_sharpe=ms, retorno_objetivo=ro)

"""Selección de carteras eficientes por nivel de riesgo.

Dado el conjunto de puntos de la frontera eficiente (markowitz.puntos_frontera,
que ya trae los pesos por punto), aquí se eligen las carteras para distintos
niveles de volatilidad. Funciones puras: solo leen la frontera; no calculan
métricas de riesgo (eso es de la capa RIESGO).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorOptimizacion


def _columnas_peso(activos) -> list[str]:
    return [f"peso·{a}" for a in activos]


def rango_volatilidad(puntos: pd.DataFrame) -> tuple[float, float]:
    return float(puntos["volatilidad"].min()), float(puntos["volatilidad"].max())


def volatilidad_de_fraccion(vol_min: float, vol_max: float, fraccion: float) -> float:
    """Volatilidad objetivo como fracción del rango [min, max] de la frontera."""
    fraccion = min(max(fraccion, 0.0), 1.0)
    return vol_min + fraccion * (vol_max - vol_min)


def cartera_por_volatilidad(
    puntos: pd.DataFrame,
    activos,
    volatilidad_objetivo: float,
) -> tuple[pd.Series, float, float]:
    """Devuelve (pesos, retorno, volatilidad) del punto eficiente más cercano.

    'Eficiente' = entre los puntos cuya volatilidad es >= objetivo se toma el de
    menor volatilidad (el más eficiente que alcanza ese riesgo); si el objetivo
    supera el máximo, se toma el punto de máxima volatilidad.
    """
    if puntos.empty:
        raise ErrorOptimizacion("OPTIMIZACION", "Frontera vacía: no hay carteras por nivel de riesgo.")
    cols = _columnas_peso(activos)
    vol = puntos["volatilidad"].to_numpy()
    candidatos = puntos[vol >= volatilidad_objetivo - 1e-12]
    fila = candidatos.iloc[0] if not candidatos.empty else puntos.iloc[vol.argmax()]
    pesos = pd.Series([float(fila[c]) for c in cols], index=list(activos))
    return pesos, float(fila["retorno"]), float(fila["volatilidad"])


def tabla_niveles(puntos: pd.DataFrame, activos, n_niveles: int) -> pd.DataFrame:
    """Pesos en n niveles escalonados de la frontera (min-var -> máx retorno).

    Índice = volatilidad anual del nivel; columnas = retorno + un peso por activo.
    Sirve para ver cómo cambia la composición al subir el riesgo (área apilada).
    """
    vol_min, vol_max = rango_volatilidad(puntos)
    filas = {}
    for fraccion in np.linspace(0.0, 1.0, n_niveles):
        objetivo = volatilidad_de_fraccion(vol_min, vol_max, fraccion)
        pesos, retorno, vol = cartera_por_volatilidad(puntos, activos, objetivo)
        filas[round(vol, 6)] = {"retorno": retorno, **{a: float(pesos[a]) for a in activos}}
    tabla = pd.DataFrame.from_dict(filas, orient="index").sort_index()
    tabla.index.name = "volatilidad"
    return tabla

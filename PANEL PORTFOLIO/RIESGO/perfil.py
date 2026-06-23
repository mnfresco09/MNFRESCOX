"""Carteras recomendadas por nivel de riesgo (resultado principal del panel).

Combina la selección de puntos de la frontera (OPTIMIZACION.perfil_riesgo) con
la medición del riesgo de esos pesos fijos sobre TODA la muestra alineada
(VaR/CVaR/drawdown). El retorno y la volatilidad esperados son in-sample (de la
frontera); las métricas de riesgo describen cómo se habrían comportado esos
pesos fijos en el pasado. No es walk-forward: se etiqueta como histórico.
"""

from __future__ import annotations

import numpy as np

from CONFIGURACION import _tecnico
from CONTRATOS.modelos import (
    CarteraNivel,
    Configuracion,
    DatosAlineados,
    ResultadoFrontera,
    ResultadoPerfilRiesgo,
)
from OPTIMIZACION.perfil_riesgo import (
    cartera_por_volatilidad,
    rango_volatilidad,
    tabla_niveles,
    volatilidad_de_fraccion,
)

from .metricas import metricas_cartera


def _metricas_historicas(datos, pesos, configuracion):
    """Métricas de riesgo de unos pesos fijos rebalanceados a diario sobre la muestra."""
    simples = np.expm1(datos.log_retornos)
    serie = (simples * pesos.reindex(simples.columns)).sum(axis=1)
    return metricas_cartera(
        serie, configuracion.nivel_confianza,
        configuracion.tasa_libre_riesgo_anual, configuracion.dias_anio,
    )


def evaluar_perfil(
    datos: DatosAlineados,
    frontera: ResultadoFrontera,
    configuracion: Configuracion,
) -> ResultadoPerfilRiesgo:
    puntos = frontera.puntos
    activos = list(datos.activos)
    vol_min, vol_max = rango_volatilidad(puntos)

    objetivos: list[tuple[str, float]] = []
    for nivel, fraccion in _tecnico.FRACCION_VOL_NIVEL.items():
        objetivos.append((nivel, volatilidad_de_fraccion(vol_min, vol_max, fraccion)))
    if configuracion.volatilidad_objetivo is not None:
        objetivos.append(("personalizado", float(configuracion.volatilidad_objetivo)))

    carteras: list[CarteraNivel] = []
    for nivel, vol_obj in objetivos:
        pesos, retorno, vol = cartera_por_volatilidad(puntos, activos, vol_obj)
        carteras.append(CarteraNivel(
            nivel=nivel,
            volatilidad_objetivo=float(vol_obj),
            pesos=pesos,
            retorno_esperado=retorno,
            volatilidad_esperada=vol,
            metricas_historicas=_metricas_historicas(datos, pesos, configuracion),
        ))

    por_nivel = {cartera.nivel: cartera for cartera in carteras}
    recomendada = por_nivel[configuracion.perfil_riesgo]
    niveles_frontera = tabla_niveles(puntos, activos, _tecnico.N_NIVELES_RIESGO)
    return ResultadoPerfilRiesgo(
        carteras=tuple(carteras),
        recomendada=recomendada,
        niveles_frontera=niveles_frontera,
    )

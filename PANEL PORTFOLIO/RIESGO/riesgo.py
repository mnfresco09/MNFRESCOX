"""Fachada de RIESGO: walk-forward + métricas OOS + regímenes + stress + crisis."""

from __future__ import annotations

from CONTRATOS.modelos import (
    Configuracion,
    DatosAlineados,
    ResultadoAnalisis,
    ResultadoRiesgo,
)
from OPTIMIZACION.asignadores import METODOS

from .convexidad import analizar_convexidad
from .metricas import metricas_cartera
from .regimenes_riesgo import diversificacion_en_crisis, metricas_por_regimen
from .stress import evaluar_stress
from .walk_forward import ejecutar_walk_forward


def evaluar_riesgo(
    datos: DatosAlineados,
    analisis: ResultadoAnalisis,
    configuracion: Configuracion,
) -> ResultadoRiesgo:
    walk_forward = ejecutar_walk_forward(datos, configuracion)

    metricas = {
        metodo: metricas_cartera(
            walk_forward.retornos[metodo],
            configuracion.nivel_confianza,
            configuracion.tasa_libre_riesgo_anual,
            configuracion.dias_anio,
        )
        for metodo in METODOS
    }

    por_regimen = metricas_por_regimen(
        walk_forward.retornos, analisis.regimenes, configuracion.dias_anio
    )
    stress = evaluar_stress(walk_forward.retornos, configuracion)

    # Pesos representativos = últimos pesos derivados de cada método.
    pesos_finales = {
        metodo: walk_forward.pesos_diarios[metodo].iloc[-1] for metodo in METODOS
    }
    diversificacion_crisis = diversificacion_en_crisis(
        datos, analisis.regimenes, pesos_finales, configuracion
    )

    convexidad = analizar_convexidad(
        walk_forward.retornos, datos.log_retornos, configuracion.activo_referencia
    )

    return ResultadoRiesgo(
        walk_forward=walk_forward,
        metricas=metricas,
        metricas_por_regimen=por_regimen,
        stress=stress,
        diversificacion_crisis=diversificacion_crisis,
        convexidad=convexidad,
    )

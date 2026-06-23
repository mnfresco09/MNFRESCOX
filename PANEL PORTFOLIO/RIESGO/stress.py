"""Stress testing histórico sobre episodios de crisis conocidos.

Evalúa cada cartera del walk-forward (OOS) sobre ventanas como 2008, COVID-2020 o
2022. HONESTIDAD: un episodio solo se evalúa si hay cobertura out-of-sample real
en esas fechas; si el histórico/ventana de estimación no llega (p. ej. 2008 con
datos que empiezan en 2019), se marca como NO evaluable en vez de inventar nada.
"""

from __future__ import annotations

import pandas as pd

from CONTRATOS.modelos import Configuracion, ResultadoStress

from .metricas import metricas_cartera

MIN_OBS_STRESS = 15   # mínimo de días OOS dentro de la ventana para evaluarla


def evaluar_stress(
    retornos: pd.DataFrame,
    configuracion: Configuracion,
) -> dict[str, ResultadoStress]:
    resultado: dict[str, ResultadoStress] = {}
    for ventana in configuracion.ventanas_stress:
        mascara = (retornos.index >= ventana.inicio) & (retornos.index <= ventana.fin)
        sub = retornos.loc[mascara]
        observaciones = int(len(sub))
        evaluable = observaciones >= MIN_OBS_STRESS
        metricas = {}
        if evaluable:
            for metodo in retornos.columns:
                metricas[metodo] = metricas_cartera(
                    sub[metodo],
                    configuracion.nivel_confianza,
                    configuracion.tasa_libre_riesgo_anual,
                    configuracion.dias_anio,
                )
        resultado[ventana.nombre] = ResultadoStress(
            nombre=ventana.nombre,
            evaluable=evaluable,
            observaciones=observaciones,
            metricas=metricas,
        )
    return resultado

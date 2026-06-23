"""Fachada de la capa ANALISIS.

Ensambla el `ResultadoAnalisis` (contrato) a partir de los datos alineados:
momentos, covarianza Ledoit-Wolf, correlación media y de cola, PCA y régimen.
No optimiza ni reporta; solo describe los datos.
"""

from __future__ import annotations

import pandas as pd

from CONTRATOS.modelos import Configuracion, DatosAlineados, ResultadoAnalisis

from .correlacion import (
    correlacion_cola,
    correlacion_media,
    diferencia_cola_menos_media,
)
from .momentos import covarianza_ledoit_wolf, retornos_esperados, volatilidades
from .pca import calcular_pca
from .regimenes import etiquetar_regimenes

# Decil peor del activo de referencia para la correlación de cola.
PERCENTIL_COLA = 0.10


def analizar(datos: DatosAlineados, configuracion: Configuracion) -> ResultadoAnalisis:
    log_retornos = datos.log_retornos

    retornos_esp = retornos_esperados(log_retornos, configuracion.dias_anio)
    covarianza, _shrinkage = covarianza_ledoit_wolf(log_retornos, configuracion.dias_anio)
    vols = volatilidades(covarianza)

    corr_media = correlacion_media(log_retornos)
    corr_cola, observaciones_cola = correlacion_cola(
        log_retornos, configuracion.activo_referencia, PERCENTIL_COLA
    )
    diferencia = diferencia_cola_menos_media(corr_cola, corr_media)

    pca = calcular_pca(log_retornos)

    # Régimen sobre el activo de referencia, recortado al calendario de retornos.
    regimenes = etiquetar_regimenes(
        datos.cierres[configuracion.activo_referencia],
        configuracion.parametros_regimen,
    ).reindex(log_retornos.index)

    return ResultadoAnalisis(
        log_retornos=log_retornos,
        retornos_esperados=retornos_esp,
        covarianza=covarianza,
        volatilidades=vols,
        correlacion_media=corr_media,
        correlacion_cola=corr_cola,
        diferencia_correlacion_cola=diferencia,
        observaciones_cola=observaciones_cola,
        pca=pca,
        regimenes=regimenes,
    )

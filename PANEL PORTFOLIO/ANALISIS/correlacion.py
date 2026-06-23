"""Correlación media y correlación condicional de cola.

La correlación MEDIA esconde el riesgo real: muchos activos diversifican en
calma pero se sincronizan en las caídas. Por eso calculamos también la
correlación CONDICIONAL DE COLA, usando solo los días del peor decil de
retornos del activo de referencia (p. ej. ^GSPC). La diferencia (cola − media)
desenmascara los activos cuya diversificación se evapora justo cuando importa.
"""

from __future__ import annotations

import pandas as pd

from CONTRATOS.errores import ErrorAnalisis


def correlacion_media(log_retornos: pd.DataFrame) -> pd.DataFrame:
    return log_retornos.corr()


def correlacion_cola(
    log_retornos: pd.DataFrame,
    activo_referencia: str,
    percentil: float = 0.10,
) -> tuple[pd.DataFrame, int]:
    """Correlación usando solo los días del peor `percentil` del activo de referencia.

    Devuelve (matriz de correlación de cola, nº de observaciones usadas).
    """
    if activo_referencia not in log_retornos.columns:
        raise ErrorAnalisis(f"El activo de referencia '{activo_referencia}' no está en los datos.")
    if not 0 < percentil < 0.5:
        raise ErrorAnalisis("El percentil de cola debe estar en (0, 0.5).")

    referencia = log_retornos[activo_referencia]
    umbral = referencia.quantile(percentil)        # frontera del peor decil
    mascara = referencia <= umbral
    observaciones = int(mascara.sum())
    if observaciones < log_retornos.shape[1] + 1:
        raise ErrorAnalisis(
            f"Días de cola insuficientes ({observaciones}) para una correlación fiable."
        )
    return log_retornos.loc[mascara].corr(), observaciones


def diferencia_cola_menos_media(
    correlacion_cola_: pd.DataFrame,
    correlacion_media_: pd.DataFrame,
) -> pd.DataFrame:
    """Positivo = la pareja se correlaciona MÁS en las colas que de media (malo)."""
    return correlacion_cola_ - correlacion_media_

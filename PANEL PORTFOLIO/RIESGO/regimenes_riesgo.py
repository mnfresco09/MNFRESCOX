"""Comportamiento de cada método DENTRO de cada régimen y diversificación en crisis.

Es análisis descriptivo del PASADO en regímenes pasados: muestra qué método
amortiguó en los tramos bajistas/crisis y cuál creció en los alcistas. No predice
ni garantiza protección futura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ANALISIS.diversificacion import numero_efectivo_apuestas, ratio_diversificacion
from ANALISIS.momentos import covarianza_ledoit_wolf
from ANALISIS.regimenes import CRISIS
from CONTRATOS.modelos import Configuracion, DatosAlineados


def metricas_por_regimen(
    retornos: pd.DataFrame,
    regimenes: pd.Series,
    dias_anio: int,
) -> dict[str, pd.DataFrame]:
    """Para cada método, retorno/volatilidad/drawdown anualizados por régimen."""
    regimen_oos = regimenes.reindex(retornos.index).dropna()
    resultado: dict[str, pd.DataFrame] = {}
    etiquetas = sorted(regimen_oos.unique())
    for metodo in retornos.columns:
        filas = {}
        for etiqueta in etiquetas:
            dias = regimen_oos[regimen_oos == etiqueta].index
            serie = retornos.loc[dias, metodo]
            if serie.empty:
                continue
            equity = (1.0 + serie).cumprod()
            caida = equity / equity.cummax() - 1.0
            filas[etiqueta] = {
                "dias": int(len(serie)),
                "retorno_anual": float(serie.mean() * dias_anio),
                "volatilidad_anual": float(serie.std(ddof=1) * np.sqrt(dias_anio)) if len(serie) > 1 else 0.0,
                "max_drawdown": float(caida.min()),
            }
        resultado[metodo] = pd.DataFrame(filas).T
    return resultado


def diversificacion_en_crisis(
    datos: DatosAlineados,
    regimenes: pd.Series,
    pesos_por_metodo: dict[str, pd.Series],
    configuracion: Configuracion,
) -> pd.DataFrame:
    """Ratio de diversificación y número efectivo de apuestas en CRISIS vs. global.

    La covarianza de crisis se estima (Ledoit-Wolf) solo con los días de crisis;
    así se ve si la diversificación de cada cartera se mantiene o se evapora
    justo cuando los activos se sincronizan.
    """
    log_retornos = datos.log_retornos
    cov_global, _ = covarianza_ledoit_wolf(log_retornos, configuracion.dias_anio)

    dias_crisis = regimenes.reindex(log_retornos.index)
    crisis = log_retornos.loc[dias_crisis == CRISIS]
    hay_crisis = len(crisis) >= log_retornos.shape[1] + 1
    cov_crisis = (
        covarianza_ledoit_wolf(crisis, configuracion.dias_anio)[0] if hay_crisis else None
    )

    filas = {}
    for metodo, pesos in pesos_por_metodo.items():
        fila = {
            "ratio_global": ratio_diversificacion(pesos, cov_global),
            "enb_global": numero_efectivo_apuestas(pesos, cov_global),
        }
        if cov_crisis is not None:
            fila["ratio_crisis"] = ratio_diversificacion(pesos, cov_crisis)
            fila["enb_crisis"] = numero_efectivo_apuestas(pesos, cov_crisis)
        else:
            fila["ratio_crisis"] = float("nan")
            fila["enb_crisis"] = float("nan")
        filas[metodo] = fila
    return pd.DataFrame(filas).T

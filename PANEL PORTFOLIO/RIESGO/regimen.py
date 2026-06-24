"""Detección del régimen de mercado actual (baja / alta volatilidad / crisis).

Compara la volatilidad reciente del activo de referencia con su propia historia
(percentil) y mide el nivel medio de correlación reciente del universo. El
régimen contextualiza TODO el informe: el mismo VaR significa cosas distintas en
calma que en estrés.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.modelos import Configuracion, MomentsResult, PortfolioInput, RegimenMercado


def detectar_regimen(
    entrada: PortfolioInput,
    momentos: MomentsResult,
    cfg: Configuracion,
) -> RegimenMercado:
    par = cfg.parametros_regimen
    ref = entrada.log_retornos[cfg.activo_referencia]
    ventana = par.ventana_volatilidad

    vol_movil = ref.rolling(ventana).std(ddof=0) * np.sqrt(cfg.dias_anio)
    vol_movil = vol_movil.dropna()
    vol_actual = float(vol_movil.iloc[-1]) if not vol_movil.empty else float(ref.std() * np.sqrt(cfg.dias_anio))
    percentil = float((vol_movil <= vol_actual).mean()) if not vol_movil.empty else 0.5

    # Correlación media reciente (fuera de diagonal) sobre la última ventana larga.
    reciente = entrada.log_retornos.iloc[-min(len(entrada.log_retornos), par.ventana_media_larga):]
    corr = reciente.corr().to_numpy()
    fuera = corr[~np.eye(corr.shape[0], dtype=bool)]
    corr_media = float(np.mean(fuera)) if fuera.size else 0.0

    # Drawdown reciente del activo de referencia para distinguir crisis.
    equity = (1.0 + np.expm1(ref)).cumprod()
    dd = float((equity / equity.cummax() - 1.0).iloc[-1])

    if dd <= par.drawdown_crisis or percentil >= par.percentil_volatilidad_crisis:
        etiqueta = "crisis"
        desc = ("Régimen de CRISIS: volatilidad en el extremo alto de su historia y/o "
                "drawdown severo. Las correlaciones tienden a 1 y la diversificación se "
                "evapora; trate los VaR como suelos optimistas.")
    elif percentil >= 0.66:
        etiqueta = "alta_volatilidad"
        desc = ("Régimen de ALTA VOLATILIDAD: el riesgo táctico supera a su media histórica. "
                "Conviene inclinarse hacia perfiles Bajo/Medio.")
    else:
        etiqueta = "baja_volatilidad"
        desc = ("Régimen de BAJA/NORMAL VOLATILIDAD: el riesgo táctico está contenido respecto "
                "a su historia. Hay margen para perfiles de mayor riesgo si el mandato lo permite.")

    return RegimenMercado(
        etiqueta=etiqueta,
        volatilidad_actual=vol_actual,
        percentil_volatilidad=percentil,
        correlacion_media_actual=corr_media,
        descripcion=desc,
    )

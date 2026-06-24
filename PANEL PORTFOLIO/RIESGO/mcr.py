"""Descomposición del riesgo: Contribución Marginal al Riesgo (MCR).

Pregunta 4 (de dónde viene el riesgo): para una cartera w y covarianza Σ,

    σ_p = sqrt(wᵀΣw)
    MCR_i = (Σw)_i / σ_p            (derivada de σ_p respecto a w_i)
    CTR_i = w_i · MCR_i             (contribución total; Σ_i CTR_i = σ_p)

Permite ver qué activo aporta el riesgo aunque su peso sea pequeño. Se usa la
covarianza TÁCTICA (EWMA) para describir el riesgo de mañana; cae a la
estructural si la táctica no está disponible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.modelos import DescomposicionRiesgo


def descomponer_riesgo(pesos: pd.Series, covarianza: pd.DataFrame) -> DescomposicionRiesgo:
    activos = list(covarianza.index)
    w = pesos.reindex(activos).to_numpy(dtype=float)
    sigma = covarianza.to_numpy(dtype=float)
    var = float(w @ sigma @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    if vol <= 0:
        ceros = pd.Series(0.0, index=activos)
        return DescomposicionRiesgo(ceros, ceros, ceros, float((w ** 2).sum()))
    marginal = (sigma @ w) / vol
    contribucion = w * marginal
    contribucion_pct = contribucion / vol
    hhi = float((w ** 2).sum())
    return DescomposicionRiesgo(
        mcr=pd.Series(marginal, index=activos),
        contribucion=pd.Series(contribucion, index=activos),
        contribucion_pct=pd.Series(contribucion_pct, index=activos),
        concentracion_hhi=hhi,
    )

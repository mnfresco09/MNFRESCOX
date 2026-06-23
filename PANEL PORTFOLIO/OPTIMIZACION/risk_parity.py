"""Risk parity: igualar la CONTRIBUCIÓN al riesgo de cada activo.

No es el ingenuo 1/volatilidad (que ignora correlaciones). Se resuelve por
optimización: se minimiza la dispersión entre las contribuciones marginales al
riesgo, de modo que cada activo aporte la misma cantidad de riesgo a la cartera.
Definido para carteras solo-largas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import Restricciones

from .comun import limites


def risk_parity(covarianza: pd.DataFrame, restricciones: Restricciones) -> pd.Series:
    if not restricciones.solo_largos:
        raise ErrorOptimizacion("OPTIMIZACION", "Risk parity requiere cartera solo-larga.")
    cov = covarianza.to_numpy(dtype=float)
    n = cov.shape[0]

    def dispersion_contribuciones(w):
        varianza = w @ cov @ w
        if varianza <= 0:
            return 1e6
        contribuciones = w * (cov @ w) / np.sqrt(varianza)   # contribución de cada activo
        objetivo = contribuciones.mean()
        return float(((contribuciones - objetivo) ** 2).sum())

    resultado = minimize(
        dispersion_contribuciones,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=limites(restricciones, n),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    if not resultado.success:
        raise ErrorOptimizacion("OPTIMIZACION", f"Risk parity no convergió: {resultado.message}")
    return pd.Series(resultado.x, index=covarianza.index)

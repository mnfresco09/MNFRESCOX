"""Cartera de Máxima Diversificación (Choueifaty & Coignard, 2008).

Maximiza el ratio de diversificación DR(w) = (wᵀσ) / √(wᵀΣw), donde σ es el
vector de volatilidades individuales. Intuición: el numerador es el riesgo si los
activos no se ayudaran entre sí; el denominador es el riesgo real de la cartera.
El cociente es grande cuando los activos se mueven en direcciones distintas
(idealmente opuestas), de modo que las caídas de unos se compensan con las
subidas de otros. Es la forma robusta y reconocida de buscar pesos que exploten
la anticorrelación: si algo baja y algo sube, la cartera tiende a quedarse igual.

No promete convexidad pura (quedarse plano al caer y volar al subir): eso exige
opciones o cobertura dinámica. Con pesos fijos, esto aproxima la parte de
"compensación cuando los activos se mueven en sentidos opuestos".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import Restricciones

from .comun import limites


def max_diversificacion(covarianza: pd.DataFrame, restricciones: Restricciones) -> pd.Series:
    cov = covarianza.to_numpy(dtype=float)
    n = cov.shape[0]
    sigma = np.sqrt(np.clip(np.diag(cov), 1e-18, None))

    def negativo_dr(w: np.ndarray) -> float:
        vol = np.sqrt(max(w @ cov @ w, 1e-18))
        return -float((w @ sigma) / vol)

    cotas = limites(restricciones, n)
    restr = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    resultado = minimize(
        negativo_dr,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=cotas,
        constraints=restr,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not resultado.success:
        raise ErrorOptimizacion(
            "OPTIMIZACION", f"Máxima diversificación no convergió: {resultado.message}"
        )
    return pd.Series(resultado.x, index=covarianza.index)

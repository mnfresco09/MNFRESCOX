"""Min-CVaR (Rockafellar-Uryasev) como PROGRAMA LINEAL sobre escenarios reales.

No asume normalidad: usa directamente los retornos diarios observados como
escenarios. Minimiza el CVaR (Expected Shortfall) de la pérdida de la cartera al
nivel de confianza dado, mediante la formulación lineal de Rockafellar-Uryasev:

    min_{w,t,z}  t + 1/((1-α)·T) · Σ_s z_s
    s.a.         z_s ≥ -(r_s · w) - t,   z_s ≥ 0,   Σ w = 1,   cotas por activo

donde t aproxima el VaR y z_s las pérdidas más allá de él. Se resuelve con cvxpy.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import Restricciones


def min_cvar(
    escenarios: pd.DataFrame,
    restricciones: Restricciones,
    nivel_confianza: float,
) -> pd.Series:
    matriz = escenarios.to_numpy(dtype=float)
    n_escenarios, n_activos = matriz.shape
    if n_escenarios < 10:
        raise ErrorOptimizacion("OPTIMIZACION", "Min-CVaR necesita más escenarios de retorno.")

    w = cp.Variable(n_activos)
    t = cp.Variable()
    z = cp.Variable(n_escenarios, nonneg=True)
    perdidas = -(matriz @ w)            # pérdida de cartera en cada escenario

    restr = [cp.sum(w) == 1, z >= perdidas - t]
    restr.append(w >= (0.0 if restricciones.solo_largos else -1.0))
    if restricciones.peso_maximo is not None:
        restr.append(w <= restricciones.peso_maximo)

    objetivo = t + (1.0 / ((1.0 - nivel_confianza) * n_escenarios)) * cp.sum(z)
    problema = cp.Problem(cp.Minimize(objetivo), restr)
    problema.solve(solver=cp.CLARABEL)

    if problema.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise ErrorOptimizacion("OPTIMIZACION", f"Min-CVaR no resolvió (estado={problema.status}).")
    pesos = np.clip(np.asarray(w.value, dtype=float), 0.0 if restricciones.solo_largos else None, None)
    pesos = pesos / pesos.sum()
    return pd.Series(pesos, index=escenarios.columns)

"""Estrategia Challenger: optimización directa de CVaR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import (
    Configuracion,
    MomentsResult,
    PortfolioCandidate,
    PortfolioInput,
    ResultadoFrontera,
)

from .base import OptimizadorBase, construir_candidate_comun


class OptimizadorCVaR(OptimizadorBase):
    """Minimiza Expected Shortfall histórico por programación lineal."""

    nombre = "CVAR"

    def optimizar(
        self,
        entrada: PortfolioInput,
        momentos: MomentsResult,
        cfg: Configuracion,
        frontera: ResultadoFrontera | None = None,
    ) -> tuple[PortfolioCandidate, ...]:
        pesos = self._resolver_min_cvar(entrada, cfg)
        return (
            construir_candidate_comun(
                nivel="cvar",
                motor=self.nombre,
                pesos=pesos,
                entrada=entrada,
                momentos=momentos,
                cfg=cfg,
            ),
        )

    def _resolver_min_cvar(self, entrada: PortfolioInput, cfg: Configuracion) -> pd.Series:
        activos = list(entrada.log_retornos.columns)
        retornos = np.expm1(entrada.log_retornos.to_numpy(dtype=float))
        n_obs, n_activos = retornos.shape
        alpha = 1.0 - cfg.nivel_confianza_95

        # Variables: [w_1..w_n, eta, u_1..u_T].
        n_vars = n_activos + 1 + n_obs
        objetivo = np.zeros(n_vars)
        objetivo[n_activos] = 1.0
        objetivo[n_activos + 1:] = 1.0 / (alpha * n_obs)

        # u_t >= loss_t - eta = -R_t·w - eta  ->  -R_t·w - eta - u_t <= 0
        a_ub = np.zeros((n_obs, n_vars))
        a_ub[:, :n_activos] = -retornos
        a_ub[:, n_activos] = -1.0
        a_ub[:, n_activos + 1:] = -np.eye(n_obs)
        b_ub = np.zeros(n_obs)

        a_eq = np.zeros((1, n_vars))
        a_eq[0, :n_activos] = 1.0
        b_eq = np.array([1.0])

        inf = cfg.restricciones.peso_minimo if cfg.restricciones.solo_largos else -abs(cfg.restricciones.peso_maximo or 1.0)
        sup = cfg.restricciones.peso_maximo if cfg.restricciones.peso_maximo is not None else 1.0
        bounds = [(float(inf), float(sup))] * n_activos
        bounds.append((None, None))              # eta
        bounds.extend([(0.0, None)] * n_obs)     # u_t

        res = linprog(
            c=objetivo,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise ErrorOptimizacion("OPTIMIZACION", f"Min-CVaR no convergió: {res.message}")
        return pd.Series(res.x[:n_activos], index=activos)

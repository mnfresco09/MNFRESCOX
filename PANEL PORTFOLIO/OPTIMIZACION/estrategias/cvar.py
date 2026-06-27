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
from OPTIMIZACION import optimizador as opt

from .base import (
    OptimizadorBase,
    construir_candidate_comun,
    diagnosticos_curva,
    retornos_cartera_simples,
)

PERCENTILES = (("bajo", 0.20), ("medio", 0.50), ("alto", 0.80))


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
        curva = self._construir_frontera_cvar(entrada, momentos, cfg)
        curva_riesgo = curva.sort_values("cvar_abs").reset_index(drop=True)
        candidatos: list[PortfolioCandidate] = []
        for nivel, percentil in PERCENTILES:
            fila = _fila_en_percentil_riesgo(curva_riesgo, percentil)
            candidatos.append(self._candidate_desde_fila(nivel, fila, entrada, momentos, cfg))
        fila_max_k = curva.iloc[int(curva["k_ratio"].to_numpy(dtype=float).argmax())]
        candidatos.append(self._candidate_desde_fila("max_k_ratio", fila_max_k, entrada, momentos, cfg))
        return tuple(candidatos)

    def _construir_frontera_cvar(
        self,
        entrada: PortfolioInput,
        momentos: MomentsResult,
        cfg: Configuracion,
    ) -> pd.DataFrame:
        activos = list(momentos.cov_estructural.index)
        mu = momentos.retornos_ajustados.reindex(activos).astype(float)
        retornos = np.expm1(entrada.log_retornos.reindex(columns=activos).to_numpy(dtype=float))

        w_min_vol = opt.minima_varianza(momentos.cov_estructural.reindex(index=activos, columns=activos), cfg.restricciones)
        retorno_min_vol = float(w_min_vol.reindex(activos).to_numpy(dtype=float) @ mu.to_numpy(dtype=float))
        _ret_min, retorno_max = opt.rango_retorno_factible(mu, cfg.restricciones, len(activos))
        inicio, fin = sorted((retorno_min_vol, retorno_max))
        # Optimizador CVaR es computacionalmente pesado (linprog). 
        # Reducimos los puntos a un máximo de 20 para acelerarlo x6 sin perder resolución en los perfiles.
        n_puntos = min(int(cfg.n_puntos_frontera), 20)
        objetivos = np.linspace(inicio, fin, max(n_puntos, 4))

        filas = []
        for objetivo in objetivos:
            pesos = resolver_min_cvar(
                objetivo_retorno=float(objetivo),
                entrada=entrada,
                cfg=cfg,
                mu_anual=mu.to_numpy(dtype=float),
            )
            w = pesos.reindex(activos).to_numpy(dtype=float)
            retorno = float(w @ mu.to_numpy(dtype=float))
            cvar_abs = _cvar_abs(retornos @ w, cfg.nivel_confianza_99)
            _geom, _r2, k_ratio = diagnosticos_curva(retornos_cartera_simples(pesos, entrada.log_retornos), cfg.dias_anio)
            fila = {
                "retorno": retorno,
                "cvar_abs": cvar_abs,
                "k_ratio": float(k_ratio or 0.0),
            }
            fila.update({f"peso·{a}": float(p) for a, p in zip(activos, w)})
            filas.append(fila)

        if not filas:
            raise ErrorOptimizacion("OPTIMIZACION", "No se pudo construir la frontera CVaR.")
        return pd.DataFrame(filas)

    def _candidate_desde_fila(
        self,
        nivel: str,
        fila: pd.Series,
        entrada: PortfolioInput,
        momentos: MomentsResult,
        cfg: Configuracion,
    ) -> PortfolioCandidate:
        activos = list(momentos.cov_estructural.index)
        pesos = pd.Series([float(fila[f"peso·{a}"]) for a in activos], index=activos)
        return construir_candidate_comun(
            nivel=nivel,
            motor=self.nombre,
            pesos=pesos,
            entrada=entrada,
            momentos=momentos,
            cfg=cfg,
        )

def resolver_min_cvar(
    objetivo_retorno: float,
    entrada: PortfolioInput,
    cfg: Configuracion,
    mu_anual: np.ndarray,
) -> pd.Series:
    activos = list(entrada.log_retornos.columns)
    retornos = np.expm1(entrada.log_retornos.to_numpy(dtype=float))
    n_obs, n_activos = retornos.shape
    alpha = 1.0 - cfg.nivel_confianza_99

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

    fila_retorno = np.zeros((1, n_vars))
    fila_retorno[0, :n_activos] = -mu_anual
    a_ub = np.vstack([a_ub, fila_retorno])
    b_ub = np.concatenate([b_ub, np.array([-float(objetivo_retorno)])])

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



def _cvar_abs(retornos: np.ndarray, nivel: float) -> float:
    alpha = 1.0 - nivel
    var = float(np.quantile(retornos, alpha))
    cola = retornos[retornos <= var]
    cvar = float(cola.mean()) if cola.size else var
    return abs(cvar)


def _fila_en_percentil_riesgo(curva: pd.DataFrame, percentil: float) -> pd.Series:
    objetivo = float(np.quantile(curva["cvar_abs"].to_numpy(dtype=float), percentil))
    idx = int((curva["cvar_abs"] - objetivo).abs().argmin())
    return curva.iloc[idx]

"""Contrato Strategy común para motores de optimización de cartera."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import (
    Configuracion,
    MomentsResult,
    PortfolioCandidate,
    PortfolioInput,
    ResultadoFrontera,
)
from RIESGO.mcr import descomponer_riesgo


class OptimizadorBase(ABC):
    """Interfaz común para optimizadores intercambiables."""

    nombre: str

    @abstractmethod
    def optimizar(
        self,
        entrada: PortfolioInput,
        momentos: MomentsResult,
        cfg: Configuracion,
        frontera: ResultadoFrontera | None = None,
    ) -> tuple[PortfolioCandidate, ...]:
        """Devuelve una o más carteras con el contrato público común."""


def retornos_cartera_simples(pesos: pd.Series, log_retornos: pd.DataFrame) -> np.ndarray:
    """Retornos simples diarios de una cartera con pesos fijos."""
    activos = list(pesos.index)
    simples = np.expm1(log_retornos.reindex(columns=activos))
    return simples.to_numpy(dtype=float) @ pesos.to_numpy(dtype=float)


def diagnosticos_curva(retornos: np.ndarray, dias_anio: int) -> tuple[float, float | None, float | None]:
    """CAGR, R² y K-ratio aproximado de la curva de capital con pesos fijos."""
    ret = np.asarray(retornos, dtype=float)
    ret = np.clip(ret, -0.999999, None)
    geom = float(np.expm1(np.log1p(ret).mean() * float(dias_anio)))

    capital = np.cumprod(1.0 + ret)
    y = np.log(np.maximum(capital, 1e-18))
    x = np.arange(y.size, dtype=float)
    if y.size < 3 or float(np.var(y)) <= 1e-18:
        return geom, 0.0, 0.0

    pendiente, intercepto = np.polyfit(x, y, 1)
    estimada = pendiente * x + intercepto
    resid = y - estimada
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) @ (y - y.mean())))
    r2 = float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot))) if ss_tot > 1e-18 else 0.0

    denom = float(((x - x.mean()) @ (x - x.mean())))
    if denom <= 1e-18 or y.size <= 2:
        return geom, r2, 0.0
    var_res = ss_res / max(y.size - 2, 1)
    se_pendiente = float(np.sqrt(max(var_res, 0.0) / denom))
    k_ratio = float(pendiente / se_pendiente) if se_pendiente > 1e-18 else 0.0
    return geom, r2, k_ratio


def proyectar_a_restricciones(pesos: pd.Series, cfg: Configuracion) -> pd.Series:
    """Proyecta pesos preliminares a las restricciones duras operativas."""
    activos = list(pesos.index)
    objetivo = pesos.to_numpy(dtype=float)
    objetivo = np.clip(objetivo, 0.0, None) if cfg.restricciones.solo_largos else objetivo
    if abs(float(objetivo.sum())) <= 1e-15:
        objetivo = np.full(len(activos), 1.0 / len(activos))
    objetivo = objetivo / objetivo.sum()

    inf = cfg.restricciones.peso_minimo if cfg.restricciones.solo_largos else -abs(cfg.restricciones.peso_maximo or 1.0)
    sup = cfg.restricciones.peso_maximo if cfg.restricciones.peso_maximo is not None else 1.0
    bnds = [(float(inf), float(sup))] * len(activos)
    if all(lo - 1e-10 <= p <= hi + 1e-10 for p, (lo, hi) in zip(objetivo, bnds)):
        return pd.Series(objetivo, index=activos)

    res = minimize(
        lambda w: float(((w - objetivo) @ (w - objetivo))),
        objetivo,
        method="SLSQP",
        bounds=bnds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success:
        raise ErrorOptimizacion("OPTIMIZACION", f"No se pudieron proyectar pesos: {res.message}")
    return pd.Series(res.x, index=activos)


def construir_candidate_comun(
    *,
    nivel: str,
    motor: str,
    pesos: pd.Series,
    entrada: PortfolioInput,
    momentos: MomentsResult,
    cfg: Configuracion,
) -> PortfolioCandidate:
    """Construye la salida común de cualquier estrategia."""
    activos = list(momentos.cov_estructural.index)
    w = pesos.reindex(activos).astype(float)
    ret_diarios = retornos_cartera_simples(w, entrada.log_retornos)
    ret_geom, r2, k_ratio = diagnosticos_curva(ret_diarios, cfg.dias_anio)

    cov_e = momentos.cov_estructural.to_numpy(dtype=float)
    cov_t = momentos.cov_tactica.to_numpy(dtype=float)
    wv = w.to_numpy(dtype=float)
    vol_e = float(np.sqrt(max(wv @ cov_e @ wv, 0.0)))
    vol_t = float(np.sqrt(max(wv @ cov_t @ wv, 0.0)))
    sharpe = (ret_geom - cfg.tasa_libre_riesgo_anual) / vol_e if vol_e > 0 else 0.0

    return PortfolioCandidate(
        nivel=nivel,
        motor_optimizacion=motor,
        pesos=w,
        retorno_esperado=ret_geom,
        retorno_geometrico=ret_geom,
        volatilidad_estructural=vol_e,
        volatilidad_tactica=vol_t,
        sharpe=float(sharpe),
        descomposicion=descomponer_riesgo(w, momentos.cov_tactica),
        r2_curva_capital=r2,
        k_ratio=k_ratio,
    )

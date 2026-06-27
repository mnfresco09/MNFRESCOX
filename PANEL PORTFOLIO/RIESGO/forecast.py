"""Riesgo prospectivo de una cartera (pregunta 4).

Para unos pesos fijos calcula:
  • VaR/CVaR HISTÓRICO   — distribución empírica realizada (sin supuestos).
  • VaR/CVaR PARAMÉTRICO  — vol táctica EWMA T+1 × cuantil normal (forecast).
  • VaR/CVaR FHS          — Filtered Historical Simulation (motor Rust/fallback):
                            estandariza por la vol EWMA, reescala a la vol T+1.
  • SimulationSummary     — Monte Carlo por bootstrapping a horizonte (fan chart,
                            prob. de pérdida, CDaR).

Convención: VaR/CVaR son retornos NEGATIVOS (pérdida en la cola). NUNCA se
llaman "pérdida máxima": son estimaciones bajo los supuestos del modelo.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from CONTRATOS.modelos import (
    Configuracion,
    MomentsResult,
    PortfolioCandidate,
    RiskForecast,
    SimulationSummary,
)

from . import motor_bindings


def _retornos_cartera(pesos: pd.Series, log_retornos: pd.DataFrame) -> np.ndarray:
    """Serie de retornos SIMPLES diarios de la cartera con pesos fijos."""
    simples = np.expm1(log_retornos)
    w = pesos.reindex(simples.columns).to_numpy(dtype=float)
    return (simples.to_numpy() @ w)


def _ewma_vol_path(retornos: np.ndarray, lam: float) -> np.ndarray:
    """Senda de volatilidad diaria EWMA (RiskMetrics)."""
    var = np.empty_like(retornos)
    var[0] = retornos[0] ** 2
    for t in range(1, retornos.shape[0]):
        var[t] = lam * var[t - 1] + (1.0 - lam) * retornos[t - 1] ** 2
    return np.sqrt(np.maximum(var, 1e-18))


def _var_cvar_hist(retornos: np.ndarray, nivel: float) -> tuple[float, float]:
    alpha = 1.0 - nivel
    var = float(np.quantile(retornos, alpha))
    cola = retornos[retornos <= var]
    cvar = float(cola.mean()) if cola.size else var
    return var, cvar


def _var_cvar_param(sigma: float, mu: float, nivel: float) -> tuple[float, float]:
    """VaR/CVaR paramétrico normal a 1 día con vol táctica `sigma`."""
    alpha = 1.0 - nivel
    z = norm.ppf(alpha)
    var = mu + sigma * z
    cvar = mu - sigma * norm.pdf(z) / alpha
    return float(var), float(cvar)


def calcular_forecast(
    pesos: pd.Series,
    log_retornos: pd.DataFrame,
    momentos: MomentsResult,
    cfg: Configuracion,
) -> RiskForecast:
    ret = _retornos_cartera(pesos, log_retornos)
    n95, n99 = cfg.nivel_confianza_95, cfg.nivel_confianza_99

    # Vol táctica diaria T+1 de la cartera (de la covarianza EWMA anual).
    w = pesos.reindex(momentos.cov_tactica.index).to_numpy(dtype=float)
    var_anual_t1 = float(w @ momentos.cov_tactica.to_numpy() @ w)
    sigma_t1 = float(np.sqrt(max(var_anual_t1, 0.0)) / np.sqrt(cfg.dias_anio))
    mu_diario = float(np.mean(ret))

    vh95, ch95 = _var_cvar_hist(ret, n95)
    vh99, ch99 = _var_cvar_hist(ret, n99)
    vp95, cp95 = _var_cvar_param(sigma_t1, mu_diario, n95)
    vp99, cp99 = _var_cvar_param(sigma_t1, mu_diario, n99)

    # FHS: residuos estandarizados por la senda EWMA, reescalados a sigma_t1.
    vol_path = _ewma_vol_path(ret, cfg.lambda_ewma)
    residuos = (ret - mu_diario) / vol_path
    fhs_res, fuente_fhs = motor_bindings.fhs(residuos, sigma_t1, niveles=(n95, n99))
    vf95, cf95 = fhs_res.get(round(n95, 2), fhs_res.get(n95, (vp95, cp95)))
    vf99, cf99 = fhs_res.get(round(n99, 2), fhs_res.get(n99, (vp99, cp99)))
    # FHS centra en 0; añade la deriva diaria para coherencia con los demás.
    vf95, cf95 = vf95 + mu_diario, cf95 + mu_diario
    vf99, cf99 = vf99 + mu_diario, cf99 + mu_diario

    return RiskForecast(
        horizonte_dias=cfg.horizonte_dias,
        volatilidad_tactica_diaria=sigma_t1,
        var_hist_95=vh95, var_hist_99=vh99, cvar_hist_95=ch95, cvar_hist_99=ch99,
        var_param_95=vp95, var_param_99=vp99, cvar_param_95=cp95, cvar_param_99=cp99,
        var_fhs_95=vf95, var_fhs_99=vf99, cvar_fhs_95=cf95, cvar_fhs_99=cf99,
        fuente_fhs=fuente_fhs,
    )


def calcular_simulacion(
    pesos: pd.Series,
    log_retornos: pd.DataFrame,
    cfg: Configuracion,
) -> SimulationSummary:
    ret = _retornos_cartera(pesos, log_retornos)
    resumen, fuente = motor_bindings.montecarlo(
        ret, cfg.horizonte_dias, cfg.n_trayectorias_mc,
        percentiles=cfg.percentiles_fan, seed=cfg.semilla,
    )
    sendas = np.asarray(resumen["sendas"], dtype=float)  # (n_perc, horizonte)
    dias = pd.RangeIndex(1, sendas.shape[1] + 1, name="dia")
    sendas_df = pd.DataFrame(sendas.T, index=dias, columns=[f"p{p}" for p in cfg.percentiles_fan])
    return SimulationSummary(
        horizonte_dias=cfg.horizonte_dias,
        percentiles=tuple(cfg.percentiles_fan),
        sendas_percentil=sendas_df,
        prob_perdida=float(resumen["prob_perdida"]),
        cdar_30d=float(resumen["cdar"]),
        retorno_mediano=float(resumen["retorno_mediano"]),
        perdida_p5=float(resumen["perdida_p5"]),
        fuente=fuente,
    )


def enriquecer_candidatos_riesgo(
    candidatos: tuple[PortfolioCandidate, ...],
    log_retornos: pd.DataFrame,
    momentos: MomentsResult,
    cfg: Configuracion,
    max_workers: int | None = None,
) -> tuple[PortfolioCandidate, ...]:
    """Añade forecast y simulación a candidatos preservando orden.

    No cambia ninguna métrica ni reduce calidad: solo ejecuta en paralelo el
    cálculo independiente de cada cartera. `executor.map` conserva el orden de
    `candidatos`, algo importante para el reporte y los tests.
    """
    if not candidatos:
        return ()

    def enriquecer(candidato: PortfolioCandidate) -> PortfolioCandidate:
        fc = calcular_forecast(candidato.pesos, log_retornos, momentos, cfg)
        sim = calcular_simulacion(candidato.pesos, log_retornos, cfg)
        return replace(candidato, forecast=fc, simulacion=sim)

    n_workers = max_workers
    if n_workers is None:
        n_workers = min(len(candidatos), max((os.cpu_count() or 1) - 1, 1), 8)
    if n_workers <= 1 or len(candidatos) == 1:
        return tuple(enriquecer(c) for c in candidatos)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return tuple(pool.map(enriquecer, candidatos))

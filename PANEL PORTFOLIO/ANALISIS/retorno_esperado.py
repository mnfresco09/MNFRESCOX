"""Estimador de retorno esperado μ con política de fallback institucional.

Pregunta 3 (insumo): ¿qué retorno esperamos? El μ histórico crudo es el insumo
MÁS ruidoso de toda la optimización, así que NUNCA se usa tal cual.

Jerarquía (con fallback explícito):
  1. Black-Litterman  → SOLO si hay views defendibles en config.
  2. Shrinkage conservador (James-Stein hacia la gran media) → fallback por
     defecto, y también si Black-Litterman falla numéricamente.

Devuelve siempre un μ ANUAL por activo (no pesos): la frontera y los perfiles se
encargan de traducir μ + Σ en pesos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.modelos import Configuracion

from .momentos import retornos_shrinkage

TAU = 0.05


def _equilibrio_implicito(cov: pd.DataFrame, mu_hist: pd.Series, rf: float) -> np.ndarray:
    """Π = δ·Σ·w_mkt con w_mkt equiponderada (no hay capitalizaciones)."""
    n = cov.shape[0]
    w_mkt = np.full(n, 1.0 / n)
    sigma = cov.to_numpy(dtype=float)
    mu = mu_hist.reindex(cov.index).to_numpy(dtype=float)
    var_mercado = float(w_mkt @ sigma @ w_mkt)
    delta = (float(w_mkt @ mu) - rf) / var_mercado if var_mercado > 0 else 1.0
    return delta * (sigma @ w_mkt)


def _black_litterman_mu(
    mu_hist: pd.Series,
    cov: pd.DataFrame,
    cfg: Configuracion,
) -> pd.Series:
    """Posterior de Black-Litterman como μ anual total (exceso + rf)."""
    rf = cfg.tasa_libre_riesgo_anual
    activos = list(cov.index)
    sigma = cov.to_numpy(dtype=float)
    pi = _equilibrio_implicito(cov, mu_hist, rf)

    k = len(cfg.views_black_litterman)
    p = np.zeros((k, len(activos)))
    q = np.zeros(k)
    indice = {a: i for i, a in enumerate(activos)}
    for j, view in enumerate(cfg.views_black_litterman):
        suma = 0.0
        for activo, coef in view.activos:
            p[j, indice[activo]] = coef
            suma += coef
        q[j] = view.retorno_anual - rf * suma
    tau_sigma = TAU * sigma
    omega = np.diag([
        max(1.0 / v.confianza - 1.0, 1e-8) * float(p[j] @ tau_sigma @ p[j])
        for j, v in enumerate(cfg.views_black_litterman)
    ])
    a = p @ tau_sigma @ p.T + omega
    ajuste = tau_sigma @ p.T @ np.linalg.pinv(a) @ (q - p @ pi)
    return pd.Series(pi + ajuste + rf, index=activos)


def estimar_retorno_esperado(
    log_retornos: pd.DataFrame,
    cov_estructural: pd.DataFrame,
    cfg: Configuracion,
) -> tuple[pd.Series, str]:
    """Devuelve (μ_ajustado anual, fuente). Aplica la política de fallback."""
    mu_hist = log_retornos.mean() * float(cfg.dias_anio)
    if cfg.views_black_litterman:
        try:
            mu = _black_litterman_mu(mu_hist, cov_estructural, cfg)
            if np.isfinite(mu.to_numpy()).all():
                return mu.reindex(cov_estructural.index), "black_litterman"
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            pass  # fallback a shrinkage
    mu = retornos_shrinkage(log_retornos, cfg.dias_anio, cfg.shrinkage_retorno)
    return mu.reindex(cov_estructural.index), "shrinkage"

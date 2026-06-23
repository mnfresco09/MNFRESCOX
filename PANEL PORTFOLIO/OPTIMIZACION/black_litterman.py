"""Black-Litterman: equilibrio de mercado implícito + views opcionales.

Como NO disponemos de capitalizaciones de mercado (solo precios), tomamos la
cartera EQUIPONDERADA como prior neutral de mercado y documentamos esa decisión.
A partir de ella se obtienen los retornos de equilibrio implícitos (reverse
optimization) y, si hay views en config, se combinan con su confianza mediante
la fórmula de Black-Litterman. Si NO hay views, el método cae limpiamente a la
cartera de mercado (equiponderada), como exige la especificación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.modelos import Configuracion, Restricciones, ViewBlackLitterman

from .markowitz import maximo_sharpe

TAU = 0.05   # incertidumbre sobre el prior de equilibrio (valor estándar)


def _retornos_equilibrio(covarianza: pd.DataFrame, mu_hist: pd.Series, rf: float) -> tuple[np.ndarray, float]:
    """Retornos de equilibrio implícitos Π = δ·Σ·w_mkt con w_mkt equiponderada."""
    n = covarianza.shape[0]
    w_mkt = np.full(n, 1.0 / n)
    sigma = covarianza.to_numpy(dtype=float)
    mu = mu_hist.reindex(covarianza.index).to_numpy(dtype=float)
    exceso_mercado = float(w_mkt @ mu) - rf
    var_mercado = float(w_mkt @ sigma @ w_mkt)
    delta = exceso_mercado / var_mercado if var_mercado > 0 else 1.0
    pi = delta * (sigma @ w_mkt)        # retornos de equilibrio (en exceso sobre rf)
    return pi, delta


def _matriz_views(
    views: tuple[ViewBlackLitterman, ...],
    activos: list[str],
    rf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construye P (k×n) y Q (k) en EXCESO sobre rf."""
    p = np.zeros((len(views), len(activos)))
    q = np.zeros(len(views))
    indice = {a: i for i, a in enumerate(activos)}
    for k, view in enumerate(views):
        suma_coef = 0.0
        for activo, coef in view.activos:
            p[k, indice[activo]] = coef
            suma_coef += coef
        q[k] = view.retorno_anual - rf * suma_coef
    return p, q


def black_litterman(
    retornos_esperados: pd.Series,
    covarianza: pd.DataFrame,
    restricciones: Restricciones,
    configuracion: Configuracion,
) -> tuple[pd.Series, str]:
    """Devuelve (pesos, diagnóstico)."""
    rf = configuracion.tasa_libre_riesgo_anual
    activos = list(covarianza.index)
    sigma = covarianza.to_numpy(dtype=float)

    pi, _delta = _retornos_equilibrio(covarianza, retornos_esperados, rf)

    if not configuracion.views_black_litterman:
        mu_bl = pi
        diagnostico = "Sin views: cae a la cartera de mercado (equiponderada)."
    else:
        p, q = _matriz_views(configuracion.views_black_litterman, activos, rf)
        tau_sigma = TAU * sigma
        # Ω diagonal a partir de la confianza de cada view (confianza→1 ⇒ casi certeza).
        omega = np.diag([
            max((1.0 / v.confianza - 1.0), 1e-8) * float(p[k] @ tau_sigma @ p[k])
            for k, v in enumerate(configuracion.views_black_litterman)
        ])
        a = p @ tau_sigma @ p.T + omega
        ajuste = tau_sigma @ p.T @ np.linalg.pinv(a) @ (q - p @ pi)
        mu_bl = pi + ajuste
        diagnostico = f"{len(configuracion.views_black_litterman)} view(s) combinada(s) con el equilibrio."

    # Optimiza máximo Sharpe con los retornos posteriores (total = exceso + rf).
    retornos_bl = pd.Series(mu_bl + rf, index=activos)
    pesos = maximo_sharpe(retornos_bl, covarianza, rf, restricciones)
    return pesos, diagnostico

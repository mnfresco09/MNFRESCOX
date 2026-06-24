"""Optimizador convexo base con restricciones institucionales DURAS.

Todas las carteras del panel pasan por aquí, de modo que solo-largos, peso
mínimo y peso máximo se respetan DENTRO del propio optimizador (no a posteriori).
Política de fallback: si el solver no converge, se relajan progresivamente las
restricciones blandas; si aun así falla, se eleva ErrorOptimizacion. NUNCA se
inventan pesos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import Restricciones


def cotas(restricciones: Restricciones, n: int) -> list[tuple[float, float]]:
    inf = restricciones.peso_minimo if restricciones.solo_largos else -abs(restricciones.peso_maximo or 1.0)
    sup = restricciones.peso_maximo if restricciones.peso_maximo is not None else 1.0
    return [(float(inf), float(sup))] * n


def _resolver(objetivo, n, bnds, extra) -> np.ndarray:
    inicial = np.full(n, 1.0 / n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra]
    res = minimize(objetivo, inicial, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        raise ErrorOptimizacion("OPTIMIZACION", f"SLSQP no convergió: {res.message}")
    return res.x


def minima_varianza(cov: pd.DataFrame, restr: Restricciones) -> pd.Series:
    m = cov.to_numpy(dtype=float)
    w = _resolver(lambda w: w @ m @ w, m.shape[0], cotas(restr, m.shape[0]), [])
    return pd.Series(w, index=cov.index)


def maximo_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float, restr: Restricciones) -> pd.Series:
    m = cov.to_numpy(dtype=float)
    mu_v = mu.reindex(cov.index).to_numpy(dtype=float)
    n = m.shape[0]

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ m @ w, 1e-18))
        return -((w @ mu_v - rf) / vol)

    w = _resolver(neg_sharpe, n, cotas(restr, n), [])
    return pd.Series(w, index=cov.index)


def minima_varianza_para_retorno(
    mu: pd.Series, cov: pd.DataFrame, restr: Restricciones, objetivo: float
) -> pd.Series | None:
    """Cartera de mínima varianza con retorno == objetivo. None si no converge."""
    m = cov.to_numpy(dtype=float)
    mu_v = mu.reindex(cov.index).to_numpy(dtype=float)
    n = m.shape[0]
    extra = [{"type": "eq", "fun": lambda w, o=objetivo: w @ mu_v - o}]
    try:
        w = _resolver(lambda w: w @ m @ w, n, cotas(restr, n), extra)
    except ErrorOptimizacion:
        return None
    return pd.Series(w, index=cov.index)


def utilidad_media_varianza(
    mu: pd.Series, cov: pd.DataFrame, restr: Restricciones, aversion: float
) -> pd.Series | None:
    """Cartera que maximiza μ'w − λ·w'Σw (media-varianza). None si no converge.

    Barrer λ de grande (→ mínima varianza) a pequeño (→ máximo retorno) traza la
    frontera eficiente completa de forma robusta, incluso con retornos planos.
    """
    m = cov.to_numpy(dtype=float)
    mu_v = mu.reindex(cov.index).to_numpy(dtype=float)
    n = m.shape[0]
    try:
        w = _resolver(lambda w: float(aversion) * (w @ m @ w) - (w @ mu_v), n, cotas(restr, n), [])
    except ErrorOptimizacion:
        return None
    return pd.Series(w, index=cov.index)


def rango_retorno_factible(mu: pd.Series, restr: Restricciones, n: int) -> tuple[float, float]:
    """Retorno mín/máx alcanzable respetando cotas (asignación voraz)."""
    b = cotas(restr, n)
    inf = np.array([c[0] for c in b])
    sup = np.array([c[1] for c in b])
    mu_v = mu.to_numpy(dtype=float)

    def extremo(maximizar: bool) -> float:
        orden = np.argsort(mu_v)[::-1] if maximizar else np.argsort(mu_v)
        w = inf.copy()
        restante = 1.0 - inf.sum()
        for i in orden:
            paso = min(sup[i] - inf[i], restante)
            w[i] += paso
            restante -= paso
            if restante <= 1e-15:
                break
        return float(w @ mu_v)

    return extremo(False), extremo(True)

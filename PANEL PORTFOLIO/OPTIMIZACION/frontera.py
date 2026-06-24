"""Frontera eficiente restringida — el 100% de puntos factibles.

Pregunta 3 (mapa): para una rejilla DENSA de retornos objetivo entre el mínimo y
el máximo factibles, se halla la mínima volatilidad alcanzable respetando las
restricciones duras. Además se genera una NUBE de carteras factibles (fondo de
densidad riesgo-retorno) para visualizar el universo completo de opciones.

Usa la covarianza ESTRUCTURAL (Ledoit-Wolf) y el μ ajustado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorOptimizacion
from CONTRATOS.modelos import Configuracion, ResultadoFrontera, Restricciones

from . import optimizador as opt


def _metricas(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def construir_frontera(
    mu: pd.Series,
    cov_estructural: pd.DataFrame,
    cfg: Configuracion,
) -> ResultadoFrontera:
    restr = cfg.restricciones
    rf = cfg.tasa_libre_riesgo_anual
    activos = list(cov_estructural.index)
    n = len(activos)
    mu_v = mu.reindex(activos).to_numpy(dtype=float)
    cov_v = cov_estructural.to_numpy(dtype=float)

    w_mv = opt.minima_varianza(cov_estructural, restr)
    w_ms = opt.maximo_sharpe(mu, cov_estructural, rf, restr)

    # Frontera por barrido de aversión al riesgo λ: de máximo retorno (λ→0) a
    # mínima varianza (λ→∞). Robusto aunque los retornos sean casi planos.
    aversiones = np.logspace(-2.0, 3.0, cfg.n_puntos_frontera)
    filas = []
    vistos: set[tuple[float, float]] = set()
    for lam in aversiones:
        w = opt.utilidad_media_varianza(mu, cov_estructural, restr, float(lam))
        if w is None:
            continue
        wv = w.to_numpy()
        ret, vol, sharpe = _metricas(wv, mu_v, cov_v, rf)
        clave = (round(ret, 6), round(vol, 6))
        if clave in vistos:
            continue
        vistos.add(clave)
        fila = {"retorno": ret, "volatilidad": vol, "sharpe": sharpe}
        fila.update({f"peso·{a}": float(p) for a, p in zip(activos, wv)})
        filas.append(fila)
    if not filas:
        raise ErrorOptimizacion("OPTIMIZACION", "No se pudo construir la frontera eficiente.")
    puntos = pd.DataFrame(filas).sort_values("volatilidad").reset_index(drop=True)

    nube = _nube_factible(mu_v, cov_v, restr, n, cfg, activos, rf)

    return ResultadoFrontera(
        puntos=puntos,
        nube_factible=nube,
        minima_varianza_pesos=w_mv,
        maximo_sharpe_pesos=w_ms,
    )


def _nube_factible(
    mu_v: np.ndarray,
    cov_v: np.ndarray,
    restr: Restricciones,
    n: int,
    cfg: Configuracion,
    activos: list[str],
    rf: float,
) -> pd.DataFrame:
    """Carteras factibles aleatorias (Dirichlet) para el fondo de densidad."""
    rng = np.random.default_rng(cfg.semilla)
    tope = restr.peso_maximo
    suelo = restr.peso_minimo
    muestras = rng.dirichlet(np.ones(n), size=cfg.n_carteras_factibles * 3)
    mask = np.ones(muestras.shape[0], dtype=bool)
    if tope is not None:
        mask &= (muestras <= tope + 1e-12).all(axis=1)
    if suelo > 0:
        mask &= (muestras >= suelo - 1e-12).all(axis=1)
    muestras = muestras[mask][: cfg.n_carteras_factibles]
    if muestras.shape[0] == 0:
        muestras = rng.dirichlet(np.ones(n), size=2000)  # respaldo sin filtro
    ret = muestras @ mu_v
    var = np.einsum("ij,jk,ik->i", muestras, cov_v, muestras)
    vol = np.sqrt(np.clip(var, 0.0, None))
    sharpe = np.where(vol > 0, (ret - rf) / vol, 0.0)
    return pd.DataFrame({"retorno": ret, "volatilidad": vol, "sharpe": sharpe})

"""Frontera eficiente Media-Varianza restringida.

Pregunta 3 (mapa): para una rejilla densa de retornos objetivo entre la cartera
de mínima varianza y el máximo retorno factible, se halla la mínima volatilidad
alcanzable respetando las restricciones duras. La nube de fondo muestra carteras
factibles aleatorias para entender el universo completo; la línea es la frontera
Media-Varianza de referencia.

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

    # Frontera real por retornos objetivo: evita que una escala concreta de λ
    # deje fuera extremos cuando μ domina numéricamente a la varianza.
    retorno_min_var = float(w_mv.reindex(activos).to_numpy(dtype=float) @ mu_v)
    _, retorno_max = opt.rango_retorno_factible(mu.reindex(activos), restr, n)
    objetivos = np.linspace(
        min(retorno_min_var, retorno_max),
        max(retorno_min_var, retorno_max),
        max(int(cfg.n_puntos_frontera), 2),
    )
    filas = []
    vistos: set[tuple[float, ...]] = set()

    def agregar(w: pd.Series | None) -> None:
        if w is None:
            return
        wv = w.to_numpy()
        clave = tuple(np.round(wv, 8))
        if clave in vistos:
            return
        vistos.add(clave)
        ret, vol, sharpe = _metricas(wv, mu_v, cov_v, rf)
        fila = {"retorno": ret, "volatilidad": vol, "sharpe": sharpe}
        fila.update({f"peso·{a}": float(p) for a, p in zip(activos, wv)})
        filas.append(fila)

    agregar(w_mv)
    for objetivo in objetivos:
        w = opt.minima_varianza_para_retorno(mu, cov_estructural, restr, float(objetivo))
        agregar(w)
    agregar(w_ms)

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
    datos = {"retorno": ret, "volatilidad": vol, "sharpe": sharpe}
    datos.update({f"peso·{a}": muestras[:, i] for i, a in enumerate(activos)})
    return pd.DataFrame(datos)

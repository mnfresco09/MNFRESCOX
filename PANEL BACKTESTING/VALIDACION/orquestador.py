"""Orquestación de la validación OOS (Fase 2).

Ejecuta CPCV y WFA usando dos funciones INYECTADAS, de modo que la lógica de
validación queda totalmente desacoplada del motor Rust y de Optuna:

  - `optimizar(train_idx) -> params`         (el "paso de ajuste" dentro del fold;
                                              envuelve a runner._optimizar_combinacion)
  - `evaluar(params, idx)  -> dict_metricas` (mide SIN reoptimizar)

Así este módulo es verificable con callbacks de prueba y, en producción, se le
pasan los callbacks reales que llaman al pipeline existente. La regla de oro de
la Fase 2: **se optimiza sobre train y se mide sobre un test que la optimización
nunca vio.**
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from VALIDACION import cpcv as _cpcv
from VALIDACION import distribucion as _dist
from VALIDACION import wfa as _wfa

Optimizar = Callable[[np.ndarray], dict]
Evaluar = Callable[[dict, np.ndarray], dict]


@dataclass(frozen=True)
class ResultadoCPCV:
    metrica: str
    valores_oos: list[float]
    valores_is: list[float]
    distribucion_oos: _dist.DistribucionOOS
    ratio_oos_is: float          # media OOS / media IS (degradación)
    params_por_fold: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ResultadoWFA:
    metrica: str
    valores_oos: list[float]
    valores_is: list[float]
    efficiency: float            # media OOS / media IS, acotada a [0, +)
    anchored: bool


def ejecutar_cpcv(
    n_obs: int,
    *,
    optimizar: Optimizar,
    evaluar: Evaluar,
    n_grupos: int = 6,
    k: int = 2,
    embargo: float = 0.01,
    duracion_trade: int = 1,
    metrica: str = "sharpe_ratio",
) -> ResultadoCPCV:
    """Corre CPCV completo y devuelve la distribución OOS de `metrica`."""
    folds = _cpcv.generar_folds(
        n_obs,
        n_grupos=n_grupos,
        k=k,
        embargo=embargo,
        duracion_trade=duracion_trade,
    )
    valores_oos: list[float] = []
    valores_is: list[float] = []
    params_por_fold: list[dict] = []

    for fold in folds:
        params = optimizar(fold.train_idx)
        params_por_fold.append(params)
        test_idx = _cpcv.indices_de_rangos(fold.test_rangos)
        m_oos = evaluar(params, test_idx)
        m_is = evaluar(params, fold.train_idx)
        valores_oos.append(_metrica(m_oos, metrica))
        valores_is.append(_metrica(m_is, metrica))

    distribucion_oos = _dist.resumir(valores_oos)
    ratio = _ratio(valores_oos, valores_is)
    return ResultadoCPCV(
        metrica=metrica,
        valores_oos=valores_oos,
        valores_is=valores_is,
        distribucion_oos=distribucion_oos,
        ratio_oos_is=ratio,
        params_por_fold=params_por_fold,
    )


def ejecutar_wfa(
    n_obs: int,
    *,
    optimizar: Optimizar,
    evaluar: Evaluar,
    n_ventanas: int = 5,
    fraccion_test: float = 0.2,
    anchored: bool = False,
    metrica: str = "sharpe_ratio",
) -> ResultadoWFA:
    """Corre Walk-Forward y devuelve la WFA efficiency (degradación OOS/IS)."""
    ventanas = _wfa.generar_ventanas(
        n_obs,
        n_ventanas=n_ventanas,
        fraccion_test=fraccion_test,
        anchored=anchored,
    )
    valores_oos: list[float] = []
    valores_is: list[float] = []
    for v in ventanas:
        params = optimizar(v.train_idx)
        valores_oos.append(_metrica(evaluar(params, v.test_idx), metrica))
        valores_is.append(_metrica(evaluar(params, v.train_idx), metrica))

    media_oos = float(np.mean(valores_oos)) if valores_oos else 0.0
    media_is = float(np.mean(valores_is)) if valores_is else 0.0
    return ResultadoWFA(
        metrica=metrica,
        valores_oos=valores_oos,
        valores_is=valores_is,
        efficiency=_wfa.wfa_efficiency(media_oos, media_is),
        anchored=anchored,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrica(metricas: dict, clave: str) -> float:
    valor = metricas.get(clave) if isinstance(metricas, dict) else None
    try:
        f = float(valor)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


def _ratio(oos: Sequence[float], is_: Sequence[float]) -> float:
    media_is = float(np.mean(is_)) if len(is_) else 0.0
    media_oos = float(np.mean(oos)) if len(oos) else 0.0
    if media_is <= 0.0 or not np.isfinite(media_is):
        return 0.0
    return media_oos / media_is

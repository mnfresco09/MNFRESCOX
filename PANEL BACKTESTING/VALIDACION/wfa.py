"""Walk-Forward Analysis (rolling y anchored) — complemento de CPCV.

CPCV da la distribución OOS; WFA responde a una pregunta distinta y también
valiosa: *¿sobrevive el edge avanzando en el tiempo a medida que cambian los
regímenes?* Se reporta la **WFA efficiency** (rendimiento OOS / IS): por debajo
de ~0.5 la estrategia está sobreajustada al periodo.

Módulo puramente geométrico (índices): produce las ventanas (train, test). La
optimización/evaluación real la hace `orquestador.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class VentanaWFA:
    """Una ventana walk-forward: train [t0, t1) seguido de test [t1, t2)."""

    indice: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def generar_ventanas(
    n_obs: int,
    *,
    n_ventanas: int = 5,
    fraccion_test: float = 0.2,
    anchored: bool = False,
) -> list[VentanaWFA]:
    """Genera ventanas walk-forward sobre [0, n_obs).

    Parameters
    ----------
    n_ventanas:
        Número de tramos de test consecutivos (pasos hacia adelante).
    fraccion_test:
        Tamaño de cada tramo de test como fracción del histórico.
    anchored:
        - False (rolling): el train es una ventana deslizante de tamaño fijo.
        - True (anchored): el train arranca siempre en 0 y crece.
    """
    if n_ventanas < 1:
        raise ValueError("n_ventanas debe ser >= 1.")
    if not (0.0 < fraccion_test < 1.0):
        raise ValueError("fraccion_test debe estar en (0, 1).")

    test_size = max(1, int(round(n_obs * fraccion_test)))
    train_size = n_obs - n_ventanas * test_size
    if train_size < test_size:
        raise ValueError(
            f"Configuración WFA inviable: con n_ventanas={n_ventanas} y "
            f"fraccion_test={fraccion_test} no queda train suficiente "
            f"(train={train_size}, test={test_size}). Reduce alguno."
        )

    ventanas: list[VentanaWFA] = []
    for i in range(n_ventanas):
        test_ini = train_size + i * test_size
        test_fin = min(n_obs, test_ini + test_size)
        train_ini = 0 if anchored else i * test_size
        train_idx = np.arange(train_ini, test_ini, dtype=np.int64)
        test_idx = np.arange(test_ini, test_fin, dtype=np.int64)
        if test_idx.size == 0:
            break
        ventanas.append(VentanaWFA(indice=i, train_idx=train_idx, test_idx=test_idx))
    return ventanas


def wfa_efficiency(rendimiento_oos: float, rendimiento_is: float) -> float:
    """Eficiencia walk-forward = rendimiento OOS / rendimiento IS.

    Por convención se acota a [0, +). Si el IS no es positivo, la eficiencia no
    está definida de forma útil y se devuelve 0 (no hay base contra la que medir
    degradación). > 0.6 es bueno; < 0.5 indica sobreajuste al periodo.
    """
    is_ = float(rendimiento_is)
    oos = float(rendimiento_oos)
    if not isfinite(is_) or is_ <= 0.0 or not isfinite(oos):
        return 0.0
    return max(0.0, oos / is_)

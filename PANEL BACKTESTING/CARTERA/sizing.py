"""Sizing por riesgo (Fase 6).

El colateral fijo arriesga notional constante con independencia del régimen de
volatilidad: arriesgas mucho más riesgo real en mercados volátiles que en
tranquilos, sin querer. El volatility targeting corrige eso dimensionando para
**riesgo constante**.
"""

from __future__ import annotations

from math import isfinite, sqrt

import numpy as np


def vol_ewma(retornos, *, lambda_: float = 0.94, factor_anual: float | None = None) -> float:
    """Volatilidad estimada por EWMA (RiskMetrics) de los retornos.

    Da más peso a los retornos recientes. `lambda_` es el factor de decaimiento
    (0.94 ≈ diario RiskMetrics). Si `factor_anual` se da (p. ej. sqrt(365)),
    devuelve la volatilidad anualizada.
    """
    if not (0.0 < lambda_ < 1.0):
        raise ValueError("lambda_ debe estar en (0, 1).")
    r = np.asarray(retornos, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    # Varianza EWMA recursiva inicializada con la varianza muestral.
    var = float(np.var(r)) if r.size > 1 else float(r[0] ** 2)
    media = float(r.mean())
    for x in r:
        var = lambda_ * var + (1.0 - lambda_) * (x - media) ** 2
    vol = sqrt(max(var, 0.0))
    if factor_anual is not None:
        vol *= float(factor_anual)
    return vol


def volatility_target_size(
    capital: float,
    vol_objetivo: float,
    vol_realizada: float,
    *,
    apalancamiento_max: float | None = None,
) -> float:
    """Tamaño de posición para arriesgar `vol_objetivo` de riesgo.

        tamaño = (vol_objetivo / vol_realizada) · capital

    Si la volatilidad realizada es 0 o no finita, devuelve 0 (no se puede
    dimensionar el riesgo). El resultado se acota opcionalmente a
    `apalancamiento_max · capital`.
    """
    cap = float(capital)
    vo = float(vol_objetivo)
    vr = float(vol_realizada)
    if vr <= 0.0 or not isfinite(vr) or cap <= 0.0:
        return 0.0
    tamano = (vo / vr) * cap
    if apalancamiento_max is not None:
        tamano = min(tamano, float(apalancamiento_max) * cap)
    return max(0.0, tamano)


def kelly_fraccional(
    retorno_esperado: float,
    varianza: float,
    *,
    fraccion: float = 0.5,
) -> float:
    """Fracción de Kelly (sizing óptimo) escalada por `fraccion`.

        f* = retorno_esperado / varianza

    NUNCA se usa full Kelly: la varianza del resultado es brutal. `fraccion`
    (típicamente 0.25–0.5) reduce el tamaño para domar esa varianza. Se acota a
    [0, fraccion] (sin apalancar por encima de la fracción objetivo).
    """
    var = float(varianza)
    if var <= 0.0 or not isfinite(var):
        return 0.0
    f_estrella = float(retorno_esperado) / var
    f = float(fraccion) * f_estrella
    return float(min(max(f, 0.0), float(fraccion)))

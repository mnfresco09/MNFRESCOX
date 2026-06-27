"""Estadísticos de la distribución de métricas OOS (Fase 2).

El output honesto de CPCV no es un número, es una distribución. Este módulo la
resume: media, desviación, percentiles (p5, p25, mediana, p75, p95) y la
fracción de trayectorias positivas. El p25 > 0 es el criterio verde del
protocolo para la distribución de Sharpe OOS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistribucionOOS:
    n: int
    media: float
    desviacion: float
    p5: float
    p25: float
    mediana: float
    p75: float
    p95: float
    fraccion_positiva: float
    minimo: float
    maximo: float

    def como_dict(self) -> dict:
        return {
            "n": self.n,
            "media": self.media,
            "desviacion": self.desviacion,
            "p5": self.p5,
            "p25": self.p25,
            "mediana": self.mediana,
            "p75": self.p75,
            "p95": self.p95,
            "fraccion_positiva": self.fraccion_positiva,
            "minimo": self.minimo,
            "maximo": self.maximo,
        }


def resumir(valores) -> DistribucionOOS:
    """Resume una colección de métricas OOS (p. ej. Sharpes de cada trayectoria)."""
    arr = np.asarray(list(valores), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No hay valores finitos para resumir la distribución OOS.")
    return DistribucionOOS(
        n=int(arr.size),
        media=float(arr.mean()),
        desviacion=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        p5=float(np.percentile(arr, 5)),
        p25=float(np.percentile(arr, 25)),
        mediana=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        p95=float(np.percentile(arr, 95)),
        fraccion_positiva=float((arr > 0.0).mean()),
        minimo=float(arr.min()),
        maximo=float(arr.max()),
    )

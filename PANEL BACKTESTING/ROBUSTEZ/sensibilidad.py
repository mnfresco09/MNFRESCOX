"""Sensibilidad a la fecha de inicio (Fase 5).

Desplaza el comienzo del backtest unos días/semanas y re-mide. Si los resultados
cambian drásticamente, la estrategia es frágil a las condiciones iniciales —
bandera roja. La evaluación real (correr el motor desde cada offset) se inyecta
como callback, de modo que la lógica de dispersión es pura y verificable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensibilidadFecha:
    offsets: list[int]
    valores: list[float]
    media: float
    desviacion: float
    coef_variacion: float       # desviacion / |media|; dispersión relativa
    fragil: bool                # True si la dispersión supera el umbral

    def resumen(self) -> str:
        estado = "FRÁGIL" if self.fragil else "estable"
        return (
            f"Sensibilidad a fecha inicio: {estado} "
            f"(media={self.media:.4g}, CV={self.coef_variacion:.2%})"
        )


def sensibilidad_fecha_inicio(
    offsets: Sequence[int],
    evaluar: Callable[[int], float],
    *,
    umbral_cv: float = 0.25,
) -> SensibilidadFecha:
    """Evalúa la métrica desplazando el inicio y mide su dispersión relativa.

    Parameters
    ----------
    offsets:
        Desplazamientos del inicio (en velas/días) a probar, p. ej. [0, 5, 10, 20].
    evaluar:
        `evaluar(offset) -> metrica`. En producción corre el backtest empezando
        `offset` velas más tarde y devuelve, p. ej., el Sharpe.
    umbral_cv:
        Coeficiente de variación por encima del cual se marca como frágil.
    """
    offsets_l = [int(o) for o in offsets]
    if not offsets_l:
        raise ValueError("offsets no puede estar vacío.")
    valores = [float(evaluar(o)) for o in offsets_l]
    arr = np.asarray(valores, dtype=np.float64)
    media = float(arr.mean())
    desv = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    cv = desv / abs(media) if media != 0.0 else float("inf")
    return SensibilidadFecha(
        offsets=offsets_l,
        valores=valores,
        media=media,
        desviacion=desv,
        coef_variacion=cv,
        fragil=cv > umbral_cv,
    )

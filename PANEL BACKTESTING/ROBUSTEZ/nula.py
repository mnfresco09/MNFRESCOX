"""Estrategia nula / línea base de ruido (Fase 5).

El control de laboratorio que casi nadie hace y que separa el rigor del
autoengaño: corre el MISMO pipeline sobre retornos barajados o entradas
aleatorias para establecer cuál es el rendimiento de "sin edge" **bajo tu propia
carga de testing múltiple**. Si tu estrategia real no supera claramente a la nula
procesada por la misma maquinaria, no tienes nada.

Este módulo aporta la parte estadística pura: generar la distribución nula y
contrastar el valor real contra ella (p-valor empírico). El "mismo pipeline" se
ejecuta inyectando la maquinaria real desde fuera.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContrasteNula:
    valor_real: float
    media_nula: float
    p95_nula: float
    p_valor: float          # fracción de la nula que iguala o supera al real
    supera: bool            # el real bate a la nula al nivel pedido

    def resumen(self) -> str:
        veredicto = "SUPERA" if self.supera else "NO SUPERA"
        return (
            f"{veredicto} la nula: real={self.valor_real:.4g}, "
            f"nula media={self.media_nula:.4g}, p95={self.p95_nula:.4g}, p={self.p_valor:.3f}"
        )


def barajar(retornos, *, seed: int | None = None) -> np.ndarray:
    """Devuelve una copia barajada de los retornos (destruye el orden temporal)."""
    rng = np.random.default_rng(seed)
    r = np.asarray(retornos, dtype=np.float64).copy()
    rng.shuffle(r)
    return r


def distribucion_nula(
    n_iter: int,
    generador: Callable[[np.random.Generator], float],
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Construye la distribución nula evaluando `generador` n_iter veces.

    `generador(rng) -> float` produce una realización de la métrica bajo la
    hipótesis nula (p. ej. el Sharpe de una secuencia de entradas aleatorias
    procesada por el pipeline real). Se mantiene genérico para no acoplarse al
    motor.
    """
    if n_iter < 1:
        raise ValueError("n_iter debe ser >= 1.")
    rng = np.random.default_rng(seed)
    return np.array([float(generador(rng)) for _ in range(n_iter)], dtype=np.float64)


def contrastar(
    valor_real: float,
    distribucion: np.ndarray,
    *,
    nivel: float = 0.95,
) -> ContrasteNula:
    """Contrasta el valor real contra la distribución nula (p-valor empírico).

    `supera` es True si el real está por encima del percentil `nivel` de la nula
    (por defecto, bate al 95% de las realizaciones de ruido).
    """
    dist = np.asarray(distribucion, dtype=np.float64)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        raise ValueError("La distribución nula no contiene valores finitos.")
    p_valor = float((dist >= float(valor_real)).mean())
    umbral = float(np.percentile(dist, nivel * 100.0))
    return ContrasteNula(
        valor_real=float(valor_real),
        media_nula=float(dist.mean()),
        p95_nula=umbral,
        p_valor=p_valor,
        supera=float(valor_real) > umbral,
    )

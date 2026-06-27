"""Función objetivo robusta (Fase 4).

El fallo no es solo *qué* métrica maximizas, es que maximizar cualquier métrica
in-sample hasta su pico es la definición de sobreajuste. Estos helpers reorientan
la optimización hacia regiones robustas:

  1. `score_meseta`          → premia la MEDIANA de un vecindario de parámetros
                               (meseta plana), no el mejor punto (pico estrecho).
  2. `limite_inferior_oos`   → optimiza sobre el percentil bajo de la distribución
                               OOS (p25), no sobre el pico IS.
  3. `penalizacion_turnover` → cada operación paga costes; penaliza el exceso.
  4. `penalizacion_complejidad` → más parámetros libres = más grados de libertad
                               para sobreajustar.
  5. `direcciones_multiobjetivo` / `vector_multiobjetivo` → frente de Pareto
                               (NSGA-II de Optuna) en vez de colapsar todo a un
                               escalar.

Son funciones PURAS. No reemplazan `puntuacion.calcular_score` (el camino caliente
intacto): se aplican como envoltura cuando se quiere optimización robusta, y se
documenta el punto de integración en `samplers.py`/`runner.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Orden canónico de los objetivos del frente de Pareto y sus direcciones.
OBJETIVOS_PARETO = ("rentabilidad", "drawdown", "turnover")
DIRECCIONES_PARETO = ("maximize", "minimize", "minimize")


def score_meseta(scores_vecindario: Sequence[float], *, estadistico: str = "mediana") -> float:
    """Resume el rendimiento de un VECINDARIO de parámetros en un único score.

    Un edge robusto vive en una meseta: varios valores cercanos rinden parecido,
    así que la mediana del vecindario es alta. Un artefacto de sobreajuste vive en
    un pico: solo el valor exacto funciona y la mediana del vecindario se desploma.
    Por eso se premia la mediana (o un percentil bajo), no el máximo.
    """
    arr = np.asarray(list(scores_vecindario), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if estadistico == "mediana":
        return float(np.median(arr))
    if estadistico == "media":
        return float(arr.mean())
    if estadistico == "min":
        return float(arr.min())
    if estadistico.startswith("p"):
        try:
            q = float(estadistico[1:])
        except ValueError as exc:
            raise ValueError(f"estadistico percentil mal formado: {estadistico!r}") from exc
        return float(np.percentile(arr, q))
    raise ValueError(f"estadistico no soportado: {estadistico!r}")


def limite_inferior_oos(valores_oos: Sequence[float], *, percentil: float = 25.0) -> float:
    """Límite inferior del rendimiento OOS (percentil bajo de la distribución CPCV).

    Optimizar sobre el p25 del Sharpe OOS premia estrategias robustamente buenas,
    no puntualmente espectaculares.
    """
    arr = np.asarray(list(valores_oos), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, percentil))


def penalizacion_turnover(
    score: float,
    n_trades: int,
    *,
    trades_objetivo: int = 100,
    factor: float = 0.0,
) -> float:
    """Resta al score una penalización proporcional al exceso de operaciones.

    Penaliza el turnover por encima de `trades_objetivo`. `factor` controla la
    intensidad (0 = sin penalización). El exceso se normaliza por el objetivo
    para que la penalización sea adimensional respecto al score.
    """
    if factor <= 0.0 or trades_objetivo <= 0:
        return float(score)
    exceso = max(0, int(n_trades) - int(trades_objetivo))
    return float(score) - factor * (exceso / float(trades_objetivo))


def penalizacion_complejidad(score: float, n_parametros: int, *, factor: float = 0.0) -> float:
    """Resta una penalización lineal por número de parámetros libres."""
    if factor <= 0.0:
        return float(score)
    return float(score) - factor * max(0, int(n_parametros))


def direcciones_multiobjetivo() -> tuple[str, ...]:
    """Direcciones para `optuna.create_study(directions=...)` con NSGA-II."""
    return DIRECCIONES_PARETO


def vector_multiobjetivo(metricas: dict) -> tuple[float, float, float]:
    """Construye el vector objetivo (rentabilidad, drawdown, turnover) desde métricas.

    Pensado para `study.optimize` multiobjetivo: rentabilidad a maximizar,
    drawdown y turnover a minimizar. Lee claves del dict de métricas del sistema.
    """
    rentabilidad = _num(metricas.get("sharpe_anualizado", metricas.get("roi_total", 0.0)))
    drawdown = abs(_num(metricas.get("max_drawdown", 0.0)))
    turnover = _num(metricas.get("trades_por_dia", metricas.get("total_trades", 0.0)))
    return (rentabilidad, drawdown, turnover)


def _num(valor) -> float:
    try:
        f = float(valor)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0

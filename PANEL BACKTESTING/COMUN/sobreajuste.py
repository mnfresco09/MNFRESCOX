"""Métricas anti-sobreajuste basadas en particiones combinatorias (Fase 3).

Aquí vive el **PBO — Probability of Backtest Overfitting** (Bailey, Borwein,
López de Prado & Zhu, 2017) calculado por **CSCV — Combinatorially Symmetric
Cross-Validation**.

El PBO responde a una pregunta directa y demoledora: *¿qué probabilidad hay de
que la configuración que elegí por ser la mejor in-sample rinda por debajo de la
mediana fuera de muestra?* Si esa probabilidad supera 0.5, tu "ganadora" es —más
probable que no— peor que la mediana fuera de muestra: una estrategia muerta.

El Deflated Sharpe Ratio y el Minimum Backtest Length viven en
`COMUN/estadistica.py` (familia Sharpe); aquí está la parte que necesita la
matriz de rendimiento trial × tiempo.

Diseño
------
Este módulo es matemática pura sobre NumPy: recibe una matriz de rendimiento por
observación y por configuración, y no sabe nada del motor ni de Polars. La
construcción de esa matriz (replay de todos los trials sobre la malla temporal)
se conecta en la Fase 2/CPCV; separar el cálculo permite verificarlo de forma
aislada.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite, log

import numpy as np


@dataclass(frozen=True)
class ResultadoPBO:
    """Resultado del cálculo de PBO por CSCV."""

    pbo: float                  # probabilidad de sobreajuste, en [0, 1]
    n_combinaciones: int        # nº de particiones IS/OOS evaluadas
    n_configuraciones: int      # nº de trials/estrategias (columnas)
    logits: np.ndarray          # logit del rango relativo OOS por combinación
    rangos_relativos: np.ndarray  # omega por combinación, en (0, 1)

    @property
    def es_sobreajuste(self) -> bool:
        """Heurística del protocolo: PBO > 0.5 ⇒ estrategia muerta."""
        return self.pbo > 0.5


def pbo_cscv(
    matriz_rendimiento: np.ndarray,
    *,
    s: int = 16,
) -> ResultadoPBO:
    """Calcula el PBO por CSCV sobre una matriz de rendimiento.

    Parameters
    ----------
    matriz_rendimiento:
        Array 2D de forma `(T, N)`: `T` observaciones (filas, p. ej. retornos por
        periodo) para cada una de las `N` configuraciones (columnas). Debe estar
        alineada en el tiempo: la fila `t` es el mismo instante para todas las
        columnas.
    s:
        Número de bloques temporales disjuntos (debe ser PAR). Se forman todas
        las `C(s, s/2)` particiones que asignan la mitad de los bloques a IS y la
        otra mitad a OOS. Valor típico: 16.

    Returns
    -------
    ResultadoPBO
        Con `pbo` = fracción de particiones en las que la mejor configuración
        IS queda por debajo de la mediana OOS.

    Notas
    -----
    La métrica de rendimiento usada para rankear es el **Sharpe por observación**
    (media / desviación típica de las filas seleccionadas), coherente con la
    convención del resto del sistema. Se calcula de forma exacta y eficiente a
    partir de sumas y sumas de cuadrados por bloque (cada partición es O(N·s)).
    """
    M = np.asarray(matriz_rendimiento, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"matriz_rendimiento debe ser 2D (T, N); recibido ndim={M.ndim}.")
    t_obs, n_cfg = M.shape
    if n_cfg < 2:
        raise ValueError("Se necesitan al menos 2 configuraciones (columnas) para el PBO.")
    if s < 2 or s % 2 != 0:
        raise ValueError(f"s debe ser un entero par >= 2; recibido s={s}.")
    if t_obs < s:
        raise ValueError(
            f"Pocas observaciones ({t_obs}) para {s} bloques. "
            f"Reduce s o usa una serie más larga."
        )

    bloques = _particionar_filas(t_obs, s)
    # Sumas por bloque (S, N) y sumas de cuadrados por bloque (S, N) y conteos (S,).
    sum_b = np.stack([M[ini:fin].sum(axis=0) for (ini, fin) in bloques])
    sumsq_b = np.stack([(M[ini:fin] ** 2).sum(axis=0) for (ini, fin) in bloques])
    cnt_b = np.array([fin - ini for (ini, fin) in bloques], dtype=np.float64)

    indices_bloques = range(s)
    logits: list[float] = []
    rangos: list[float] = []

    for combo_is in combinations(indices_bloques, s // 2):
        mask_is = np.zeros(s, dtype=bool)
        mask_is[list(combo_is)] = True

        sr_is = _sharpe_desde_sumas(sum_b[mask_is], sumsq_b[mask_is], cnt_b[mask_is])
        sr_oos = _sharpe_desde_sumas(sum_b[~mask_is], sumsq_b[~mask_is], cnt_b[~mask_is])

        mejor = int(np.argmax(sr_is))
        # Rango relativo (omega) de la mejor-IS dentro de OOS: fracción de
        # configuraciones que rinden peor o igual que ella fuera de muestra.
        rango = float((sr_oos <= sr_oos[mejor]).sum()) / (n_cfg + 1.0)
        rango = min(max(rango, 1.0 / (n_cfg + 1.0)), n_cfg / (n_cfg + 1.0))
        rangos.append(rango)
        logits.append(log(rango / (1.0 - rango)))

    logits_arr = np.array(logits, dtype=np.float64)
    # PBO = P(logit <= 0) = fracción donde la mejor-IS cae bajo la mediana OOS.
    pbo = float((logits_arr <= 0.0).mean()) if logits_arr.size else 0.0

    return ResultadoPBO(
        pbo=pbo,
        n_combinaciones=int(logits_arr.size),
        n_configuraciones=int(n_cfg),
        logits=logits_arr,
        rangos_relativos=np.array(rangos, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _particionar_filas(t_obs: int, s: int) -> list[tuple[int, int]]:
    """Divide [0, t_obs) en `s` bloques contiguos casi iguales [ini, fin)."""
    cortes = np.linspace(0, t_obs, s + 1, dtype=int)
    return [(int(cortes[i]), int(cortes[i + 1])) for i in range(s)]


def _sharpe_desde_sumas(
    sum_sel: np.ndarray, sumsq_sel: np.ndarray, cnt_sel: np.ndarray
) -> np.ndarray:
    """Sharpe por columna (media/desv. típica) a partir de sumas agregadas.

    Reconstruye media y varianza muestral exactas desde las sumas y sumas de
    cuadrados de los bloques seleccionados, sin recorrer las filas originales.
    Columnas con varianza nula o no finita devuelven Sharpe 0.
    """
    n = float(cnt_sel.sum())
    suma = sum_sel.sum(axis=0)
    suma_sq = sumsq_sel.sum(axis=0)
    media = suma / n
    # Varianza muestral: (Σx² − n·media²) / (n − 1).
    var = (suma_sq - n * media * media) / max(n - 1.0, 1.0)
    var = np.where(var > 0.0, var, np.nan)
    sharpe = media / np.sqrt(var)
    return np.where(np.isfinite(sharpe), sharpe, 0.0)

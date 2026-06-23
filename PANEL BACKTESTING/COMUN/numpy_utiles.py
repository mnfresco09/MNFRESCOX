"""Utilidades numpy compartidas por todas las capas.

Este modulo no depende de ninguna capa de dominio (ni Polars, ni MOTOR, ni
NUCLEO): solo numpy. Por eso puede importarse desde cualquier sitio sin crear
ciclos, incluido el puente Rust en `MOTOR/wrapper.py`.

Centraliza el patron repetido por todo el codigo de "devolver un ndarray
C-contiguo del dtype esperado, sin copiar si ya lo esta".
"""

from __future__ import annotations

import numpy as np


def a_contiguo(arr: np.ndarray, dtype) -> np.ndarray:
    """Devuelve `arr` como ndarray C-contiguo del `dtype` pedido.

    No copia si el array ya tiene el dtype correcto y es contiguo en memoria;
    en caso contrario hace una unica copia con `np.ascontiguousarray`.

        timestamps = a_contiguo(serie.to_numpy(), np.int64)
        precios    = a_contiguo(precios, np.float64)
    """
    arr = np.asarray(arr)
    objetivo = np.dtype(dtype)
    if arr.dtype == objetivo and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=objetivo)

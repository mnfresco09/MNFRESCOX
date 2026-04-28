"""Base de toda estrategia.

Cada estrategia define **sus propios indicadores** dentro de su script
(`generar_señales`, `generar_salidas`, `indicadores_para_grafica`). El
motor Rust sólo se ocupa de la simulación y de la gestión de capital;
los indicadores son territorio exclusivo de la estrategia.

Lo que aporta esta clase:

  - Buffers numpy estables sobre los que la estrategia opera (`self.close`,
    `self.high`, `self.low`, `self.open`, `self.volume`). Se inyectan con
    `bind(arrays)` antes de empezar la combinación y permanecen vivos
    durante toda ella, lo que hace que el cache pueda identificar series
    por identidad de buffer entre trials.

  - Un cache opcional (`self.cache`) para memoizar resultados pesados
    cuando Optuna repite parámetros enteros (TPE/QMC los reusan a menudo).
    Es la propia estrategia quien decide qué cacheables son significativos.
    El helper `self.memo("nombre", periodo, ..., calcular=lambda: ...)`
    encapsula el patrón típico.

  - Utilidades vectoriales puras para construir señales y máscaras
    (`serie_senales`, `shift`). No imponen ningún cálculo de indicador.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import polars as pl


class CacheIndicadores:
    """Cache key→resultado, vivo lo que dura una combinación (un par
    `activo×timeframe`). Es opcional: una estrategia puede ignorarlo y
    recalcular siempre.
    """

    __slots__ = ("_cache",)

    def __init__(self) -> None:
        self._cache: dict[tuple, Any] = {}

    def get(self, key: tuple) -> Optional[Any]:
        return self._cache.get(key)

    def put(self, key: tuple, value: Any) -> None:
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class BaseEstrategia(ABC):
    """Contrato:

      - `ID` (int único) y `NOMBRE` (str).
      - `espacio_busqueda(trial)` define el espacio Optuna.
      - `generar_señales(df, params) → pl.Series` (Int8: 1, -1, 0).
      - `generar_salidas(df, params) → pl.Series` (sólo si EXIT_TYPE='CUSTOM').
      - `indicadores_para_grafica(df, params) → list[dict]` (opcional).

    REGLA ABSOLUTA: las señales no pueden mirar la vela actual ni el futuro.
    El motor entra siempre en el open de la vela siguiente.
    """

    ID: int
    NOMBRE: str
    COLUMNAS_REQUERIDAS: set[str] = set()

    def __init__(self) -> None:
        self._arrays = None
        self._cache: CacheIndicadores | None = None

    # ── Inyección desde el runner ─────────────────────────────────────────

    def bind(self, arrays, cache: CacheIndicadores | None = None) -> None:
        """Asigna buffers numpy y cache antes de la combinación. El runner
        llama a esto una vez por (estrategia × ctx), no por trial: los
        buffers son estables y el cache se reusa entre trials."""
        self._arrays = arrays
        self._cache = cache

    def desvincular(self) -> None:
        """Libera buffers y cache al terminar la combinación."""
        self._arrays = None
        self._cache = None

    # ── Acceso a buffers ──────────────────────────────────────────────────

    @property
    def open(self) -> np.ndarray:
        return self._req_arrays().opens

    @property
    def high(self) -> np.ndarray:
        return self._req_arrays().highs

    @property
    def low(self) -> np.ndarray:
        return self._req_arrays().lows

    @property
    def close(self) -> np.ndarray:
        return self._req_arrays().closes

    @property
    def volume(self) -> np.ndarray:
        return self._req_arrays().volumes

    @property
    def cache(self) -> CacheIndicadores | None:
        """Cache de la combinación. `None` si la estrategia se está usando
        fuera del runner (p. ej. en pruebas). La estrategia debe tolerar
        ambos casos."""
        return self._cache

    def _req_arrays(self):
        if self._arrays is None:
            raise RuntimeError(
                "Estrategia sin buffers vinculados. El runner debe llamar a "
                "estrategia.bind(arrays, cache) antes de generar señales."
            )
        return self._arrays

    # ── Memoización opcional ──────────────────────────────────────────────

    def memo(self, *clave_partes, calcular: Callable[[], Any]) -> Any:
        """Devuelve el resultado de `calcular()` cacheándolo bajo la clave
        `(nombre_estrategia, *clave_partes)`. Pensado para indicadores
        cuyo cálculo es caro y cuyos parámetros se repiten entre trials.

            ema21 = self.memo("ema", id(self.close), 21,
                              calcular=lambda: _calcula_ema(self.close, 21))

        Si no hay cache disponible, calcula directamente sin memoizar.
        """
        if self._cache is None:
            return calcular()
        key = (type(self).__name__, *clave_partes)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = calcular()
        self._cache.put(key, result)
        return result

    # ── Utilidades vectoriales para construir señales/salidas ─────────────

    @staticmethod
    def serie_senales(longitud: int, mascara_long: np.ndarray, mascara_short: np.ndarray) -> pl.Series:
        """Construye una `pl.Series(int8)` a partir de dos máscaras booleanas.
        Es el patrón canónico para devolver señales y salidas custom.
        """
        arr = np.zeros(int(longitud), dtype=np.int8)
        arr[mascara_long] = 1
        arr[mascara_short] = -1
        return pl.Series("senal", arr)

    @staticmethod
    def shift(arr: np.ndarray, n: int = 1, fill: float = float("nan")) -> np.ndarray:
        """Equivalente vectorial a `pl.Series.shift(n)` con relleno NaN."""
        if n == 0:
            return arr
        out = np.empty_like(arr, dtype=np.float64)
        if n > 0:
            out[:n] = fill
            out[n:] = arr[:-n]
        else:
            k = -n
            out[-k:] = fill
            out[:-k] = arr[k:]
        return out

    # ── API obligatoria de la estrategia ──────────────────────────────────

    @abstractmethod
    def espacio_busqueda(self, trial) -> dict:
        """Define el espacio Optuna del trial."""

    def parametros_por_defecto(self) -> dict:
        """Parámetros fijos para una ejecución sin Optuna."""
        return {}

    @abstractmethod
    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        """Devuelve `pl.Series` Int8 con valores en {-1, 0, 1}."""

    def generar_salidas(self, df: pl.DataFrame, params: dict) -> pl.Series:
        """Sólo si EXIT_TYPE='CUSTOM'. Devuelve `pl.Series` Int8."""
        raise NotImplementedError(
            f"{self.__class__.__name__} no define generar_salidas para EXIT_TYPE='CUSTOM'."
        )

    def indicadores_para_grafica(self, df: pl.DataFrame, params: dict) -> list[dict]:
        """Indicadores para el reporte HTML. Sobrescribir si interesa."""
        return []

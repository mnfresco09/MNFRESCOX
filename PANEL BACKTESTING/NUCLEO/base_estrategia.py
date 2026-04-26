from abc import ABC, abstractmethod
import polars as pl


class BaseEstrategia(ABC):
    """
    Clase madre de todas las estrategias.

    Contrato obligatorio para cada estrategia hija:
      - ID     : entero único en todo el sistema
      - NOMBRE : texto descriptivo
      - espacio_busqueda(trial) → dict con los parámetros del trial
      - generar_señales(df, params) → pl.Series de Señal (por vela)
      - generar_salidas(df, params) → pl.Series Int8 si se usa EXIT_TYPE="CUSTOM"

    REGLA ABSOLUTA: generar_señales solo puede usar datos de velas
    anteriores a la actual. La entrada siempre ocurre en el open
    de la vela siguiente a la señal — esto lo garantiza el motor.

    Para salidas CUSTOM, generar_salidas usa el mismo histórico OHLCV y
    devuelve:
      0  → no cerrar
      1  → cerrar LONG abierto
     -1  → cerrar SHORT abierto

    La salida CUSTOM se ejecuta al close de la vela donde aparece la
    condición. El SL de seguridad sigue teniendo prioridad dentro del motor.
    """

    ID: int
    NOMBRE: str
    COLUMNAS_REQUERIDAS: set[str] = set()

    @abstractmethod
    def espacio_busqueda(self, trial) -> dict:
        """
        Define los parámetros y sus rangos para Optuna.
        Recibe un optuna.Trial y devuelve un dict {nombre: valor}.
        """

    def parametros_por_defecto(self) -> dict:
        """
        Parametros fijos para una ejecucion simple sin Optuna.
        Las estrategias pueden sobrescribirlo para la fase de prueba end-to-end.
        """
        return {}

    @abstractmethod
    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        """
        Recibe el DataFrame OHLCV y los parámetros del trial.
        Devuelve una pl.Series de tipo Int8 con valores Señal:
          0  → sin señal
          1  → señal LONG
         -1  → señal SHORT
        """

    def generar_salidas(self, df: pl.DataFrame, params: dict) -> pl.Series:
        """
        Contrato opcional para EXIT_TYPE="CUSTOM".

        Debe devolver una pl.Series Int8 con la misma longitud que df:
          0  → no cerrar
          1  → cerrar LONG abierto
         -1  → cerrar SHORT abierto

        Igual que las señales, no puede usar información futura. Si una
        estrategia no implementa este método, no puede ejecutarse con CUSTOM.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no define generar_salidas para EXIT_TYPE='CUSTOM'."
        )

    # ------------------------------------------------------------------
    # Indicadores vectorizados (disponibles para todas las estrategias)
    # ------------------------------------------------------------------

    def sma(self, serie: pl.Series, periodo: int) -> pl.Series:
        """Media móvil simple."""
        return serie.rolling_mean(window_size=periodo)

    def ema(self, serie: pl.Series, periodo: int) -> pl.Series:
        """Media móvil exponencial (factor de suavizado = 2 / (periodo + 1))."""
        alpha = 2.0 / (periodo + 1)
        return serie.ewm_mean(alpha=alpha, adjust=False)

    def rsi(self, serie: pl.Series, periodo: int) -> pl.Series:
        """
        RSI con suavizado de Wilder (EWM con alpha = 1/periodo).
        Los primeros 'periodo' valores serán nulos.
        """
        delta = serie.diff()
        ganancia = delta.clip(lower_bound=0)
        perdida  = (-delta).clip(lower_bound=0)

        alpha = 1.0 / periodo
        media_gan = ganancia.ewm_mean(alpha=alpha, adjust=False)
        media_per = perdida.ewm_mean(alpha=alpha, adjust=False)

        rs  = media_gan / media_per
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Anular las primeras 'periodo' filas: el RSI no es fiable sin historia suficiente
        mascara = pl.Series([None] * periodo + [1] * (len(rsi) - periodo), dtype=pl.Float64)
        return rsi * mascara

    def atr(self, df: pl.DataFrame, periodo: int) -> pl.Series:
        """Average True Range con suavizado de Wilder."""
        prev_close = df["close"].shift(1)
        tr = pl.Series([
            max(h - l, abs(h - pc), abs(l - pc))
            for h, l, pc in zip(df["high"], df["low"], prev_close)
        ])
        alpha = 1.0 / periodo
        return tr.ewm_mean(alpha=alpha, adjust=False)

    def bollinger(
        self,
        serie: pl.Series,
        periodo: int,
        desviaciones: float = 2.0,
    ) -> tuple[pl.Series, pl.Series, pl.Series]:
        """
        Bandas de Bollinger.
        Devuelve (banda_superior, media, banda_inferior).
        """
        media = serie.rolling_mean(window_size=periodo)
        std   = serie.rolling_std(window_size=periodo)
        return (
            media + desviaciones * std,
            media,
            media - desviaciones * std,
        )

import polars as pl
from NUCLEO.base_estrategia import BaseEstrategia
from NUCLEO.tipos import Señal


class RSIReversion(BaseEstrategia):
    """
    Estrategia de reversión a la media basada en RSI.

    Señal LONG:  el RSI estaba por debajo de 'sobreventa' y lo cruza hacia arriba.
    Señal SHORT: el RSI estaba por encima de 'sobrecompra' y lo cruza hacia abajo.

    La entrada ocurre en el open de la vela siguiente a la señal.
    """

    ID     = 1
    NOMBRE = "RSI Reversión"
    COLUMNAS_REQUERIDAS = set()

    def parametros_por_defecto(self) -> dict:
        return {
            "rsi_periodo": 14,
            "sobreventa": 30,
            "sobrecompra": 70,
        }

    def espacio_busqueda(self, trial) -> dict:
        return {
            "rsi_periodo":  trial.suggest_int("rsi_periodo",  7,  28),
            "sobreventa":   trial.suggest_int("sobreventa",  20,  40),
            "sobrecompra":  trial.suggest_int("sobrecompra", 60,  80),
        }

    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        periodo    = params["rsi_periodo"]
        sobreventa = params["sobreventa"]
        sobrecompra = params["sobrecompra"]

        rsi      = self.rsi(df["close"], periodo)
        rsi_prev = rsi.shift(1)

        señal_long  = (rsi_prev < sobreventa)  & (rsi >= sobreventa)
        señal_short = (rsi_prev > sobrecompra) & (rsi <= sobrecompra)

        señales = pl.Series(
            [Señal.NINGUNA] * len(df),
            dtype=pl.Int8,
        )
        señales = señales.scatter(señal_long.arg_true(),  Señal.LONG)
        señales = señales.scatter(señal_short.arg_true(), Señal.SHORT)

        return señales

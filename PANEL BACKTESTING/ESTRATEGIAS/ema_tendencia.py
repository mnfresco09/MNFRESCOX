import polars as pl

from NUCLEO.base_estrategia import BaseEstrategia
from NUCLEO.tipos import Señal


class EMATendencia(BaseEstrategia):
    """
    Estrategia tendencial por cruce de medias exponenciales.

    Señal LONG: la EMA rapida cruza por encima de la EMA lenta.
    Señal SHORT: la EMA rapida cruza por debajo de la EMA lenta.
    """

    ID = 2
    NOMBRE = "EMA Tendencia"
    COLUMNAS_REQUERIDAS = set()

    def parametros_por_defecto(self) -> dict:
        return {
            "ema_rapida": 21,
            "ema_lenta": 89,
        }

    def espacio_busqueda(self, trial) -> dict:
        return {
            "ema_rapida": trial.suggest_int("ema_rapida", 8, 40),
            "ema_lenta": trial.suggest_int("ema_lenta", 50, 180),
        }

    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        ema_rapida = self.ema(df["close"], int(params["ema_rapida"]))
        ema_lenta = self.ema(df["close"], int(params["ema_lenta"]))

        diferencia = ema_rapida - ema_lenta
        diferencia_previa = diferencia.shift(1)

        señal_long = (diferencia_previa <= 0) & (diferencia > 0)
        señal_short = (diferencia_previa >= 0) & (diferencia < 0)

        señales = pl.Series([Señal.NINGUNA] * len(df), dtype=pl.Int8)
        señales = señales.scatter(señal_long.arg_true(), Señal.LONG)
        señales = señales.scatter(señal_short.arg_true(), Señal.SHORT)
        return señales

    def generar_salidas(self, df: pl.DataFrame, params: dict) -> pl.Series:
        ema_rapida = self.ema(df["close"], int(params["ema_rapida"]))
        ema_lenta = self.ema(df["close"], int(params["ema_lenta"]))

        diferencia = ema_rapida - ema_lenta
        diferencia_previa = diferencia.shift(1)

        salida_long = ((diferencia_previa >= 0) & (diferencia < 0)).fill_null(False)
        salida_short = ((diferencia_previa <= 0) & (diferencia > 0)).fill_null(False)

        salidas = pl.Series([0] * len(df), dtype=pl.Int8)
        salidas = salidas.scatter(salida_long.arg_true(), 1)
        salidas = salidas.scatter(salida_short.arg_true(), -1)
        return salidas

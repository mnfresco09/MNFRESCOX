"""
Exporta el DataFrame final a Parquet con compresión snappy.
Usa escritura atómica (archivo .tmp + rename) para garantizar
que nunca quede un Parquet parcial en HISTORICO/.
"""
import shutil
from pathlib import Path

import polars as pl

from .utils import get_logger

log = get_logger(__name__)

# Orden canónico de columnas en el Parquet final
ORDEN_COLUMNAS = [
    "timestamp",
    "open", "high", "low", "close",
    "volume", "quote_volume", "num_trades",
    "taker_buy_volume", "taker_buy_quote_volume",
    "taker_sell_volume", "vol_delta",
    "premium_close",
    "predicted_funding_rate",
]


def _calcular_predicted_funding_rate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula el funding rate previsto vela a vela siguiendo la formula exacta de Binance.

    Para cada vela:
      1. Identificar el inicio del periodo de funding al que pertenece.
         Los periodos son fijos: 00:00, 08:00 y 16:00 UTC.
         inicio_periodo = floor(hora_UTC / 8h) * 8h

      2. Calcular el TWAP ponderado del premium_close desde el inicio
         del periodo hasta esa vela inclusive.
         Peso de la vela N dentro del periodo = N (la primera vale 1, la segunda 2, etc.)
         TWAP = sum(premium[i] * peso[i]) / sum(peso[i])

      3. Aplicar la formula de Binance:
         predicted_FR = TWAP + clamp(0.0001 - TWAP, -0.0005, 0.0005)
         donde 0.0001 es el interest rate fijo de 0.01% y
         0.0005 es el damper de 0.05%

    El resultado coincide con el predicted funding rate que Binance muestra
    en tiempo real si los premiumIndexKlines están alineados en UTC.
    """
    INTEREST_RATE = 0.0001
    DAMPER = 0.0005

    ocho_horas_us = 8 * 60 * 60 * 1_000_000
    un_minuto_us = 60 * 1_000_000
    ts_us = pl.col("timestamp").cast(pl.Int64)

    df = df.with_columns([
        ((ts_us // ocho_horas_us) * ocho_horas_us).alias("_inicio_periodo"),
    ])

    df = df.with_columns([
        (((ts_us - pl.col("_inicio_periodo")) // un_minuto_us + 1).alias("_posicion")),
    ])

    df = df.with_columns([
        (pl.col("premium_close") * pl.col("_posicion")).alias("_weighted_premium"),
    ])

    df = df.with_columns([
        pl.col("_weighted_premium")
        .cum_sum()
        .over("_inicio_periodo")
        .alias("_cum_weighted_premium"),

        (pl.col("_posicion") * (pl.col("_posicion") + 1) / 2).alias("_cum_posiciones"),
    ])

    df = df.with_columns([
        (pl.col("_cum_weighted_premium") / pl.col("_cum_posiciones")).alias("_twap"),
    ])

    df = df.with_columns([
        (
            pl.col("_twap")
            + (pl.lit(INTEREST_RATE) - pl.col("_twap")).clip(
                lower_bound=-DAMPER,
                upper_bound=DAMPER,
            )
        ).alias("predicted_funding_rate"),
    ])

    return df.drop([
        "_inicio_periodo",
        "_posicion",
        "_weighted_premium",
        "_cum_weighted_premium",
        "_cum_posiciones",
        "_twap",
    ])


def guardar(df: pl.DataFrame, destino: Path) -> None:
    """
    Guarda df en Parquet snappy en destino.
    Escribe primero a .tmp.parquet y renombra al final para garantizar atomicidad.
    Si falla, el .tmp es eliminado y el destino anterior no se toca.
    """
    # 1. Columnas derivadas de order flow
    df = df.with_columns([
        (pl.col("volume") - pl.col("taker_buy_volume")).alias("taker_sell_volume"),
        (pl.col("taker_buy_volume") - (pl.col("volume") - pl.col("taker_buy_volume"))).alias("vol_delta"),
    ])

    # 2. Funding rate previsto vela a vela
    df = _calcular_predicted_funding_rate(df)

    # 3. Validaciones
    _validar_microestructura(df)

    # 4. Verificar columnas y ordenar
    cols_faltantes = [c for c in ORDEN_COLUMNAS if c not in df.columns]
    if cols_faltantes:
        raise ValueError(f"Columnas esperadas no encontradas: {cols_faltantes}")

    df = df.select(ORDEN_COLUMNAS)

    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp.parquet")

    try:
        df.write_parquet(tmp, compression="snappy")
        shutil.move(str(tmp), str(destino))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    tam_mb = destino.stat().st_size / 1_048_576
    log.info(f"Parquet guardado: {destino.name}  ({tam_mb:.1f} MB, {len(df):,} filas)")


def _validar_microestructura(df: pl.DataFrame) -> None:
    # taker_sell_volume >= 0 (si es negativo, taker_buy_volume > volume en origen)
    mask_neg = df["taker_sell_volume"] < 0
    if mask_neg.any():
        bad_ts = df.filter(mask_neg)["timestamp"][0]
        raise ValueError(
            f"taker_sell_volume negativo en {bad_ts}: "
            f"taker_buy_volume supera a volume en los datos de origen"
        )

    # |vol_delta| <= volume (si lo supera, igualmente indica corrupción en origen)
    mask_delta = df["vol_delta"].abs() > df["volume"]
    if mask_delta.any():
        bad_ts = df.filter(mask_delta)["timestamp"][0]
        raise ValueError(
            f"|vol_delta| > volume en {bad_ts}: "
            f"los datos de taker_buy_volume son inconsistentes con volume"
        )

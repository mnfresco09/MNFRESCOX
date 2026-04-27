"""
Detección y normalización de timestamps.
Futuros USD-M siempre en milisegundos (13 dígitos).
Salida siempre: Polars Datetime(us, UTC).
"""
import polars as pl


def detectar_precision(valor: int) -> str:
    """Detecta la precisión por el número de dígitos del entero."""
    digitos = len(str(abs(int(valor))))
    if digitos <= 13:
        return "ms"
    if digitos <= 16:
        return "us"
    raise ValueError(f"Timestamp con {digitos} dígitos no reconocido: {valor}")


def normalizar(serie: pl.Series) -> pl.Series:
    """
    Convierte una Serie de enteros (ms o us) a Datetime(us, UTC).
    El nombre de la serie de salida se preserva.
    """
    primer_valor = int(serie[0])
    precision = detectar_precision(primer_valor)

    us = serie.cast(pl.Int64) * (1000 if precision == "ms" else 1)
    return us.cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias(serie.name)

"""
Lectura de CSVs de Binance Vision y normalización de columnas.
Los CSV no tienen cabecera; los nombres se asignan según el tipo.
"""
import zipfile
from pathlib import Path

import polars as pl

from . import timestamps
from .utils import get_logger

log = get_logger(__name__)

# Nombres de columnas crudas por tipo (posición exacta en el CSV)
_COLS_CRUDAS = {
    "klines": [
        "timestamp", "open", "high", "low", "close", "volume",
        "_close_time", "quote_volume", "num_trades",
        "taker_buy_volume", "taker_buy_quote_volume", "_ignore",
    ],
    "premiumIndexKlines": [
        "timestamp", "open", "high", "low", "close", "volume",
        "_close_time", "_i1", "_i2", "_i3", "_i4",
    ],
}

# Columnas a conservar en el DataFrame final de cada tipo
_COLS_MANTENER = {
    "klines": [
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_volume", "num_trades", "taker_buy_volume", "taker_buy_quote_volume",
    ],
    "premiumIndexKlines": ["timestamp", "open", "high", "low", "close"],
}


def extraer_csv(zip_path: Path, destino_dir: Path) -> Path:
    """Extrae el único CSV del ZIP al directorio destino y retorna su ruta."""
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"No hay CSV en {zip_path.name}")
        nombre_csv = csvs[0]
        zf.extract(nombre_csv, destino_dir)
    return destino_dir / nombre_csv


def parsear_csv(csv_path: Path, tipo: str) -> pl.DataFrame:
    """
    Lee el CSV, asigna nombres de columnas, normaliza tipos.
    Retorna DataFrame con columnas finales listas para combinar.
    """
    nombres = _COLS_CRUDAS[tipo]

    df = pl.read_csv(
        csv_path,
        has_header=False,
        new_columns=nombres,
        infer_schema_length=0,      # todo como Utf8 para control total
        truncate_ragged_lines=True,  # tolerancia a líneas extra
    )

    if df.is_empty():
        return _df_vacio(tipo)

    # Quitar fila de cabecera si la primera celda de timestamp no es numérica
    primer_ts = df[0, "timestamp"].strip()
    if not primer_ts.lstrip("-").isdigit():
        df = df.slice(1)

    if df.is_empty():
        return _df_vacio(tipo)

    # Normalizar timestamp: entero -> Datetime(us, UTC)
    df = df.with_columns(
        timestamps.normalizar(df["timestamp"].cast(pl.Int64)).alias("timestamp")
    )

    # Conservar solo las columnas necesarias
    df = df.select(_COLS_MANTENER[tipo])

    # Cast tipos finales
    float_cols = [c for c in df.columns if c not in ("timestamp", "num_trades")]
    int_cols   = ["num_trades"] if "num_trades" in df.columns else []

    cast_exprs = (
        [pl.col(c).cast(pl.Float64) for c in float_cols]
        + [pl.col(c).cast(pl.Int64) for c in int_cols]
    )
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def _df_vacio(tipo: str) -> pl.DataFrame:
    """DataFrame vacío con el esquema correcto para el tipo dado."""
    schema = {}
    for col in _COLS_MANTENER[tipo]:
        if col == "timestamp":
            schema[col] = pl.Datetime("us", "UTC")
        elif col == "num_trades":
            schema[col] = pl.Int64
        else:
            schema[col] = pl.Float64
    return pl.DataFrame(schema=schema)

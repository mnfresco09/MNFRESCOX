"""Carga estricta de cierres desde los Parquet propios del panel."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorDatos
from CONTRATOS.rutas import nombre_archivo_historico


def _leer_tabla(archivo: Path, ticker: str) -> pd.DataFrame:
    if not archivo.exists():
        raise ErrorDatos(
            "DATOS",
            f"No existe el histórico de '{ticker}' ({archivo.name}).",
        )
    try:
        tabla = pd.read_parquet(archivo)
    except Exception as exc:
        raise ErrorDatos(
            "DATOS",
            f"No se pudo leer '{archivo.name}': {exc}",
        ) from exc
    if tabla.empty:
        raise ErrorDatos("DATOS", f"'{archivo.name}' está vacío.")
    if tuple(tabla.columns) != ("fecha", "cierre"):
        raise ErrorDatos(
            "DATOS",
            f"'{archivo.name}' tiene un esquema inesperado; se exige fecha/cierre.",
        )
    return tabla


def _convertir_serie(tabla: pd.DataFrame, ticker: str, archivo: Path) -> pd.Series:
    try:
        fechas = pd.DatetimeIndex(pd.to_datetime(tabla["fecha"], errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ErrorDatos(
            "DATOS",
            f"'{archivo.name}' contiene fechas inválidas.",
        ) from exc
    if fechas.hasnans:
        raise ErrorDatos("DATOS", f"'{archivo.name}' contiene fechas nulas.")
    if fechas.tz is not None:
        raise ErrorDatos(
            "DATOS",
            f"'{archivo.name}' contiene fechas con zona horaria.",
        )
    if fechas.has_duplicates:
        raise ErrorDatos(
            "DATOS",
            f"'{archivo.name}' contiene fechas duplicadas.",
        )
    if not fechas.is_monotonic_increasing:
        raise ErrorDatos(
            "DATOS",
            f"'{archivo.name}' contiene fechas desordenadas.",
        )
    try:
        valores = pd.to_numeric(tabla["cierre"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ErrorDatos(
            "DATOS",
            f"'{archivo.name}' contiene cierres no numéricos.",
        ) from exc
    if valores.isna().any():
        raise ErrorDatos("DATOS", f"'{archivo.name}' contiene cierres nulos.")
    if not np.isfinite(valores.to_numpy()).all():
        raise ErrorDatos("DATOS", f"'{archivo.name}' contiene cierres no finitos.")
    if (valores <= 0).any():
        raise ErrorDatos(
            "DATOS",
            f"'{archivo.name}' contiene cierres no positivos.",
        )
    if len(valores) < 2:
        raise ErrorDatos(
            "DATOS",
            f"'{archivo.name}' contiene menos de dos observaciones.",
        )
    return pd.Series(valores.to_numpy(), index=fechas, name=ticker)


def cargar_cierres(
    tickers: Sequence[str],
    carpeta_historico: Path,
) -> dict[str, pd.Series]:
    """Devuelve cierres validados sin ordenar, rellenar ni eliminar filas."""

    carpeta = Path(carpeta_historico)
    nombres = tuple(nombre_archivo_historico(ticker) for ticker in tickers)
    if len(set(nombres)) != len(nombres):
        raise ErrorDatos(
            "DATOS",
            "Dos tickers distintos apuntan al mismo archivo histórico.",
        )
    cierres: dict[str, pd.Series] = {}
    for ticker, nombre in zip(tickers, nombres, strict=True):
        archivo = carpeta / nombre
        cierres[ticker] = _convertir_serie(
            _leer_tabla(archivo, ticker),
            ticker,
            archivo,
        )
    return cierres

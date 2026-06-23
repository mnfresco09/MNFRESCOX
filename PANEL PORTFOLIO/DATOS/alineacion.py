"""Intersección de calendarios y cálculo de log-retornos."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorDatos
from CONTRATOS.modelos import DatosAlineados


def _validar_series(cierres: Mapping[str, pd.Series]) -> None:
    if len(cierres) < 2:
        raise ErrorDatos("DATOS", "Se requieren al menos dos activos.")
    for ticker, serie in cierres.items():
        if not isinstance(serie, pd.Series) or serie.empty:
            raise ErrorDatos("DATOS", f"'{ticker}' no contiene una serie válida.")
        if not isinstance(serie.index, pd.DatetimeIndex):
            raise ErrorDatos("DATOS", f"'{ticker}' no usa un índice de fechas.")
        if serie.index.has_duplicates:
            raise ErrorDatos("DATOS", f"'{ticker}' contiene fechas duplicadas.")
        if not serie.index.is_monotonic_increasing:
            raise ErrorDatos("DATOS", f"'{ticker}' contiene fechas desordenadas.")
        valores = serie.to_numpy(dtype=float)
        if not np.isfinite(valores).all() or (valores <= 0).any():
            raise ErrorDatos("DATOS", f"'{ticker}' contiene cierres inválidos.")


def alinear_y_calcular_retornos(
    cierres: Mapping[str, pd.Series],
    min_retornos: int,
) -> DatosAlineados:
    """Intersecta fechas comunes; nunca rellena huecos ni fines de semana."""

    _validar_series(cierres)
    if min_retornos < 1:
        raise ErrorDatos("DATOS", "min_retornos debe ser positivo.")
    tabla = pd.concat(cierres, axis=1, join="inner")
    if tabla.empty or len(tabla) < 2:
        raise ErrorDatos("DATOS", "No hay suficientes cierres en el calendario común.")
    if tabla.index.has_duplicates or not tabla.index.is_monotonic_increasing:
        raise ErrorDatos("DATOS", "El calendario común no es único y creciente.")
    if tabla.isna().any().any():
        raise ErrorDatos("DATOS", "La intersección contiene valores nulos.")

    log_retornos = np.log(tabla / tabla.shift(1)).iloc[1:]
    if not np.isfinite(log_retornos.to_numpy()).all():
        raise ErrorDatos("DATOS", "Los log-retornos contienen valores no finitos.")
    if len(log_retornos) < min_retornos:
        raise ErrorDatos(
            "DATOS",
            f"Retornos alineados insuficientes: {len(log_retornos)} < {min_retornos}.",
        )
    columnas_constantes = tuple(
        columna
        for columna in log_retornos.columns
        if log_retornos[columna].nunique(dropna=False) <= 1
    )
    if columnas_constantes:
        raise ErrorDatos(
            "DATOS",
            "Activos con retornos sin variación: "
            + ", ".join(columnas_constantes),
        )
    tabla.columns = list(cierres)
    log_retornos.columns = list(cierres)
    return DatosAlineados(
        activos=tuple(cierres),
        cierres=tabla,
        log_retornos=log_retornos,
    )


def recortar_datos(datos: DatosAlineados, n_retornos: int) -> DatosAlineados:
    """Extrae una ventana final con n retornos y n+1 cierres."""

    if n_retornos < 1:
        raise ErrorDatos("DATOS", "n_retornos debe ser positivo.")
    disponibles = len(datos.log_retornos)
    if disponibles < n_retornos:
        raise ErrorDatos(
            "DATOS",
            f"Ventana actual insuficiente: {disponibles} < {n_retornos}.",
        )
    return DatosAlineados(
        activos=datos.activos,
        cierres=datos.cierres.iloc[-(n_retornos + 1) :].copy(),
        log_retornos=datos.log_retornos.iloc[-n_retornos:].copy(),
    )

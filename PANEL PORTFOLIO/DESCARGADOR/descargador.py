"""Descarga y publicación validada de cierres diarios.

Esta capa no calcula retornos ni alinea calendarios. Descarga cada activo,
valida el contenido y publica la cesta como una transacción: si cualquier
activo o reemplazo falla, los históricos anteriores permanecen intactos.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorDatos
from CONTRATOS.modelos import ResumenActivo
from CONTRATOS.rutas import nombre_archivo_historico

ProveedorCierres = Callable[[str, str, str], pd.Series | None]

UMBRAL_HUECO_DIAS = 7


@dataclass(frozen=True)
class _ActivoPreparado:
    resumen: ResumenActivo
    ruta_temporal: Path


def descargar_desde_yahoo(
    ticker: str,
    fecha_inicio: str,
    fecha_fin: str,
) -> pd.Series | None:
    """Descarga Adj Close y cae a Close solo si el ajustado no existe."""

    import yfinance as yf

    fin_exclusivo = (pd.Timestamp(fecha_fin) + timedelta(days=1)).date().isoformat()
    tabla = yf.download(
        ticker,
        start=fecha_inicio,
        end=fin_exclusivo,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if tabla is None or tabla.empty:
        return None
    if isinstance(tabla.columns, pd.MultiIndex):
        tabla.columns = tabla.columns.get_level_values(0)
    columna = (
        "Adj Close"
        if "Adj Close" in tabla.columns
        else "Close"
        if "Close" in tabla.columns
        else None
    )
    if columna is None:
        return None
    cierres = tabla[columna]
    if isinstance(cierres, pd.DataFrame):
        cierres = cierres.iloc[:, 0]
    cierres = cierres.copy()
    cierres.name = "cierre"
    return cierres


def _normalizar_indice(cierres: pd.Series, ticker: str) -> pd.Series:
    if not isinstance(cierres.index, pd.DatetimeIndex):
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' no devolvió un índice de fechas.",
        )
    normalizados = cierres.copy()
    indice = pd.DatetimeIndex(cierres.index)
    if indice.tz is not None:
        indice = indice.tz_convert(None)
    normalizados.index = indice
    return normalizados


def _validar_cierres(cierres: pd.Series | None, ticker: str) -> pd.Series:
    if cierres is None or not isinstance(cierres, pd.Series) or cierres.empty:
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' no devolvió cierres diarios.",
        )
    validados = _normalizar_indice(cierres, ticker)
    if len(validados) < 2:
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' devolvió menos de dos filas.",
        )
    if validados.index.has_duplicates:
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' contiene fechas duplicadas.",
        )
    if not validados.index.is_monotonic_increasing:
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' contiene fechas desordenadas.",
        )
    try:
        validados = pd.to_numeric(validados, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' contiene cierres no numéricos.",
        ) from exc
    if validados.isna().any():
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' contiene cierres nulos.",
        )
    if not np.isfinite(validados.to_numpy()).all():
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' contiene cierres no finitos.",
        )
    if (validados <= 0).any():
        raise ErrorDatos(
            "DESCARGADOR",
            f"'{ticker}' debe contener únicamente cierres positivos.",
        )
    validados.name = "cierre"
    return validados


def _analizar_huecos(indice: pd.DatetimeIndex) -> tuple[int, int]:
    diferencias = indice.to_series().diff().dropna().dt.days
    if diferencias.empty:
        return 0, 0
    return (
        int((diferencias > UMBRAL_HUECO_DIAS).sum()),
        int(diferencias.max()),
    )


def _preparar_activo(
    ticker: str,
    cierres: pd.Series | None,
    carpeta_temporal: Path,
) -> _ActivoPreparado:
    validados = _validar_cierres(cierres, ticker)
    nombre_archivo = nombre_archivo_historico(ticker)
    ruta_temporal = carpeta_temporal / nombre_archivo
    tabla = pd.DataFrame(
        {
            "fecha": validados.index,
            "cierre": validados.to_numpy(),
        }
    )
    tabla.to_parquet(ruta_temporal, index=False)
    comprobacion = pd.read_parquet(ruta_temporal)
    if list(comprobacion.columns) != ["fecha", "cierre"] or len(comprobacion) != len(
        tabla
    ):
        raise ErrorDatos(
            "DESCARGADOR",
            f"No se pudo verificar el Parquet temporal de '{ticker}'.",
        )
    huecos, hueco_maximo = _analizar_huecos(validados.index)
    return _ActivoPreparado(
        resumen=ResumenActivo(
            ticker=ticker,
            archivo=nombre_archivo,
            filas=len(validados),
            fecha_inicio=validados.index[0],
            fecha_fin=validados.index[-1],
            huecos_sospechosos=huecos,
            hueco_max_dias=hueco_maximo,
        ),
        ruta_temporal=ruta_temporal,
    )


def _publicar_preparados(
    preparados: Sequence[_ActivoPreparado],
    carpeta_historico: Path,
    carpeta_respaldos: Path,
) -> None:
    carpeta_respaldos.mkdir()
    respaldos: dict[Path, Path] = {}
    publicados: list[Path] = []
    try:
        for preparado in preparados:
            definitivo = carpeta_historico / preparado.resumen.archivo
            if definitivo.exists():
                respaldo = carpeta_respaldos / definitivo.name
                shutil.copy2(definitivo, respaldo)
                respaldos[definitivo] = respaldo
        for preparado in preparados:
            definitivo = carpeta_historico / preparado.resumen.archivo
            preparado.ruta_temporal.replace(definitivo)
            publicados.append(definitivo)
    except Exception as exc:
        for definitivo in publicados:
            if definitivo not in respaldos:
                definitivo.unlink(missing_ok=True)
        for definitivo, respaldo in respaldos.items():
            shutil.copy2(respaldo, definitivo)
        raise ErrorDatos(
            "DESCARGADOR",
            f"No se pudo publicar la cesta completa: {exc}",
        ) from exc


def descargar_cesta(
    tickers: Sequence[str],
    fecha_inicio: str,
    fecha_fin: str,
    carpeta_historico: Path,
    proveedor: ProveedorCierres = descargar_desde_yahoo,
) -> tuple[ResumenActivo, ...]:
    """Descarga, valida y publica una cesta completa."""

    nombres = tuple(nombre_archivo_historico(ticker) for ticker in tickers)
    if len(set(nombres)) != len(nombres):
        raise ErrorDatos(
            "DESCARGADOR",
            "Dos tickers distintos producen el mismo archivo histórico.",
        )
    carpeta = Path(carpeta_historico)
    carpeta.parent.mkdir(parents=True, exist_ok=True)
    carpeta.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".portfolio_descarga_", dir=carpeta.parent) as temporal:
        carpeta_temporal = Path(temporal)
        preparados: list[_ActivoPreparado] = []
        for ticker in tickers:
            try:
                cierres = proveedor(ticker, fecha_inicio, fecha_fin)
            except ErrorDatos:
                raise
            except Exception as exc:
                raise ErrorDatos(
                    "DESCARGADOR",
                    f"Falló la descarga de '{ticker}': {exc}",
                ) from exc
            preparados.append(_preparar_activo(ticker, cierres, carpeta_temporal))
        _publicar_preparados(
            preparados,
            carpeta,
            carpeta_temporal / "respaldos",
        )
    return tuple(preparado.resumen for preparado in preparados)


def imprimir_resumen(resumenes: Sequence[ResumenActivo]) -> None:
    """Imprime cobertura y huecos de cada activo descargado."""

    print("\n[DESCARGADOR] Resumen de descarga:")
    print(
        f"  {'TICKER':<14}{'FILAS':>8}  {'INICIO':<12}{'FIN':<12}"
        f"{'HUECOS':>8}{'MAX(d)':>8}"
    )
    for resumen in resumenes:
        print(
            f"  {resumen.ticker:<14}{resumen.filas:>8}  "
            f"{resumen.fecha_inicio.date().isoformat():<12}"
            f"{resumen.fecha_fin.date().isoformat():<12}"
            f"{resumen.huecos_sospechosos:>8}{resumen.hueco_max_dias:>8}"
        )
    print()

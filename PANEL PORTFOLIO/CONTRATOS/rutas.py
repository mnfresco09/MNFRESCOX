"""Convenciones compartidas de nombres y rutas del panel."""

from __future__ import annotations

import re


def slug_ticker(ticker: str) -> str:
    """Convierte un ticker de Yahoo en un nombre de archivo seguro."""

    slug = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").upper()
    return slug or "ACTIVO"


def nombre_archivo_historico(ticker: str) -> str:
    """Nombre canónico del histórico diario de un activo."""

    return f"{slug_ticker(ticker)}_1d.parquet"

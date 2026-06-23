"""Flujo inteligente de datos: usar lo descargado o refrescar si cambió la cesta.

Regla pedida: los tickers y fechas viven en CONFIGURACION. Si lo que hay en
HISTORICO coincide exactamente (mismos tickers y mismas fechas) y los Parquet
están presentes, se reutiliza sin volver a descargar. Si cambia cualquier ticker
o fecha, se BORRAN los Parquet anteriores y se descarga la nueva cesta.

DESCARGADOR es el único dueño de HISTORICO; DATOS sigue siendo puro y solo lee.
La descarga real ocurre en la máquina del usuario (yfinance); aquí se decide y
se registra el manifiesto.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from CONTRATOS.modelos import Configuracion, ResumenActivo
from CONTRATOS.rutas import nombre_archivo_historico

from .descargador import descargar_cesta

MANIFIESTO_DATOS = "manifiesto_datos.json"

Descargador = Callable[..., tuple[ResumenActivo, ...]]


def _clave(tickers: Sequence[str], fecha_inicio: str, fecha_fin: str) -> dict:
    return {
        "tickers": sorted(tickers),
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
    }


def _manifiesto_coincide(carpeta: Path, clave: dict) -> bool:
    ruta = carpeta / MANIFIESTO_DATOS
    if not ruta.exists():
        return False
    try:
        guardado = json.loads(ruta.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return guardado.get("clave") == clave


def _parquets_presentes(carpeta: Path, tickers: Sequence[str]) -> bool:
    return all((carpeta / nombre_archivo_historico(t)).exists() for t in tickers)


def _borrar_parquets(carpeta: Path) -> None:
    for parquet in carpeta.glob("*.parquet"):
        parquet.unlink()


def estado_cache(configuracion: Configuracion) -> str:
    """'cache' si se puede reutilizar; 'descarga' si hay que refrescar."""
    carpeta = Path(configuracion.carpeta_historico)
    clave = _clave(configuracion.tickers, configuracion.fecha_inicio, configuracion.fecha_fin)
    if _manifiesto_coincide(carpeta, clave) and _parquets_presentes(carpeta, configuracion.tickers):
        return "cache"
    return "descarga"


def asegurar_datos(
    configuracion: Configuracion,
    descargador: Descargador = descargar_cesta,
) -> tuple[str, tuple[ResumenActivo, ...] | None]:
    """Garantiza HISTORICO acorde a la configuración.

    Devuelve ('cache', None) si reutiliza, o ('descarga', resumenes) si refrescó.
    """
    carpeta = Path(configuracion.carpeta_historico)
    carpeta.mkdir(parents=True, exist_ok=True)
    clave = _clave(configuracion.tickers, configuracion.fecha_inicio, configuracion.fecha_fin)

    if estado_cache(configuracion) == "cache":
        return "cache", None

    # Cambió la cesta o faltan datos: borrar lo previo y descargar lo nuevo.
    _borrar_parquets(carpeta)
    resumenes = descargador(
        configuracion.tickers,
        configuracion.fecha_inicio,
        configuracion.fecha_fin,
        carpeta,
    )
    (carpeta / MANIFIESTO_DATOS).write_text(
        json.dumps({"clave": clave, "archivos": [r.archivo for r in resumenes]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return "descarga", resumenes

"""
Descarga ZIPs y CHECKSUMs desde Binance Vision.
Retorna None si el archivo no existe (404).
"""
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple

from . import config
from .utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Construcción de URLs
# ---------------------------------------------------------------------------

def _url_mensual(simbolo: str, tipo: str, year: int, month: int) -> Tuple[str, str]:
    nombre = f"{simbolo}-{config.INTERVALO}-{year:04d}-{month:02d}"
    base = (
        f"{config.BASE_URL}/futures/{config.MERCADO}/monthly"
        f"/{tipo}/{simbolo}/{config.INTERVALO}"
    )
    return f"{base}/{nombre}.zip", f"{base}/{nombre}.zip.CHECKSUM"


def _url_diario(simbolo: str, tipo: str, fecha) -> Tuple[str, str]:
    nombre = f"{simbolo}-{config.INTERVALO}-{fecha.year:04d}-{fecha.month:02d}-{fecha.day:02d}"
    base = (
        f"{config.BASE_URL}/futures/{config.MERCADO}/daily"
        f"/{tipo}/{simbolo}/{config.INTERVALO}"
    )
    return f"{base}/{nombre}.zip", f"{base}/{nombre}.zip.CHECKSUM"


# ---------------------------------------------------------------------------
# Descarga HTTP
# ---------------------------------------------------------------------------

def _descargar(url: str, dest: Path, silencioso: bool = False) -> Optional[bool]:
    """
    Descarga url -> dest con reintentos y progreso.
    Retorna True si OK, False si 404, None si error irrecuperable.
    """
    for intento in range(config.MAX_REINTENTOS):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "BinanceVisionDownloader/1.0"}
            )
            with urllib.request.urlopen(req, timeout=config.TIMEOUT_SEGUNDOS) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                descargado = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(config.CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        descargado += len(chunk)
                        if total and not silencioso:
                            pct = descargado / total * 100
                            print(f"\r    {dest.name}: {pct:.0f}%", end="", flush=True)
            if not silencioso:
                print()
            return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if intento < config.MAX_REINTENTOS - 1:
                log.warning(f"HTTP {e.code} para {dest.name}, reintento {intento + 1}…")
                time.sleep(2 ** intento)
                continue
            log.error(f"HTTP {e.code} irrecuperable para {url}")
            raise

        except (urllib.error.URLError, OSError) as e:
            if intento < config.MAX_REINTENTOS - 1:
                log.warning(f"Error de red ({e}) para {dest.name}, reintento {intento + 1}…")
                time.sleep(2 ** intento)
                continue
            log.error(f"Error irrecuperable descargando {url}: {e}")
            raise

    return None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def descargar_mensual(
    simbolo: str, tipo: str, year: int, month: int, directorio: Path
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Descarga ZIP mensual y su CHECKSUM.
    Retorna (zip_path, ck_path) o (None, None) si no disponible (404).
    Omite la descarga si los archivos ya existen.
    """
    url_zip, url_ck = _url_mensual(simbolo, tipo, year, month)
    nombre = f"{simbolo}-{config.INTERVALO}-{year:04d}-{month:02d}"
    zip_dest = directorio / f"{nombre}.zip"
    ck_dest  = directorio / f"{nombre}.zip.CHECKSUM"

    if zip_dest.exists() and ck_dest.exists():
        return zip_dest, ck_dest

    # CHECKSUM primero (pequeño, confirma que el período existe)
    if not _descargar(url_ck, ck_dest, silencioso=True):
        return None, None

    if not _descargar(url_zip, zip_dest):
        ck_dest.unlink(missing_ok=True)
        return None, None

    return zip_dest, ck_dest


def descargar_diario(
    simbolo: str, tipo: str, fecha, directorio: Path
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Descarga ZIP diario y su CHECKSUM.
    Retorna (zip_path, ck_path) o (None, None) si no disponible (404).
    Omite la descarga si los archivos ya existen.
    """
    url_zip, url_ck = _url_diario(simbolo, tipo, fecha)
    nombre = f"{simbolo}-{config.INTERVALO}-{fecha.year:04d}-{fecha.month:02d}-{fecha.day:02d}"
    zip_dest = directorio / f"{nombre}.zip"
    ck_dest  = directorio / f"{nombre}.zip.CHECKSUM"

    if zip_dest.exists() and ck_dest.exists():
        return zip_dest, ck_dest

    if not _descargar(url_ck, ck_dest, silencioso=True):
        return None, None

    if not _descargar(url_zip, zip_dest):
        ck_dest.unlink(missing_ok=True)
        return None, None

    return zip_dest, ck_dest

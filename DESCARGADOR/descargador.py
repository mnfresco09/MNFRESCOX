"""
Orquestador principal del DESCARGADOR.
Coordina descarga, verificación, parseo, combinación, relleno de gaps y exportación.
"""
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import polars as pl

from . import config
from .cliente import descargar_mensual, descargar_diario
from .verificador import verificar_sha256
from .parser import extraer_csv, parsear_csv
from .combinador import combinar
from .gaps import rellenar_y_validar
from .exportador import guardar
from .utils import get_logger, asegurar_dir

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rangos de fechas
# ---------------------------------------------------------------------------

def _siguiente_mes(d: date) -> date:
    return date(d.year + (d.month // 12), d.month % 12 + 1, 1)


def _rango_mensual(inicio: date, hoy: date):
    """Yields (year, month) de todos los meses cerrados desde inicio hasta el mes anterior a hoy."""
    mes_actual = hoy.replace(day=1)
    cur = inicio.replace(day=1)
    while cur < mes_actual:
        yield cur.year, cur.month
        cur = _siguiente_mes(cur)


def _rango_diario(hoy: date):
    """Yields fechas desde el primer día del mes actual hasta ayer (T-1)."""
    ayer = hoy - timedelta(days=1)
    cur  = hoy.replace(day=1)
    while cur <= ayer:
        yield cur
        cur += timedelta(days=1)


# ---------------------------------------------------------------------------
# Procesamiento de un (símbolo, tipo)
# ---------------------------------------------------------------------------

def _nombre_csv_mensual(simbolo: str, year: int, month: int) -> str:
    return f"{simbolo}-{config.INTERVALO}-{year:04d}-{month:02d}.csv"


def _nombre_csv_diario(simbolo: str, fecha: date) -> str:
    return f"{simbolo}-{config.INTERVALO}-{fecha.year:04d}-{fecha.month:02d}-{fecha.day:02d}.csv"


def _nombre_parquet(simbolo: str) -> str:
    if simbolo.endswith("USDT"):
        activo = f"{simbolo[:-4]}_USDT"
    else:
        activo = simbolo

    return f"{activo}_{config.INTERVALO}.parquet"


def _descargar_y_extraer_mensual(
    simbolo: str, tipo: str, year: int, month: int, dir_tipo: Path
) -> Optional[Path]:
    """
    Descarga, verifica SHA-256 y extrae el CSV mensual.
    Retorna la ruta del CSV extraído o None si no disponible.
    Reintenta la descarga hasta MAX_REINTENTOS veces si el SHA-256 falla.
    """
    for intento in range(config.MAX_REINTENTOS):
        zip_path, ck_path = descargar_mensual(simbolo, tipo, year, month, dir_tipo)

        if zip_path is None:
            return None  # 404: período no disponible

        if verificar_sha256(zip_path, ck_path):
            csv_path = extraer_csv(zip_path, dir_tipo)
            zip_path.unlink(missing_ok=True)
            ck_path.unlink(missing_ok=True)
            return csv_path

        log.warning(f"  SHA-256 incorrecto para {zip_path.name}, re-descargando ({intento + 1}/{config.MAX_REINTENTOS})")
        zip_path.unlink(missing_ok=True)
        ck_path.unlink(missing_ok=True)

    log.error(f"  SHA-256 fallido tras {config.MAX_REINTENTOS} intentos: {simbolo}/{tipo} {year}-{month:02d}")
    return None


def _descargar_y_extraer_diario(
    simbolo: str, tipo: str, fecha: date, dir_tipo: Path
) -> Optional[Path]:
    """Igual que la mensual pero para archivos diarios."""
    for intento in range(config.MAX_REINTENTOS):
        zip_path, ck_path = descargar_diario(simbolo, tipo, fecha, dir_tipo)

        if zip_path is None:
            return None

        if verificar_sha256(zip_path, ck_path):
            csv_path = extraer_csv(zip_path, dir_tipo)
            zip_path.unlink(missing_ok=True)
            ck_path.unlink(missing_ok=True)
            return csv_path

        log.warning(f"  SHA-256 incorrecto para {fecha}, re-descargando ({intento + 1}/{config.MAX_REINTENTOS})")
        zip_path.unlink(missing_ok=True)
        ck_path.unlink(missing_ok=True)

    log.error(f"  SHA-256 fallido tras {config.MAX_REINTENTOS} intentos: {simbolo}/{tipo} {fecha}")
    return None


def _procesar_tipo(simbolo: str, tipo: str, hoy: date) -> pl.DataFrame:
    """
    Descarga todos los archivos (mensual + diario) para un (símbolo, tipo),
    parsea los CSV y los concatena en un único DataFrame ordenado.
    """
    inicio = config.ACTIVOS[simbolo][tipo]
    dir_tipo = config.TMP / simbolo / tipo
    asegurar_dir(dir_tipo)

    dfs: List[pl.DataFrame] = []
    omitidos = 0

    # --- Archivos mensuales ---
    for year, month in _rango_mensual(inicio, hoy):
        csv_cache = dir_tipo / _nombre_csv_mensual(simbolo, year, month)

        if csv_cache.exists():
            dfs.append(parsear_csv(csv_cache, tipo))
            continue

        log.info(f"  [{tipo}] {year}-{month:02d} mensual…")
        csv_path = _descargar_y_extraer_mensual(simbolo, tipo, year, month, dir_tipo)

        if csv_path is None:
            log.warning(f"  [{tipo}] {year}-{month:02d} no disponible, omitido")
            omitidos += 1
            continue

        dfs.append(parsear_csv(csv_path, tipo))

    # --- Archivos diarios (mes en curso) ---
    for fecha in _rango_diario(hoy):
        csv_cache = dir_tipo / _nombre_csv_diario(simbolo, fecha)

        if csv_cache.exists():
            dfs.append(parsear_csv(csv_cache, tipo))
            continue

        csv_path = _descargar_y_extraer_diario(simbolo, tipo, fecha, dir_tipo)

        if csv_path is None:
            log.warning(f"  [{tipo}] {fecha} diario no disponible, omitido")
            omitidos += 1
            continue

        dfs.append(parsear_csv(csv_path, tipo))

    if omitidos:
        log.warning(f"  [{tipo}] {omitidos} períodos omitidos")

    dfs_validos = [d for d in dfs if not d.is_empty()]
    if not dfs_validos:
        raise RuntimeError(f"No se obtuvieron datos para {simbolo}/{tipo}")

    df = pl.concat(dfs_validos).sort("timestamp").unique(subset=["timestamp"], keep="first")
    log.info(f"  [{tipo}] {len(df):,} filas")
    return df


# ---------------------------------------------------------------------------
# Procesamiento de un activo completo
# ---------------------------------------------------------------------------

def _procesar_activo(simbolo: str, hoy: date) -> None:
    log.info(f"\n{'=' * 60}")
    log.info(f"  Activo: {simbolo}")
    log.info(f"{'=' * 60}")

    datos = {}
    for tipo in config.TIPOS:
        log.info(f"\n  Tipo: {tipo}")
        datos[tipo] = _procesar_tipo(simbolo, tipo, hoy)

    log.info("\n  Combinando tipos…")
    df = combinar(
        klines  = datos["klines"],
        premium = datos["premiumIndexKlines"],
    )

    log.info("  Rellenando gaps y validando…")
    df = rellenar_y_validar(df)

    destino = config.HISTORICO / _nombre_parquet(simbolo)
    log.info(f"  Exportando a {destino}…")
    guardar(df, destino)

    # Limpiar archivos temporales del activo
    tmp_activo = config.TMP / simbolo
    if tmp_activo.exists():
        shutil.rmtree(tmp_activo)
        log.info(f"  Temporales eliminados: {tmp_activo}")


# ---------------------------------------------------------------------------
# Punto de entrada público
# ---------------------------------------------------------------------------

def ejecutar(activos: Optional[List[str]] = None) -> None:
    """
    Descarga y procesa los activos indicados (o todos si activos=None).
    Lanza RuntimeError si algún activo falla.
    """
    hoy = date.today()
    activos_a_procesar = activos or list(config.ACTIVOS.keys())

    invalidos = [a for a in activos_a_procesar if a not in config.ACTIVOS]
    if invalidos:
        raise ValueError(
            f"Activos no reconocidos: {invalidos}\n"
            f"Válidos: {list(config.ACTIVOS.keys())}"
        )

    log.info("=" * 60)
    log.info("  DESCARGADOR — Binance Vision")
    log.info(f"  Fecha: {hoy}   Activos: {activos_a_procesar}")
    log.info(f"  Destino: {config.HISTORICO}")
    log.info("=" * 60)

    fallidos = []
    for simbolo in activos_a_procesar:
        try:
            _procesar_activo(simbolo, hoy)
        except Exception as e:
            log.error(f"\nError fatal en {simbolo}: {e}")
            fallidos.append(simbolo)

    if fallidos:
        raise RuntimeError(f"Activos con error: {fallidos}")

    log.info(f"\n{'=' * 60}")
    log.info(f"  Completado: {activos_a_procesar}")
    log.info("=" * 60)

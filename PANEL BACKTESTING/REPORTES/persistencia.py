"""Gestión de la carpeta de resultados de cada combinación.

Estructura de salida (una carpeta por combinación, sin subniveles ni históricos):

    RESULTADOS/
      <ESTRATEGIA>/
        <TIMEFRAME>/
          <SALIDA>/
            <ACTIVO>/
              resultados.xlsx           (Excel del mejor trial)
              grafico_operativa.html    (gráfico de precio + trades del mejor)
              analisis_optimizacion.html(dashboard interactivo de la optimización)
              informe_avanzado.html     (informe institucional: OOS, veredicto…)

Al relanzar la MISMA combinación (estrategia/timeframe/salida/activo) se borra
la carpeta previa y se crea limpia: siempre refleja el último run. La traza de
auditoría y reproducibilidad (huella, metadatos, todos los trials) vive en la
base de datos de experimentos (`REGISTRO_EXPERIMENTOS/`), no en ficheros sueltos.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

from MOTOR.wrapper import MOTIVOS


def ruta_base_combinacion(
    *,
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_nombre: str,
    exit_type: str,
) -> Path:
    """Ruta de la carpeta de la combinación SIN tocar disco (solo lectura).

    Útil para mostrarla en el monitor antes de optimizar. El borrado/creación se
    hace con `preparar_resultados_combinacion`, ya con la optimización terminada.
    """
    return _base_combinacion(
        carpeta_resultados=carpeta_resultados,
        activo=activo,
        timeframe=timeframe,
        estrategia_nombre=estrategia_nombre,
        exit_type=exit_type,
    )


def preparar_resultados_combinacion(
    *,
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_nombre: str,
    exit_type: str,
) -> Path:
    """Borra la carpeta previa de la combinación y la crea limpia. Devuelve la ruta.

    Es el único punto que decide la carpeta final donde los generadores escriben
    sus ficheros, de modo que relanzar una combinación reemplaza el resultado
    anterior por completo.
    """
    base = _base_combinacion(
        carpeta_resultados=carpeta_resultados,
        activo=activo,
        timeframe=timeframe,
        estrategia_nombre=estrategia_nombre,
        exit_type=exit_type,
    )
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _base_combinacion(
    *,
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_nombre: str,
    exit_type: str,
) -> Path:
    # Orden: ESTRATEGIA / TIMEFRAME / SALIDA / ACTIVO.
    return (
        carpeta_resultados
        / slug(estrategia_nombre).upper()
        / slug(timeframe).upper()
        / slug(exit_type).upper()
        / slug(activo).upper()
    )


def trades_dataframe(replay) -> pl.DataFrame:
    """DataFrame de trades desde las columnas numpy del replay (lo usa el Excel).

    Cero iteración Python: una asignación por columna.
    """
    if replay is None:
        raise ValueError("[REPORTES] No hay replay disponible para trades_dataframe.")
    t = replay.trades
    direccion = t["direccion"].astype(np.int8)
    pnl_neto = t["pnl"].astype(np.float64)
    comision = t["comision_total"].astype(np.float64)
    dur_seg = (t["ts_salida"] - t["ts_entrada"]).clip(min=0).astype(np.float64) / 1_000_000.0
    motivo = np.array(MOTIVOS, dtype=object)[t["motivo_salida"].astype(np.int64)]
    direccion_txt = np.where(direccion == 1, "LONG", "SHORT")

    return pl.DataFrame(
        {
            "idx_senal":      t["idx_senal"].astype(np.int64),
            "idx_entrada":    t["idx_entrada"].astype(np.int64),
            "idx_salida":     t["idx_salida"].astype(np.int64),
            "ts_senal":       t["ts_senal"].astype(np.int64),
            "ts_entrada":     t["ts_entrada"].astype(np.int64),
            "ts_salida":      t["ts_salida"].astype(np.int64),
            "direccion":      direccion,
            "direccion_txt":  direccion_txt,
            "precio_entrada": t["precio_entrada"].astype(np.float64),
            "precio_salida":  t["precio_salida"].astype(np.float64),
            "saldo_apertura": t["colateral"].astype(np.float64),
            "apalancamiento": t["apalancamiento"].astype(np.float64),
            "tamano_posicion": t["tamano_posicion"].astype(np.float64),
            "risk_vol_ewma":  t["risk_vol_ewma"].astype(np.float64),
            "risk_sl_dist_pct": t["risk_sl_dist_pct"].astype(np.float64),
            "comision_total": comision,
            "pnl_bruto":      pnl_neto + comision,
            "pnl":            pnl_neto,
            "roi":            t["roi"].astype(np.float64),
            "saldo_post":     t["saldo_post"].astype(np.float64),
            "motivo_salida":  motivo,
            "duracion_velas": t["duracion_velas"].astype(np.int64),
            "duracion_seg":   dur_seg,
        },
        schema={
            "idx_senal": pl.Int64,
            "idx_entrada": pl.Int64,
            "idx_salida": pl.Int64,
            "ts_senal": pl.Int64,
            "ts_entrada": pl.Int64,
            "ts_salida": pl.Int64,
            "direccion": pl.Int8,
            "direccion_txt": pl.String,
            "precio_entrada": pl.Float64,
            "precio_salida": pl.Float64,
            "saldo_apertura": pl.Float64,
            "apalancamiento": pl.Float64,
            "tamano_posicion": pl.Float64,
            "risk_vol_ewma": pl.Float64,
            "risk_sl_dist_pct": pl.Float64,
            "comision_total": pl.Float64,
            "pnl_bruto": pl.Float64,
            "pnl": pl.Float64,
            "roi": pl.Float64,
            "saldo_post": pl.Float64,
            "motivo_salida": pl.String,
            "duracion_velas": pl.Int64,
            "duracion_seg": pl.Float64,
        },
    )


def slug(valor: str) -> str:
    normalizado = unicodedata.normalize("NFKD", str(valor))
    ascii_text = normalizado.encode("ascii", "ignore").decode("ascii")
    slug_text = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    return slug_text or "sin_nombre"

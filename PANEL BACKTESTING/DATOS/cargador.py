import polars as pl
from pathlib import Path
from datetime import date, timedelta

from DATOS.resampleo import TIMEFRAMES_ORDENADOS


_LECTORES = {
    "feather":  lambda p: pl.read_ipc(p, memory_map=False),
    "parquet":  pl.read_parquet,
    "csv":      lambda p: pl.read_csv(p, try_parse_dates=True),
}


def cargar(activo: str, cfg) -> pl.DataFrame:
    """
    Localiza y carga el archivo de menor timeframe disponible para el activo dado.
    Devuelve un DataFrame con timestamp en UTC microsegundos y
    el rango de fechas ya filtrado según config.
    """
    ruta = _buscar_archivo(activo, cfg)
    lector = _LECTORES[cfg.FORMATO_DATOS]
    df = lector(ruta)
    df = _normalizar_timestamp(df)
    df = _filtrar_fechas(df, cfg.FECHA_INICIO, cfg.FECHA_FIN)
    return df


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _buscar_archivo(activo: str, cfg) -> Path:
    ext = {"feather": ".feather", "parquet": ".parquet", "csv": ".csv"}[cfg.FORMATO_DATOS]
    encontrados_por_tf = []
    for timeframe in TIMEFRAMES_ORDENADOS:
        patron = f"{activo}_*_{timeframe}{ext}"
        encontrados = sorted(cfg.CARPETA_HISTORICO.glob(patron))
        if encontrados:
            encontrados_por_tf.append((timeframe, patron, encontrados))

    if not encontrados_por_tf:
        raise FileNotFoundError(
            f"No se encontró ningún archivo para '{activo}' en timeframes soportados.\n"
            f"  Buscando en: {cfg.CARPETA_HISTORICO}\n"
            f"  Archivos presentes: {[f.name for f in cfg.CARPETA_HISTORICO.iterdir() if not f.name.startswith('.')]}"
        )

    _timeframe, patron, encontrados = encontrados_por_tf[0]
    if len(encontrados) > 1:
        raise ValueError(
            f"Se encontraron varios archivos para '{activo}' con patrón '{patron}':\n"
            + "\n".join(f"  - {f.name}" for f in encontrados)
            + "\nDeja solo uno en HISTORICO/."
        )

    return encontrados[0]


def _normalizar_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    """
    Garantiza que la columna 'timestamp' sea Datetime(us, UTC) en todos los casos,
    independientemente del formato original del archivo (ns, us, ms, sin tz, etc.).
    """
    col = df["timestamp"]
    dtype = col.dtype

    if dtype == pl.Utf8:
        df = df.with_columns(pl.col("timestamp").str.to_datetime(time_unit="us", time_zone="UTC"))
    elif isinstance(dtype, pl.Datetime):
        if dtype.time_unit != "us":
            df = df.with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
        if dtype.time_zone is None:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
        elif dtype.time_zone != "UTC":
            df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))

    return df


def _filtrar_fechas(df: pl.DataFrame, fecha_inicio: str, fecha_fin: str) -> pl.DataFrame:
    inicio = pl.lit(fecha_inicio).str.to_datetime(format="%Y-%m-%d", time_unit="us").dt.replace_time_zone("UTC")
    fin_exclusivo = (date.fromisoformat(fecha_fin) + timedelta(days=1)).isoformat()
    fin = pl.lit(fin_exclusivo).str.to_datetime(format="%Y-%m-%d", time_unit="us").dt.replace_time_zone("UTC")

    df = df.filter(
        (pl.col("timestamp") >= inicio) &
        (pl.col("timestamp") < fin)
    )

    if df.is_empty():
        raise ValueError(
            f"El filtro de fechas [{fecha_inicio} → {fecha_fin}] "
            f"no dejó ninguna fila. Revisa FECHA_INICIO y FECHA_FIN en config.py."
        )

    return df

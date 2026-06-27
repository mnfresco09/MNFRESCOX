import polars as pl
from pathlib import Path
from datetime import date, timedelta

from DATOS.tiempo import TIMEFRAMES_ORDENADOS


_LECTORES = {
    "feather":  lambda p: pl.read_ipc(p, memory_map=False),
    "parquet":  pl.read_parquet,
    "csv":      lambda p: pl.read_csv(p, try_parse_dates=True),
}

# Tramos del split temporal de tres bloques (Fase 0).
#   "train_val" → TRAIN/VALIDATION: [FECHA_INICIO, HOLDOUT_INICIO).
#   "holdout"   → HOLDOUT BLOQUEADO: [HOLDOUT_INICIO, FECHA_FIN].
#   "completo"  → todo el histórico: [FECHA_INICIO, FECHA_FIN].
#   "auto"      → lo decide cfg.MODO (investigacion→train_val, veredicto→completo).
TRAMOS_VALIDOS = ("auto", "train_val", "holdout", "completo")


def cargar(activo: str, cfg, *, tramo: str = "auto") -> pl.DataFrame:
    """
    Localiza y carga el archivo de menor timeframe disponible para el activo dado.
    Devuelve un DataFrame con timestamp en UTC microsegundos y el rango de fechas
    ya filtrado según el split temporal de la Fase 0.

    El parámetro `tramo` decide qué bloque del split se devuelve. Por defecto
    ("auto") respeta `cfg.MODO`: en modo "investigacion" el holdout bloqueado
    queda FÍSICAMENTE excluido del DataFrame, de modo que es imposible que entre
    en la optimización o la validación.
    """
    if tramo not in TRAMOS_VALIDOS:
        raise ValueError(f"tramo '{tramo}' no válido. Opciones: {TRAMOS_VALIDOS}")

    ruta = ruta_datos(activo, cfg)
    lector = _LECTORES[cfg.FORMATO_DATOS]
    df = lector(ruta)
    df = _normalizar_timestamp(df)

    inicio, fin_exclusivo = _rango_para_tramo(cfg, tramo)
    df = _filtrar_rango(df, inicio, fin_exclusivo, tramo=tramo)
    return df


def ruta_datos(activo: str, cfg) -> Path:
    """Ruta del fichero de datos que se usaría para `activo` (sin cargarlo).

    Expuesta para que la huella de reproducibilidad pueda hashear el fichero
    exacto que alimenta el run.
    """
    return _buscar_archivo(activo, cfg)


def limites_split(cfg) -> tuple[date, date, date]:
    """Fronteras del split temporal de tres bloques, como fechas.

    Devuelve `(inicio, holdout_inicio, fin_exclusivo)` donde:
      - [inicio, holdout_inicio)        = TRAIN/VALIDATION
      - [holdout_inicio, fin_exclusivo) = HOLDOUT BLOQUEADO

    `fin_exclusivo` es FECHA_FIN + 1 día (el filtro temporal es semiabierto por
    la derecha, coherente con el resto del sistema). Función pura, sin Polars,
    para poder verificarla de forma aislada.
    """
    inicio = date.fromisoformat(str(cfg.FECHA_INICIO))
    fin = date.fromisoformat(str(cfg.FECHA_FIN))
    holdout = date.fromisoformat(str(cfg.HOLDOUT_INICIO))
    if not (inicio < holdout <= fin):
        raise ValueError(
            f"HOLDOUT_INICIO ({holdout}) debe cumplir FECHA_INICIO ({inicio}) "
            f"< HOLDOUT_INICIO <= FECHA_FIN ({fin})."
        )
    fin_exclusivo = fin + timedelta(days=1)
    return inicio, holdout, fin_exclusivo


def _rango_para_tramo(cfg, tramo: str) -> tuple[date, date]:
    """Resuelve el tramo (incluido "auto" vía MODO) a un rango [inicio, fin_exclusivo)."""
    inicio, holdout, fin_exclusivo = limites_split(cfg)

    if tramo == "auto":
        modo = getattr(cfg, "MODO", "investigacion")
        tramo = "completo" if modo == "veredicto_final" else "train_val"

    if tramo == "train_val":
        return inicio, holdout
    if tramo == "holdout":
        return holdout, fin_exclusivo
    if tramo == "completo":
        return inicio, fin_exclusivo
    raise ValueError(f"tramo '{tramo}' no válido. Opciones: {TRAMOS_VALIDOS}")


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


def _filtrar_rango(
    df: pl.DataFrame, inicio: date, fin_exclusivo: date, *, tramo: str
) -> pl.DataFrame:
    """Filtra el DataFrame al rango semiabierto [inicio, fin_exclusivo)."""
    inicio_lit = (
        pl.lit(inicio.isoformat())
        .str.to_datetime(format="%Y-%m-%d", time_unit="us")
        .dt.replace_time_zone("UTC")
    )
    fin_lit = (
        pl.lit(fin_exclusivo.isoformat())
        .str.to_datetime(format="%Y-%m-%d", time_unit="us")
        .dt.replace_time_zone("UTC")
    )

    df = df.filter(
        (pl.col("timestamp") >= inicio_lit) &
        (pl.col("timestamp") < fin_lit)
    )

    if df.is_empty():
        raise ValueError(
            f"El filtro temporal del tramo '{tramo}' [{inicio} → {fin_exclusivo}) "
            f"no dejó ninguna fila. Revisa FECHA_INICIO, FECHA_FIN y HOLDOUT_INICIO "
            f"en config.py (y que el histórico cubra ese rango)."
        )

    return df

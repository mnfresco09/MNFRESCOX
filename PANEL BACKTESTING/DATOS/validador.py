import polars as pl
import sys

from DATOS.tiempo import MICROSEGUNDOS_POR_TIMEFRAME, serie_timestamp_us

COLUMNAS_MINIMAS = {"timestamp", "open", "high", "low", "close"}


def validar(
    df: pl.DataFrame,
    activo: str,
    estrategia_columnas: set[str] | None = None,
    timeframe: str | None = None,
    permitir_huecos: bool = False,
) -> None:
    """
    Comprueba la integridad del DataFrame antes de cualquier operación.
    Para si encuentra algún problema y explica exactamente qué falla.
    """
    errores = []

    # --- Columnas mínimas ---
    presentes = set(df.columns)
    faltantes = COLUMNAS_MINIMAS - presentes
    if faltantes:
        errores.append(f"Columnas mínimas faltantes: {sorted(faltantes)}")

    # --- Columnas extra que requiere la estrategia ---
    if estrategia_columnas:
        faltantes_extra = estrategia_columnas - presentes
        if faltantes_extra:
            errores.append(
                f"La estrategia necesita columnas que no están en los datos: {sorted(faltantes_extra)}"
            )

    if errores:
        _reportar(activo, errores)

    # --- Valores nulos en columnas de precio ---
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            continue
        nulos = df[col].null_count()
        if nulos > 0:
            errores.append(f"Columna '{col}' tiene {nulos:,} valores nulos.")

    # --- Orden cronológico ---
    timestamps = df["timestamp"]
    if not timestamps.is_sorted():
        errores.append("Las velas no están en orden cronológico ascendente.")

    # --- Duplicados ---
    duplicados = len(df) - df["timestamp"].n_unique()
    if duplicados > 0:
        errores.append(f"Hay {duplicados:,} timestamps duplicados.")

    # --- Huecos temporales ---
    if timeframe is not None and timeframe in MICROSEGUNDOS_POR_TIMEFRAME and df.height > 1:
        ts_us = serie_timestamp_us(df)
        delta_esperado = MICROSEGUNDOS_POR_TIMEFRAME[timeframe]
        deltas = ts_us.diff().drop_nulls()
        huecos = deltas.filter(deltas != delta_esperado).len()
        if huecos > 0:
            mensaje = (
                f"Hay {huecos:,} saltos temporales distintos de {timeframe}; "
                "los datos tienen huecos o velas irregulares."
            )
            if permitir_huecos:
                print(
                    f"[DATOS] {activo}: {mensaje} "
                    "Permitido por configuracion de mercado no 24/7."
                )
            else:
                errores.append(mensaje)

    # --- Precios coherentes (high >= low, precios > 0) ---
    if {"high", "low"}.issubset(presentes):
        invalidos = df.filter(pl.col("high") < pl.col("low")).height
        if invalidos > 0:
            errores.append(f"Hay {invalidos:,} velas donde high < low.")

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            continue
        negativos = df.filter(pl.col(col) <= 0).height
        if negativos > 0:
            errores.append(f"Columna '{col}' tiene {negativos:,} valores <= 0.")

    if errores:
        _reportar(activo, errores)


def _reportar(activo: str, errores: list[str]) -> None:
    print(f"\n[DATOS] Errores de validación en '{activo}':\n")
    for e in errores:
        print(f"  ✗ {e}")
    print()
    sys.exit(1)

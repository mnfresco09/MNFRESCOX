import polars as pl
import sys

COLUMNAS_MINIMAS = {"timestamp", "open", "high", "low", "close"}


def validar(df: pl.DataFrame, activo: str, estrategia_columnas: set[str] | None = None) -> None:
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

from pathlib import Path
import sys

TIMEFRAMES_VALIDOS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
FORMATOS_VALIDOS   = {"feather", "parquet", "csv"}
EXIT_TYPES_VALIDOS = {"FIXED", "BARS", "ALL"}
SAMPLERS_VALIDOS   = {"QMC", "TPE", "HYBRID"}
EXTENSIONES        = {"feather": ".feather", "parquet": ".parquet", "csv": ".csv"}


def validar(cfg) -> None:
    errores = []

    # --- Activos ---
    activos = cfg.ACTIVOS if isinstance(cfg.ACTIVOS, list) else [cfg.ACTIVOS]
    if not activos:
        errores.append("ACTIVOS está vacío.")

    # --- Formato ---
    if cfg.FORMATO_DATOS not in FORMATOS_VALIDOS:
        errores.append(f"FORMATO_DATOS '{cfg.FORMATO_DATOS}' no válido. Opciones: {FORMATOS_VALIDOS}")

    # --- Archivos en HISTORICO ---
    ext = EXTENSIONES[cfg.FORMATO_DATOS]
    for activo in activos:
        patron = f"{activo}_*_1m{ext}"
        encontrados = list(cfg.CARPETA_HISTORICO.glob(patron))
        if not encontrados:
            errores.append(
                f"No se encontró archivo para '{activo}' con patrón '{patron}' "
                f"en {cfg.CARPETA_HISTORICO}"
            )

    # --- Timeframes ---
    tfs = cfg.TIMEFRAMES if isinstance(cfg.TIMEFRAMES, list) else [cfg.TIMEFRAMES]
    for tf in tfs:
        if tf not in TIMEFRAMES_VALIDOS:
            errores.append(f"Timeframe '{tf}' no válido. Opciones: {TIMEFRAMES_VALIDOS}")

    # --- Fechas ---
    try:
        from datetime import date
        inicio = date.fromisoformat(cfg.FECHA_INICIO)
        fin    = date.fromisoformat(cfg.FECHA_FIN)
        if inicio >= fin:
            errores.append(f"FECHA_INICIO ({cfg.FECHA_INICIO}) debe ser anterior a FECHA_FIN ({cfg.FECHA_FIN}).")
    except ValueError as e:
        errores.append(f"Formato de fecha incorrecto: {e}. Usa 'AAAA-MM-DD'.")

    # --- Capital ---
    if cfg.SALDO_INICIAL <= 0:
        errores.append("SALDO_INICIAL debe ser mayor que 0.")
    if cfg.SALDO_USADO_POR_TRADE <= 0:
        errores.append("SALDO_USADO_POR_TRADE debe ser mayor que 0.")
    if cfg.SALDO_USADO_POR_TRADE > cfg.SALDO_INICIAL:
        errores.append("SALDO_USADO_POR_TRADE no puede ser mayor que SALDO_INICIAL.")
    if cfg.APALANCAMIENTO < 1:
        errores.append("APALANCAMIENTO debe ser >= 1.")
    if cfg.SALDO_MINIMO_OPERATIVO < 0:
        errores.append("SALDO_MINIMO_OPERATIVO debe ser >= 0.")
    if not (0 < cfg.COMISION_PCT < 1):
        errores.append("COMISION_PCT debe estar entre 0 y 1 (ej: 0.0005 para 0.05%).")
    if cfg.COMISION_LADOS not in (1, 2):
        errores.append("COMISION_LADOS debe ser 1 (apertura) o 2 (apertura y cierre).")

    # --- Salidas ---
    if cfg.EXIT_TYPE not in EXIT_TYPES_VALIDOS:
        errores.append(f"EXIT_TYPE '{cfg.EXIT_TYPE}' no válido. Opciones: {EXIT_TYPES_VALIDOS}")

    # Los parámetros específicos de cada tipo (SL, TP, velas) se validan
    # en sus propios módulos: SALIDAS/fijo.py y SALIDAS/velas.py

    # --- Optuna ---
    if cfg.OPTUNA_SAMPLER not in SAMPLERS_VALIDOS:
        errores.append(f"OPTUNA_SAMPLER '{cfg.OPTUNA_SAMPLER}' no válido. Opciones: {SAMPLERS_VALIDOS}")
    if cfg.N_TRIALS < 1:
        errores.append("N_TRIALS debe ser >= 1.")

    # --- Resultados ---
    if cfg.MAX_PLOTS < 0:
        errores.append("MAX_PLOTS debe ser >= 0.")
    if cfg.MAX_ARCHIVOS < 1:
        errores.append("MAX_ARCHIVOS debe ser >= 1.")

    # --- Reporte final ---
    if errores:
        print("\n[CONFIG] Se encontraron errores antes de arrancar:\n")
        for e in errores:
            print(f"  ✗ {e}")
        print()
        sys.exit(1)

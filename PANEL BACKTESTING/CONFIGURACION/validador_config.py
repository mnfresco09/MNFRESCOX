import importlib
import sys

TIMEFRAMES_VALIDOS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
FORMATOS_VALIDOS   = {"feather", "parquet", "csv"}
EXIT_TYPES_VALIDOS = {"FIXED", "BARS", "CUSTOM", "ALL"}
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

    mercado_24_7 = getattr(cfg, "MERCADO_24_7", {})
    if not isinstance(mercado_24_7, dict):
        errores.append("MERCADO_24_7 debe ser un dict, por ejemplo {'BTC': True, 'GOLD': False}.")
    else:
        for activo in activos:
            if activo not in mercado_24_7:
                errores.append(f"MERCADO_24_7 no define si '{activo}' es 24/7.")
            elif not isinstance(mercado_24_7[activo], bool):
                errores.append(f"MERCADO_24_7['{activo}'] debe ser True o False.")

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
        if inicio > fin:
            errores.append(f"FECHA_INICIO ({cfg.FECHA_INICIO}) no puede ser posterior a FECHA_FIN ({cfg.FECHA_FIN}).")
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
    else:
        _validar_modulos_salida(cfg.EXIT_TYPE, errores)

    # --- Optuna ---
    if cfg.OPTUNA_SAMPLER not in SAMPLERS_VALIDOS:
        errores.append(f"OPTUNA_SAMPLER '{cfg.OPTUNA_SAMPLER}' no válido. Opciones: {SAMPLERS_VALIDOS}")
    if cfg.N_TRIALS < 1:
        errores.append("N_TRIALS debe ser >= 1.")
    if cfg.N_JOBS == 0:
        errores.append("N_JOBS no puede ser 0. Usa 1, -1 o -2.")

    _validar_perturbaciones(cfg, errores)

    # --- Resultados ---
    if not isinstance(cfg.USAR_EXCEL, bool):
        errores.append("USAR_EXCEL debe ser True o False.")
    if cfg.MAX_PLOTS < 0:
        errores.append("MAX_PLOTS debe ser >= 0.")
    if cfg.MAX_ARCHIVOS < 1:
        errores.append("MAX_ARCHIVOS debe ser >= 1.")
    rango = str(cfg.GRAFICA_RANGO).lower()
    if rango != "all" and rango != "custom" and not (rango.endswith("m") and rango[:-1].isdigit()):
        errores.append("GRAFICA_RANGO debe ser 'all', 'custom' o un texto como '3m'.")

    # --- Reporte final ---
    if errores:
        print("\n[CONFIG] Se encontraron errores antes de arrancar:\n")
        for e in errores:
            print(f"  ✗ {e}")
        print()
        sys.exit(1)


def _validar_modulos_salida(exit_type: str, errores: list[str]) -> None:
    if exit_type in {"FIXED", "ALL"}:
        fijo = _importar_salida("fijo", errores)
        if fijo is not None:
            _validar_mayor_cero(fijo, "EXIT_SL_PCT", errores)
            _validar_mayor_cero(fijo, "EXIT_TP_PCT", errores)
            if bool(getattr(fijo, "OPTIMIZAR_SALIDAS", False)):
                _validar_rango(fijo, "EXIT_SL_MIN", "EXIT_SL_MAX", errores)
                _validar_rango(fijo, "EXIT_TP_MIN", "EXIT_TP_MAX", errores)

    if exit_type in {"BARS", "ALL"}:
        velas = _importar_salida("velas", errores)
        if velas is not None:
            _validar_mayor_cero(velas, "EXIT_SL_PCT", errores)
            _validar_entero_mayor_cero(velas, "EXIT_VELAS", errores)
            if bool(getattr(velas, "OPTIMIZAR_SALIDAS", False)):
                _validar_rango(velas, "EXIT_SL_MIN", "EXIT_SL_MAX", errores)
                _validar_rango(velas, "EXIT_VELAS_MIN", "EXIT_VELAS_MAX", errores)

    if exit_type in {"CUSTOM", "ALL"}:
        personalizada = _importar_salida("personalizada", errores)
        if personalizada is not None:
            _validar_mayor_cero(personalizada, "EXIT_SL_PCT", errores)
            if bool(getattr(personalizada, "OPTIMIZAR_SALIDAS", False)):
                _validar_rango(personalizada, "EXIT_SL_MIN", "EXIT_SL_MAX", errores)


def _validar_perturbaciones(cfg, errores: list[str]) -> None:
    activa = bool(getattr(cfg, "PERTURBACIONES_ACTIVAS", False))
    seed = getattr(cfg, "PERTURBACIONES_SEED", None)
    if seed is not None and not isinstance(seed, int):
        errores.append("PERTURBACIONES_SEED debe ser int o None.")

    if not activa:
        return

    _cfg_float_rango(cfg, "BANDA_MAX_PRECIO", errores, minimo=0.0, maximo=0.90, cerrado_min=False)
    _cfg_float_rango(cfg, "FUERZA_AMORTIGUACION", errores, minimo=0.0, maximo=1.0)
    _cfg_float_rango(cfg, "ESCALA_VOLATILIDAD", errores, minimo=0.0, maximo=None)
    _cfg_int_min(cfg, "VENTANA_VOLATILIDAD", errores, minimo=2)
    _cfg_float_rango(cfg, "SIGMA_RANGO_VELA", errores, minimo=0.0, maximo=None)
    _cfg_float_rango(cfg, "RUIDO_POSICION_OHLC", errores, minimo=0.0, maximo=0.49)
    _cfg_float_rango(cfg, "SIGMA_VOLUMEN", errores, minimo=0.0, maximo=None)
    _cfg_float_rango(cfg, "GRANULARIDAD_CUBOS", errores, minimo=0.0, maximo=None, cerrado_min=False)
    _cfg_float_rango(cfg, "INERCIA_ORDER_FLOW", errores, minimo=0.0, maximo=1.0)
    _cfg_int_min(cfg, "VENTANA_MEDIA_VOLUMEN", errores, minimo=2)


def _importar_salida(nombre: str, errores: list[str]):
    try:
        return importlib.import_module(f"SALIDAS.{nombre}")
    except Exception as exc:
        errores.append(f"No se pudo importar SALIDAS/{nombre}.py: {exc}")
        return None


def _cfg_float_rango(
    cfg,
    nombre: str,
    errores: list[str],
    *,
    minimo: float | None,
    maximo: float | None,
    cerrado_min: bool = True,
) -> None:
    try:
        valor = float(getattr(cfg, nombre))
    except Exception:
        errores.append(f"{nombre} debe existir y ser numérico cuando PERTURBACIONES_ACTIVAS=True.")
        return

    if minimo is not None:
        invalido_min = valor < minimo if cerrado_min else valor <= minimo
        if invalido_min:
            op = ">=" if cerrado_min else ">"
            errores.append(f"{nombre} debe ser {op} {minimo}.")
    if maximo is not None and valor > maximo:
        errores.append(f"{nombre} debe ser <= {maximo}.")


def _cfg_int_min(cfg, nombre: str, errores: list[str], *, minimo: int) -> None:
    try:
        valor = int(getattr(cfg, nombre))
    except Exception:
        errores.append(f"{nombre} debe existir y ser entero cuando PERTURBACIONES_ACTIVAS=True.")
        return
    if valor < minimo:
        errores.append(f"{nombre} debe ser >= {minimo}.")


def _validar_mayor_cero(modulo, atributo: str, errores: list[str]) -> None:
    try:
        valor = float(getattr(modulo, atributo))
    except Exception:
        errores.append(f"{modulo.__name__}.{atributo} debe existir y ser numérico.")
        return
    if valor <= 0:
        errores.append(f"{modulo.__name__}.{atributo} debe ser mayor que 0.")


def _validar_entero_mayor_cero(modulo, atributo: str, errores: list[str]) -> None:
    try:
        valor = int(getattr(modulo, atributo))
    except Exception:
        errores.append(f"{modulo.__name__}.{atributo} debe existir y ser entero.")
        return
    if valor <= 0:
        errores.append(f"{modulo.__name__}.{atributo} debe ser mayor que 0.")


def _validar_rango(modulo, minimo: str, maximo: str, errores: list[str]) -> None:
    try:
        valor_min = float(getattr(modulo, minimo))
        valor_max = float(getattr(modulo, maximo))
    except Exception:
        errores.append(f"{modulo.__name__}.{minimo}/{maximo} deben existir y ser numéricos.")
        return
    if valor_min <= 0 or valor_max <= 0:
        errores.append(f"{modulo.__name__}.{minimo}/{maximo} deben ser mayores que 0.")
    if valor_min > valor_max:
        errores.append(f"{modulo.__name__}.{minimo} no puede ser mayor que {maximo}.")

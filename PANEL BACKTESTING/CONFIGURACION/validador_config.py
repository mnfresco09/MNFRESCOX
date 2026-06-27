import importlib
import sys

from DATOS.tiempo import TIMEFRAMES_ORDENADOS

TIMEFRAMES_VALIDOS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
FORMATOS_VALIDOS   = {"feather", "parquet", "csv"}
EXIT_TYPES_VALIDOS = {"FIXED", "BARS", "TRAILING", "CUSTOM", "ALL"}
SAMPLERS_VALIDOS   = {"QMC", "TPE", "HYBRID"}
SCORES_VALIDOS     = {"PSR", "SHARPE", "CALMAR", "ROI"}
MODOS_VALIDOS      = {"investigacion", "veredicto_final"}
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
    # Igual que el cargador: vale cualquier timeframe soportado como base, no
    # solo 1m. Basta con encontrar al menos un archivo {activo}_*_{tf}{ext}.
    ext = EXTENSIONES[cfg.FORMATO_DATOS]
    for activo in activos:
        patrones = [f"{activo}_*_{tf}{ext}" for tf in TIMEFRAMES_ORDENADOS]
        encontrados = [
            ruta for patron in patrones for ruta in cfg.CARPETA_HISTORICO.glob(patron)
        ]
        if not encontrados:
            errores.append(
                f"No se encontró archivo para '{activo}' con patrón "
                f"'{activo}_*_<tf>{ext}' (tf en {list(TIMEFRAMES_ORDENADOS)}) "
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
    inicio = fin = None
    try:
        from datetime import date
        inicio = date.fromisoformat(cfg.FECHA_INICIO)
        fin    = date.fromisoformat(cfg.FECHA_FIN)
        if inicio > fin:
            errores.append(f"FECHA_INICIO ({cfg.FECHA_INICIO}) no puede ser posterior a FECHA_FIN ({cfg.FECHA_FIN}).")
    except ValueError as e:
        errores.append(f"Formato de fecha incorrecto: {e}. Usa 'AAAA-MM-DD'.")

    # --- Modo y holdout bloqueado (Fase 0) ---
    _validar_modo_y_holdout(cfg, inicio, fin, errores)

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
    if not (0 <= cfg.COMISION_PCT < 1):
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
    funcion_score = str(getattr(cfg, "FUNCION_SCORE", "PSR")).upper()
    if funcion_score not in SCORES_VALIDOS:
        errores.append(f"FUNCION_SCORE '{getattr(cfg, 'FUNCION_SCORE', None)}' no válido. Opciones: {SCORES_VALIDOS}")
    min_trades_score = getattr(cfg, "MIN_TRADES_SCORE", 0)
    if not isinstance(min_trades_score, int) or min_trades_score < 0:
        errores.append("MIN_TRADES_SCORE debe ser un entero >= 0.")
    if int(getattr(cfg, "TURNOVER_OBJETIVO", 100)) < 1:
        errores.append("TURNOVER_OBJETIVO debe ser un entero >= 1.")
    for nombre in ("PENALIZACION_TURNOVER_FACTOR", "PENALIZACION_COMPLEJIDAD_FACTOR"):
        valor = getattr(cfg, nombre, 0.0)
        if not isinstance(valor, (int, float)) or float(valor) < 0.0:
            errores.append(f"{nombre} debe ser un número >= 0.")
    if not isinstance(getattr(cfg, "OPTUNA_MULTIOBJETIVO", False), bool):
        errores.append("OPTUNA_MULTIOBJETIVO debe ser True o False.")
    if not isinstance(getattr(cfg, "MESETA_VECINOS", 7), int) or int(getattr(cfg, "MESETA_VECINOS", 7)) < 1:
        errores.append("MESETA_VECINOS debe ser un entero >= 1.")
    if cfg.N_JOBS == 0:
        errores.append("N_JOBS no puede ser 0. Usa 1, -1 o -2.")
    max_jobs_pert = getattr(cfg, "PERTURBACIONES_MAX_JOBS", 4)
    if not isinstance(max_jobs_pert, int) or max_jobs_pert < 1:
        errores.append("PERTURBACIONES_MAX_JOBS debe ser un entero >= 1.")
    validar_pert = getattr(cfg, "VALIDAR_PERTURBACIONES", False)
    if not isinstance(validar_pert, bool):
        errores.append("VALIDAR_PERTURBACIONES debe ser True o False.")
    _validar_semillas(cfg, errores)

    _validar_perturbaciones(cfg, errores)
    _validar_validacion(cfg, errores)

    # --- Resultados ---
    if not isinstance(cfg.USAR_EXCEL, bool):
        errores.append("USAR_EXCEL debe ser True o False.")
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
            usar_sl = getattr(velas, "USAR_SL_EMERGENCIA", True)
            if not isinstance(usar_sl, bool):
                errores.append("SALIDAS.velas.USAR_SL_EMERGENCIA debe ser True o False.")
                usar_sl = True
            if usar_sl:
                _validar_mayor_cero(velas, "EXIT_SL_PCT", errores)
            _validar_entero_mayor_cero(velas, "EXIT_VELAS", errores)
            if bool(getattr(velas, "OPTIMIZAR_SALIDAS", False)):
                if usar_sl:
                    _validar_rango(velas, "EXIT_SL_MIN", "EXIT_SL_MAX", errores)
                _validar_rango(velas, "EXIT_VELAS_MIN", "EXIT_VELAS_MAX", errores)

    if exit_type in {"TRAILING", "ALL"}:
        trailing = _importar_salida("trailing", errores)
        if trailing is not None:
            _validar_mayor_cero(trailing, "EXIT_SL_PCT", errores)
            _validar_mayor_cero(trailing, "EXIT_TRAIL_ACT_PCT", errores)
            _validar_mayor_cero(trailing, "EXIT_TRAIL_DIST_PCT", errores)
            if bool(getattr(trailing, "OPTIMIZAR_SALIDAS", False)):
                _validar_rango(trailing, "EXIT_SL_MIN", "EXIT_SL_MAX", errores)
                _validar_rango(trailing, "EXIT_TRAIL_ACT_MIN", "EXIT_TRAIL_ACT_MAX", errores)
                _validar_rango(trailing, "EXIT_TRAIL_DIST_MIN", "EXIT_TRAIL_DIST_MAX", errores)

    if exit_type in {"CUSTOM", "ALL"}:
        personalizada = _importar_salida("personalizada", errores)
        if personalizada is not None:
            _validar_mayor_cero(personalizada, "EXIT_SL_PCT", errores)
            if bool(getattr(personalizada, "OPTIMIZAR_SALIDAS", False)):
                _validar_rango(personalizada, "EXIT_SL_MIN", "EXIT_SL_MAX", errores)


def _validar_perturbaciones(cfg, errores: list[str]) -> None:
    activa = bool(getattr(cfg, "PERTURBACIONES_ACTIVAS", False))
    usar_seed = bool(getattr(cfg, "USAR_SEED", True))
    seed = getattr(cfg, "PERTURBACIONES_SEED", None)
    if seed is not None and not isinstance(seed, int):
        errores.append("PERTURBACIONES_SEED debe ser int o None.")
    if activa and usar_seed and seed is None:
        errores.append("PERTURBACIONES_SEED debe ser int cuando USAR_SEED=True y PERTURBACIONES_ACTIVAS=True.")

    if not activa:
        return

    _cfg_float_rango(cfg, "GRANULARIDAD_CUBOS", errores, minimo=0.0, maximo=None, cerrado_min=False)
    _cfg_float_rango(cfg, "PERCENTIL_TABLA", errores, minimo=0.0, maximo=0.49, cerrado_min=False)
    _validar_kernel_perturbaciones(errores)


def _validar_modo_y_holdout(cfg, inicio, fin, errores: list[str]) -> None:
    """Valida MODO y HOLDOUT_INICIO (split de tres bloques de la Fase 0).

    El holdout debe caer estrictamente dentro de (FECHA_INICIO, FECHA_FIN] para
    que exista un bloque de TRAIN/VALIDATION no vacío antes de él y un holdout
    no vacío hasta el final.
    """
    from datetime import date

    modo = getattr(cfg, "MODO", None)
    if modo not in MODOS_VALIDOS:
        errores.append(f"MODO '{modo}' no válido. Opciones: {MODOS_VALIDOS}")

    holdout_str = getattr(cfg, "HOLDOUT_INICIO", None)
    if holdout_str is None:
        errores.append("HOLDOUT_INICIO debe existir (AAAA-MM-DD) para el split de la Fase 0.")
        return
    try:
        holdout = date.fromisoformat(str(holdout_str))
    except ValueError as e:
        errores.append(f"HOLDOUT_INICIO con formato incorrecto: {e}. Usa 'AAAA-MM-DD'.")
        return

    # Solo se pueden comprobar los límites si las fechas base parsearon bien.
    if inicio is None or fin is None:
        return
    if holdout <= inicio:
        errores.append(
            f"HOLDOUT_INICIO ({holdout_str}) debe ser POSTERIOR a FECHA_INICIO "
            f"({cfg.FECHA_INICIO}); si no, no queda bloque de TRAIN/VALIDATION."
        )
    if holdout > fin:
        errores.append(
            f"HOLDOUT_INICIO ({holdout_str}) no puede ser posterior a FECHA_FIN "
            f"({cfg.FECHA_FIN}); el holdout quedaría vacío."
        )


def _validar_validacion(cfg, errores: list[str]) -> None:
    """Valida los parámetros de la validación OOS (Fases 2-7)."""
    if not isinstance(getattr(cfg, "VALIDACION_ACTIVA", False), bool):
        errores.append("VALIDACION_ACTIVA debe ser True o False.")
    if not getattr(cfg, "VALIDACION_ACTIVA", False):
        return

    n_grupos = getattr(cfg, "VALIDACION_N_GRUPOS", 6)
    k = getattr(cfg, "VALIDACION_K", 2)
    if not isinstance(n_grupos, int) or n_grupos < 2:
        errores.append("VALIDACION_N_GRUPOS debe ser un entero >= 2.")
    if not isinstance(k, int) or k < 1:
        errores.append("VALIDACION_K debe ser un entero >= 1.")
    elif isinstance(n_grupos, int) and k >= n_grupos:
        errores.append(f"VALIDACION_K ({k}) debe ser < VALIDACION_N_GRUPOS ({n_grupos}).")

    n_trials = getattr(cfg, "VALIDACION_N_TRIALS", 100)
    if not isinstance(n_trials, int) or n_trials < 1:
        errores.append("VALIDACION_N_TRIALS debe ser un entero >= 1.")

    embargo = getattr(cfg, "VALIDACION_EMBARGO", 0.01)
    if not isinstance(embargo, (int, float)) or not (0.0 <= float(embargo) < 1.0):
        errores.append("VALIDACION_EMBARGO debe estar en [0, 1).")

    dur = getattr(cfg, "VALIDACION_DURACION_TRADE", 1)
    if not isinstance(dur, int) or dur < 1:
        errores.append("VALIDACION_DURACION_TRADE debe ser un entero >= 1.")

    if not isinstance(getattr(cfg, "VALIDACION_WFA_ACTIVA", True), bool):
        errores.append("VALIDACION_WFA_ACTIVA debe ser True o False.")
    if not isinstance(getattr(cfg, "VALIDACION_WFA_ANCHORED", False), bool):
        errores.append("VALIDACION_WFA_ANCHORED debe ser True o False.")
    ventanas = getattr(cfg, "VALIDACION_WFA_VENTANAS", 5)
    if not isinstance(ventanas, int) or ventanas < 1:
        errores.append("VALIDACION_WFA_VENTANAS debe ser un entero >= 1.")
    fraccion = getattr(cfg, "VALIDACION_WFA_FRACCION", 0.15)
    if not isinstance(fraccion, (int, float)) or not (0.0 < float(fraccion) < 1.0):
        errores.append("VALIDACION_WFA_FRACCION debe estar en (0, 1).")
    elif isinstance(ventanas, int) and ventanas >= 1 and (ventanas + 1) * float(fraccion) > 1.0:
        errores.append(
            f"(VALIDACION_WFA_VENTANAS+1)·VALIDACION_WFA_FRACCION = "
            f"{(ventanas + 1) * float(fraccion):.2f} > 1: no queda train suficiente. "
            f"Reduce VALIDACION_WFA_VENTANAS o VALIDACION_WFA_FRACCION."
        )

    boot_iter = getattr(cfg, "VALIDACION_BOOTSTRAP_ITER", 10_000)
    if not isinstance(boot_iter, int) or boot_iter < 1:
        errores.append("VALIDACION_BOOTSTRAP_ITER debe ser un entero >= 1.")
    boot_bloque = getattr(cfg, "VALIDACION_BOOTSTRAP_BLOQUE", 1)
    if not isinstance(boot_bloque, int) or boot_bloque < 1:
        errores.append("VALIDACION_BOOTSTRAP_BLOQUE debe ser un entero >= 1.")
    sr_obj = getattr(cfg, "VALIDACION_SHARPE_ANUAL_OBJETIVO", 1.0)
    if not isinstance(sr_obj, (int, float)) or float(sr_obj) <= 0.0:
        errores.append("VALIDACION_SHARPE_ANUAL_OBJETIVO debe ser > 0.")
    nula_iter = getattr(cfg, "VALIDACION_NULA_ITER", 50)
    if not isinstance(nula_iter, int) or nula_iter < 0:
        errores.append("VALIDACION_NULA_ITER debe ser un entero >= 0.")


def _validar_semillas(cfg, errores: list[str]) -> None:
    usar_seed = getattr(cfg, "USAR_SEED", True)
    if not isinstance(usar_seed, bool):
        errores.append("USAR_SEED debe ser True o False.")

    optuna_seed = getattr(cfg, "OPTUNA_SEED", None)
    if optuna_seed is not None and not isinstance(optuna_seed, int):
        errores.append("OPTUNA_SEED debe ser int o None.")
    if usar_seed and optuna_seed is None:
        errores.append("OPTUNA_SEED debe ser int cuando USAR_SEED=True.")


def _validar_kernel_perturbaciones(errores: list[str]) -> None:
    try:
        from DATOS.perturbaciones import validar_kernel_numba

        validar_kernel_numba()
    except Exception as exc:
        errores.append(str(exc))


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

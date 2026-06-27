from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from datetime import date
from threading import Lock
from typing import Any, Optional

import numpy as np
import optuna
import polars as pl

try:
    from numba import njit
except ImportError:  # pragma: no cover - perturbaciones ya validan numba en runtime
    njit = None

from CONFIGURACION import config as cfg
from CONFIGURACION.validador_config import validar as validar_config
from COMUN import reproducibilidad
from COMUN.registro_experimentos import RegistroExperimentos
from DATOS.cargador import cargar, ruta_datos
from DATOS.perturbaciones import (
    ConfiguracionPerturbaciones,
    aplicar_perturbaciones,
    seed_para_trial,
)
from DATOS.resampleo import regla_agregacion, resamplear
from DATOS.tiempo import inferir_timeframe
from DATOS.validador import validar as validar_datos
from MOTOR import simular_full, simular_metricas
from NUCLEO import integridad
from NUCLEO.base_estrategia import CacheIndicadores
from NUCLEO.contexto import (
    ArraysMotor,
    ContextoCombinacion,
    SimConfigMotor,
    construir_arrays_motor,
    crear_contexto,
)
from NUCLEO.proyeccion import proyectar_senales_a_base
from NUCLEO.registro import cargar_estrategias, obtener_estrategia
from OPTIMIZACION.metricas import calcular_metricas
from OPTIMIZACION.puntuacion import calcular_score
from OPTIMIZACION.robustez_objetivo import (
    DIRECCIONES_PARETO,
    penalizacion_complejidad,
    penalizacion_turnover,
    seleccionar_meseta,
    vector_pareto,
)
from OPTIMIZACION.samplers import crear_sampler
from ROBUSTEZ import nula as robustez_nula
from ROBUSTEZ import regimen as robustez_regimen
from REPORTES.excel import generar_excel
from REPORTES.html import generar_htmls
from REPORTES.informe import generar_informe
from REPORTES.persistencia import (
    preparar_resultados_combinacion,
    ruta_base_combinacion,
)
from REPORTES.informe_institucional import generar_informe_institucional
from VALIDACION.integracion import ejecutar_validacion_completa
from VALIDACION.metricas_subconjunto import metricas_en_indices, retornos_por_trade
from REPORTES.rich import (
    MonitorOptimizacion,
    mostrar_aviso_perturbaciones,
    mostrar_fin_backtest,
    mostrar_huella_datos,
    mostrar_inicio_motor,
    mostrar_resumen_run,
)


@dataclass(frozen=True)
class ExitConfig:
    tipo: str
    sl_pct: float
    tp_pct: float
    velas: int
    sl_emergencia: bool = True
    trail_act_pct: float = 0.0
    trail_dist_pct: float = 0.0
    optimizar: bool = False
    sl_min: float | None = None
    sl_max: float | None = None
    tp_min: float | None = None
    tp_max: float | None = None
    velas_min: int | None = None
    velas_max: int | None = None
    trail_act_min: float | None = None
    trail_act_max: float | None = None
    trail_dist_min: float | None = None
    trail_dist_max: float | None = None


@dataclass
class ReplayTrial:
    """Datos del replay de un trial top: trades en columnas numpy + equity."""
    metricas_obj: object  # struct Metricas Rust (para integridad)
    trades: dict[str, np.ndarray]
    equity_curve: np.ndarray
    df_tf: Any | None = None
    indicadores: list[dict] | None = None
    perturbacion_seed: int | None = None


@dataclass
class TrialResultado:
    numero: int
    activo: str
    timeframe: str
    timeframe_ejecucion: str
    estrategia_id: int
    estrategia_nombre: str
    salida: ExitConfig
    parametros: dict
    score: float
    metricas: dict
    conteo_senales: dict[int, int]
    conteo_salidas: dict[int, int] | None = None
    perturbacion_seed: int | None = None
    replay: Optional[ReplayTrial] = None
    en_pareto: bool = False  # en el frente de Pareto (modo multiobjetivo)


@dataclass
class ActivoPreparado:
    """Todo lo necesario para correr un activo ya validado.

    Agrupa los datos cargados, su huella, la configuracion de perturbaciones
    ligada a su tabla y la lista de estrategias COMPATIBLES con las columnas
    disponibles del activo. Pasar este objeto evita arrastrar 8 parametros
    sueltos entre las funciones del bucle.
    """

    activo: str
    df_base: Any
    timeframe_base: str
    huella_base: object
    perturbaciones: ConfiguracionPerturbaciones
    estrategias: list
    columnas_requeridas: set[str]
    permitir_huecos: bool
    # Fase 0: huella de reproducibilidad del run y ruta del fichero de datos.
    huella_repro: Any = None
    ruta_datos: Any = None


def main() -> None:
    mostrar_inicio_motor()
    validar_config(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fecha_inicio = _fecha_config(cfg.FECHA_INICIO, "FECHA_INICIO")
    fecha_fin = _fecha_config(cfg.FECHA_FIN, "FECHA_FIN")

    registro = cargar_estrategias()
    estrategias = obtener_estrategia(registro, cfg.ESTRATEGIA_ID)
    activos = _como_lista(cfg.ACTIVOS)
    timeframes = _como_lista(cfg.TIMEFRAMES)
    salidas = list(_salidas_a_ejecutar())
    n_jobs = _normalizar_jobs(cfg.N_JOBS)
    perturbaciones = ConfiguracionPerturbaciones.desde_config(cfg)
    if perturbaciones.activa:
        n_jobs_perturbaciones = _normalizar_jobs_perturbaciones(n_jobs)
        if n_jobs_perturbaciones != n_jobs:
            mostrar_aviso_perturbaciones(
                n_jobs_original=n_jobs,
                n_jobs_final=n_jobs_perturbaciones,
            )
        n_jobs = n_jobs_perturbaciones

    # Las combinaciones esperadas se acumulan por activo, porque cada activo
    # puede tener distinto numero de estrategias compatibles con sus columnas.
    combinaciones_ejecutadas: set[tuple[str, str, int, str]] = set()
    combinaciones_esperadas = 0
    total_runs = 0

    for activo in activos:
        preparado = _preparar_activo(activo, perturbaciones, estrategias)
        if preparado is None:
            continue
        combinaciones_esperadas += len(preparado.estrategias) * len(timeframes) * len(salidas)
        for timeframe in timeframes:
            total_runs += _procesar_timeframe(
                preparado=preparado,
                timeframe=timeframe,
                salidas=salidas,
                n_jobs=n_jobs,
                fecha_inicio=fecha_inicio,
                fecha_fin=fecha_fin,
                combinaciones_ejecutadas=combinaciones_ejecutadas,
            )
        del preparado
        gc.collect()

    if total_runs != combinaciones_esperadas:
        raise ValueError(
            f"[RUN] Runs ejecutados incorrectos: {total_runs} != {combinaciones_esperadas}."
        )

    mostrar_fin_backtest(total_runs)


def _preparar_activo(
    activo: str,
    perturbaciones: ConfiguracionPerturbaciones,
    estrategias: list,
) -> ActivoPreparado | None:
    """Carga y valida un activo y selecciona sus estrategias compatibles.

    Una estrategia es compatible si todas sus COLUMNAS_REQUERIDAS estan en los
    datos. Las incompatibles se avisan y se omiten en lugar de abortar toda la
    campana (p. ej. estrategias de order-flow sobre datos solo-OHLCV). Si
    ninguna es compatible, se omite el activo entero.
    """
    permitir_huecos = not _es_mercado_24_7(activo)
    ruta = ruta_datos(activo, cfg)
    df_base = cargar(activo, cfg)
    columnas_disponibles = set(df_base.columns)

    compatibles, incompatibles = _separar_estrategias_por_columnas(
        estrategias, columnas_disponibles
    )
    for estrategia, faltantes in incompatibles:
        print(
            f"[RUN] Aviso: se omite '{estrategia.NOMBRE}' (ID {estrategia.ID}) en "
            f"'{activo}': faltan columnas {sorted(faltantes)} en los datos."
        )
    if not compatibles:
        print(
            f"[RUN] Aviso: '{activo}' no tiene ninguna estrategia compatible con "
            "sus columnas; se omite el activo."
        )
        return None

    columnas_requeridas = _columnas_requeridas(compatibles)
    timeframe_base = inferir_timeframe(df_base)
    validar_datos(
        df_base,
        activo,
        columnas_requeridas,
        timeframe=timeframe_base,
        permitir_huecos=permitir_huecos,
    )
    huella_base = integridad.huella_dataframe(f"{activo} carga", df_base)
    mostrar_huella_datos(huella_base)
    perturbaciones_activo = perturbaciones.con_tabla_desde(df_base)
    huella_repro = reproducibilidad.calcular_huella(cfg, ruta)
    print(f"[REPRO] {activo}: {huella_repro.resumen()}")

    return ActivoPreparado(
        activo=activo,
        df_base=df_base,
        timeframe_base=timeframe_base,
        huella_base=huella_base,
        perturbaciones=perturbaciones_activo,
        estrategias=compatibles,
        columnas_requeridas=columnas_requeridas,
        permitir_huecos=permitir_huecos,
        huella_repro=huella_repro,
        ruta_datos=ruta,
    )


def _separar_estrategias_por_columnas(
    estrategias: list,
    columnas_disponibles: set[str],
) -> tuple[list, list[tuple[Any, set[str]]]]:
    """Divide las estrategias en (compatibles, [(incompatible, columnas_faltantes)])."""
    compatibles: list = []
    incompatibles: list[tuple[Any, set[str]]] = []
    for estrategia in estrategias:
        requeridas = set(getattr(estrategia, "COLUMNAS_REQUERIDAS", set()))
        faltantes = requeridas - columnas_disponibles
        if faltantes:
            incompatibles.append((estrategia, faltantes))
        else:
            compatibles.append(estrategia)
    return compatibles, incompatibles


def _procesar_timeframe(
    *,
    preparado: ActivoPreparado,
    timeframe: str,
    salidas: list[ExitConfig],
    n_jobs: int,
    fecha_inicio: date,
    fecha_fin: date,
    combinaciones_ejecutadas: set[tuple[str, str, int, str]],
) -> int:
    """Resamplea y valida un timeframe y corre todas sus (estrategia x salida).

    Devuelve cuantos runs se completaron. El contexto y el df resampleado se
    crean una sola vez por timeframe y se comparten entre estrategias.
    """
    activo = preparado.activo
    df_base = preparado.df_base
    df_tf = resamplear(df_base, timeframe)
    validar_datos(
        df_tf,
        f"{activo} {timeframe}",
        preparado.columnas_requeridas,
        timeframe=timeframe,
        permitir_huecos=preparado.permitir_huecos,
    )
    integridad.verificar_resampleo(df_base, df_tf, timeframe)
    huella_tf = integridad.huella_dataframe(f"{activo} {timeframe}", df_tf)
    mostrar_huella_datos(huella_tf)
    ctx = crear_contexto(df_base=df_base, df_tf=df_tf, timeframe=timeframe)

    runs = 0
    for estrategia in preparado.estrategias:
        # Sin perturbaciones los buffers son estables para toda la combinacion.
        # Con perturbaciones se clona y vincula por trial.
        if not preparado.perturbaciones.activa:
            estrategia.bind(ctx.arrays_tf, CacheIndicadores())
        try:
            for salida in salidas:
                _verificar_combinacion_unica(
                    combinaciones_ejecutadas, activo, timeframe, estrategia, salida
                )
                _ejecutar_combinacion(
                    preparado=preparado,
                    ctx=ctx,
                    timeframe=timeframe,
                    estrategia=estrategia,
                    salida=salida,
                    n_jobs=n_jobs,
                    fecha_inicio=fecha_inicio,
                    fecha_fin=fecha_fin,
                )
                runs += 1
        finally:
            estrategia.desvincular()

    del df_tf, ctx
    gc.collect()
    return runs


def _verificar_combinacion_unica(
    combinaciones_ejecutadas: set[tuple[str, str, int, str]],
    activo: str,
    timeframe: str,
    estrategia,
    salida: ExitConfig,
) -> None:
    clave = (str(activo), str(timeframe), int(estrategia.ID), str(salida.tipo))
    if clave in combinaciones_ejecutadas:
        raise ValueError(f"[RUN] Combinacion duplicada detectada: {clave}.")
    combinaciones_ejecutadas.add(clave)


def _ejecutar_combinacion(
    *,
    preparado: ActivoPreparado,
    ctx: ContextoCombinacion,
    timeframe: str,
    estrategia,
    salida: ExitConfig,
    n_jobs: int,
    fecha_inicio: date,
    fecha_fin: date,
) -> None:
    """Optimiza una combinacion completa, guarda resultados y genera reportes.

    Un fallo aqui se envuelve con el contexto (activo/timeframe/estrategia/salida)
    para que el error diga exactamente que combinacion rompio.
    """
    activo = preparado.activo
    try:
        # Ruta de la combinacion SIN borrar: solo para mostrarla en el monitor.
        # El borrado de resultados previos se hace despues, ya con exito.
        resultados_base = ruta_base_combinacion(
            carpeta_resultados=cfg.CARPETA_RESULTADOS,
            activo=activo,
            timeframe=timeframe,
            estrategia_nombre=estrategia.NOMBRE,
            exit_type=salida.tipo,
        )
        trials = _optimizar_combinacion(
            activo=activo,
            timeframe=timeframe,
            estrategia=estrategia,
            salida_base=salida,
            ctx=ctx,
            timeframe_base=preparado.timeframe_base,
            n_jobs=n_jobs,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            perturbaciones=preparado.perturbaciones,
            resultados_base=resultados_base,
        )
    except Exception as exc:
        contexto = f"{activo}/{timeframe}/{estrategia.NOMBRE}/{salida.tipo}"
        raise RuntimeError(f"[RUN] Fallo en {contexto}: {exc}") from exc

    mejor = _seleccionar_mejor(trials)
    if mejor.replay is None:
        raise RuntimeError("[RUN] El mejor trial no tiene replay materializado.")

    # Borrado-tras-exito: la carpeta final de la combinacion se reemplaza por
    # completo (estrategia/timeframe/salida/activo). Relanzar la misma
    # combinacion sustituye el resultado anterior.
    run_dir = preparar_resultados_combinacion(
        carpeta_resultados=cfg.CARPETA_RESULTADOS,
        activo=activo,
        timeframe=timeframe,
        estrategia_nombre=estrategia.NOMBRE,
        exit_type=salida.tipo,
    )

    # Fase 0: registrar TODOS los trials en la base de experimentos. Esto
    # alimenta el conteo real de `N` para el Deflated Sharpe (Fase 3): el `N`
    # correcto no son los trials de este run, sino todos los probados en el
    # activo a lo largo de la investigación. La traza completa (huella,
    # metadatos, trials) vive en la BD, no en ficheros sueltos en la carpeta.
    _registrar_experimento(
        preparado=preparado,
        timeframe=timeframe,
        estrategia=estrategia,
        salida=salida,
        trials=trials,
    )

    # Carpeta final: exactamente cuatro ficheros (Excel + 2 HTML + informe
    # avanzado), todos del mejor trial / de la optimización.
    excel_path = (
        generar_excel(run_dir, trials, mejor, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
        if cfg.USAR_EXCEL else None
    )
    html_paths = generar_htmls(
        run_dir=run_dir,
        df=ctx.df_tf,
        df_indicadores=ctx.df_tf,
        trials=trials,
        estrategia=estrategia,
        grafica_rango=cfg.GRAFICA_RANGO,
        grafica_desde=cfg.GRAFICA_DESDE,
        grafica_hasta=cfg.GRAFICA_HASTA,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
    )
    informe_path = generar_informe(
        run_dir=run_dir,
        trials=trials,
        estrategia=estrategia,
        activo=activo,
        timeframe=timeframe,
        salida_tipo=salida.tipo,
    )

    # Validación fuera de muestra (CPCV/WFA/DSR/robustez) + veredicto e informe
    # avanzado. Se ejecuta tras registrar el run, para que el N del DSR ya
    # incluya los trials de esta combinación.
    _validar_y_reportar_oos(
        preparado=preparado,
        ctx=ctx,
        timeframe=timeframe,
        estrategia=estrategia,
        salida=salida,
        mejor=mejor,
        run_dir=run_dir,
    )

    mostrar_resumen_run(
        mejor=mejor,
        total_trials=cfg.N_TRIALS,
        run_dir=run_dir,
        excel_path=excel_path,
        html_paths=html_paths,
        informe_path=informe_path,
        dashboard_path=None,
    )
    del trials, mejor, run_dir, excel_path, html_paths, informe_path
    gc.collect()


def _registrar_experimento(
    *,
    preparado: ActivoPreparado,
    timeframe: str,
    estrategia,
    salida: ExitConfig,
    trials: list[TrialResultado],
) -> None:
    """Persiste el run y todos sus trials en la base de experimentos (Fase 0).

    Una incidencia aquí no debe perder los resultados ya guardados en disco,
    pero sí debe avisar de forma visible: un registro corrupto invalida el `N`
    del Deflated Sharpe. Por eso se captura, se informa y se continúa, en lugar
    de abortar la campaña entera.
    """
    huella = preparado.huella_repro
    hb = preparado.huella_base
    try:
        with RegistroExperimentos(cfg.BD_EXPERIMENTOS) as registro:
            run_id = registro.registrar_run(
                activo=preparado.activo,
                timeframe=timeframe,
                estrategia_id=int(estrategia.ID),
                estrategia_nombre=estrategia.NOMBRE,
                salida_tipo=salida.tipo,
                modo=getattr(cfg, "MODO", "investigacion"),
                n_trials=int(cfg.N_TRIALS),
                sampler=getattr(cfg, "OPTUNA_SAMPLER", None),
                funcion_score=getattr(cfg, "FUNCION_SCORE", None),
                huella=huella.como_dict() if huella is not None else None,
                ts_datos_inicio=str(getattr(hb, "ts_inicio", None)),
                ts_datos_fin=str(getattr(hb, "ts_fin", None)),
                filas_datos=getattr(hb, "filas", None),
            )
            n = registro.registrar_trials(run_id, trials)
        print(f"[REGISTRO] run {run_id}: {n} trials registrados en la base de experimentos.")
    except Exception as exc:  # noqa: BLE001 - se informa y se continúa
        print(
            f"[REGISTRO] AVISO: no se pudo registrar el experimento "
            f"({preparado.activo}/{timeframe}/{estrategia.NOMBRE}/{salida.tipo}): {exc}"
        )


# ---------------------------------------------------------------------------
# Validación fuera de muestra (Fases 2-7): adaptador del motor + orquestación
# ---------------------------------------------------------------------------

def _params_estrategia(parametros: dict) -> dict:
    """Filtra los parámetros de estrategia (excluye los de salida/riesgo)."""
    return {
        k: v for k, v in parametros.items()
        if not k.startswith("exit_") and not k.startswith("risk_")
    }


def _aplicar_penalizaciones_robustez(score: float, n_trades: int, n_parametros: int) -> float:
    """Aplica las penalizaciones de turnover y complejidad (Fase 4).

    Con los factores a 0 (por defecto) devuelve el score intacto, así que el
    comportamiento no cambia salvo que se activen explícitamente en config.
    """
    score = penalizacion_turnover(
        score,
        n_trades,
        trades_objetivo=int(getattr(cfg, "TURNOVER_OBJETIVO", 100)),
        factor=float(getattr(cfg, "PENALIZACION_TURNOVER_FACTOR", 0.0)),
    )
    score = penalizacion_complejidad(
        score,
        n_parametros,
        factor=float(getattr(cfg, "PENALIZACION_COMPLEJIDAD_FACTOR", 0.0)),
    )
    return score


def _simular_con_senales(*, ctx, salida_trial, senales_tf, salidas_custom=None) -> dict:
    """Simula sobre la serie COMPLETA dadas unas señales ya construidas.

    Reaprovecha la misma maquinaria que el replay (proyectar → simular_full), de
    modo que el realismo de ejecución coincide con el del backtest principal.
    Devuelve los trades (idx_entrada, pnl).
    """
    arrays_exec, senales_exec, salidas_exec = _preparar_ejecucion(
        ctx=ctx, salida=salida_trial, senales_tf=senales_tf, salidas_custom=salidas_custom
    )
    sim_result = simular_full(
        arrays_exec,
        senales_exec,
        sim_cfg=_sim_config(salida_trial, ctx=ctx),
        salidas_custom=salidas_exec,
    )
    trades = sim_result.take_trades()
    return {"idx_entrada": trades["idx_entrada"], "pnl": trades["pnl"]}


def _simular_trades_para_validacion(*, ctx, estrategia, salida_trial, params_estrategia) -> dict:
    """Genera las señales de la estrategia y simula (delega en `_simular_con_senales`)."""
    senales_tf = estrategia.generar_senales(ctx.df_tf, params_estrategia)
    salidas_custom = (
        estrategia.generar_salidas(ctx.df_tf, params_estrategia)
        if salida_trial.tipo == "CUSTOM"
        else None
    )
    return _simular_con_senales(
        ctx=ctx, salida_trial=salida_trial, senales_tf=senales_tf, salidas_custom=salidas_custom
    )


def _breakdown_regimen(ctx, trades: dict, saldo_por_trade: float) -> dict:
    """Rendimiento del mejor trial separado por régimen de mercado (MACD)."""
    idx = np.asarray(trades.get("idx_entrada", []), dtype=np.int64)
    pnl = np.asarray(trades.get("pnl", []), dtype=np.float64)
    if idx.size == 0:
        return {}
    etiquetas = robustez_regimen.etiquetas_macd(ctx.arrays_base.closes)
    orden = np.argsort(idx, kind="stable")
    retornos = pnl[orden] / (float(saldo_por_trade) or 1.0)
    labels = etiquetas[idx[orden]]
    res = robustez_regimen.rendimiento_por_regimen(retornos, labels)
    return {
        k: {"sharpe": v.sharpe, "n_trades": v.n_trades, "retorno_total": v.retorno_total}
        for k, v in res.items()
    }


def _contraste_nula(*, ctx, salida, mejor, sharpe_real: float) -> str | None:
    """Control de laboratorio: ¿bate el mejor trial a entradas aleatorias?

    Corre `VALIDACION_NULA_ITER` estrategias de entradas aleatorias por la MISMA
    maquinaria (mismos costes y salida), con una frecuencia de señales parecida a
    la real, y contrasta el Sharpe real contra esa distribución nula.
    """
    n_iter = int(getattr(cfg, "VALIDACION_NULA_ITER", 0) or 0)
    if n_iter <= 0 or salida.tipo == "CUSTOM" or not np.isfinite(sharpe_real):
        return None

    n_bars = int(ctx.df_tf.height)
    conteo = mejor.conteo_senales or {}
    n_senales = int(conteo.get(1, 0)) + int(conteo.get(-1, 0))
    p = min(max(n_senales / n_bars, 1e-4), 0.5) if n_bars else 0.0
    if p <= 0.0:
        return None
    saldo_por_trade = float(cfg.SALDO_USADO_POR_TRADE)

    def generador(rng: np.random.Generator) -> float:
        senales = _senales_aleatorias(rng, n_bars, p)
        trades = _simular_con_senales(ctx=ctx, salida_trial=salida, senales_tf=senales)
        r = retornos_por_trade(trades, saldo_por_trade=saldo_por_trade)
        if r.size < 2:
            return 0.0
        sd = float(r.std(ddof=1))
        return float(r.mean() / sd) if sd > 0.0 else 0.0

    dist = robustez_nula.distribucion_nula(n_iter, generador, seed=_optuna_seed())
    return robustez_nula.contrastar(float(sharpe_real), dist).resumen()


def _senales_aleatorias(rng: np.random.Generator, n: int, p: float) -> np.ndarray:
    """Vector de señales aleatorias int8 en {-1,0,1} con prob. `p` de entrada."""
    senales = np.zeros(n, dtype=np.int8)
    mask = rng.random(n) < p
    signos = rng.choice(np.array([-1, 1], dtype=np.int8), size=int(mask.sum()))
    senales[mask] = signos
    return senales


def _construir_callbacks_validacion(*, ctx, estrategia, salida_base: ExitConfig):
    """Crea los callbacks `optimizar(indices)` y `evaluar(config, indices)`.

    - `optimizar` lanza una optimización de Optuna RESTRINGIDA a los índices de
      train del fold (puntúa solo con los trades que entran en esa ventana) y
      devuelve la mejor configuración (params + salida) sin tocar el test.
    - `evaluar` mide esa configuración sobre cualquier conjunto de índices SIN
      reoptimizar. Es la garantía OOS de la Fase 2.
    """
    saldo_inicial = float(cfg.SALDO_INICIAL)
    saldo_por_trade = float(cfg.SALDO_USADO_POR_TRADE)
    min_trades = max(1, int(getattr(cfg, "MIN_TRADES_SCORE", 0) or 0))
    funcion = str(getattr(cfg, "FUNCION_SCORE", "PSR")).upper()
    clave = {"PSR": "psr", "SHARPE": "sharpe_ratio", "ROI": "roi_total"}.get(funcion, "psr")

    def _puntuar(metricas: dict) -> float:
        if int(metricas.get("total_trades", 0)) < min_trades:
            return -1e9  # un fold sin muestra suficiente no puede ganar
        valor = metricas.get(clave, 0.0)
        try:
            f = float(valor)
        except (TypeError, ValueError):
            return -1e9
        return f if np.isfinite(f) else -1e9

    def optimizar(indices_train: np.ndarray):
        sampler = crear_sampler(cfg.OPTUNA_SAMPLER, _optuna_seed(), int(cfg.VALIDACION_N_TRIALS))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        mejor = {"score": float("-inf"), "config": None}

        def objetivo(trial: optuna.Trial) -> float:
            params_est = estrategia.espacio_busqueda(trial)
            salida_trial, _ = _salida_para_trial(salida_base, trial)
            trades = _simular_trades_para_validacion(
                ctx=ctx, estrategia=estrategia, salida_trial=salida_trial,
                params_estrategia=params_est,
            )
            m = metricas_en_indices(
                trades, indices_train, saldo_inicial=saldo_inicial, saldo_por_trade=saldo_por_trade
            )
            s = _puntuar(m)
            if s > -1e9:
                s = _aplicar_penalizaciones_robustez(
                    s, int(m.get("total_trades", 0)), len(params_est)
                )
            if s > mejor["score"]:
                mejor["score"] = s
                mejor["config"] = (dict(params_est), salida_trial)
            return s

        study.optimize(objetivo, n_trials=int(cfg.VALIDACION_N_TRIALS), n_jobs=1)
        if mejor["config"] is None:
            raise RuntimeError("[VALIDACION] La optimización del fold no produjo configuración.")
        return mejor["config"]

    def evaluar(config, indices: np.ndarray) -> dict:
        params_est, salida_trial = config
        trades = _simular_trades_para_validacion(
            ctx=ctx, estrategia=estrategia, salida_trial=salida_trial, params_estrategia=params_est
        )
        return metricas_en_indices(
            trades, indices, saldo_inicial=saldo_inicial, saldo_por_trade=saldo_por_trade
        )

    return optimizar, evaluar


def _stats_registro_para_dsr(activo: str, estrategia_id: int) -> tuple[int, float]:
    """N real (configuraciones probadas) y varianza de sus Sharpes, desde la BD."""
    try:
        with RegistroExperimentos(cfg.BD_EXPERIMENTOS) as registro:
            n = registro.contar_configuraciones(activo)
            sharpes = registro.sharpes_configuraciones(activo)
    except Exception:
        return 1, 0.0
    n = max(1, int(n))
    arr = np.asarray(sharpes, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    var = float(arr.var(ddof=1)) if arr.size >= 2 else 0.0
    return n, var


def _validar_y_reportar_oos(*, preparado, ctx, timeframe, estrategia, salida, mejor, run_dir) -> None:
    """Ejecuta la validación OOS completa, emite el veredicto y el informe.

    Defensiva: un fallo aquí informa y continúa, sin perder los resultados ya
    guardados ni abortar la campaña. No es compatible con perturbaciones.
    """
    if not getattr(cfg, "VALIDACION_ACTIVA", False):
        return
    if preparado.perturbaciones.activa:
        print("[VALIDACION] Omitida: no es compatible con perturbaciones activas.")
        return

    contexto = f"{preparado.activo}/{timeframe}/{estrategia.NOMBRE}/{salida.tipo}"
    try:
        n_obs = len(ctx.arrays_base)
        optimizar, evaluar = _construir_callbacks_validacion(
            ctx=ctx, estrategia=estrategia, salida_base=salida
        )

        trades_mejor = _simular_trades_para_validacion(
            ctx=ctx, estrategia=estrategia, salida_trial=mejor.salida,
            params_estrategia=_params_estrategia(mejor.parametros),
        )
        retornos_mejor = retornos_por_trade(trades_mejor, saldo_por_trade=cfg.SALDO_USADO_POR_TRADE)

        # Fase 5: rendimiento por régimen (MACD) y control vs. estrategia nula.
        regimen = _breakdown_regimen(ctx, trades_mejor, float(cfg.SALDO_USADO_POR_TRADE))
        nula_resumen = _contraste_nula(
            ctx=ctx, salida=mejor.salida, mejor=mejor,
            sharpe_real=float(mejor.metricas.get("sharpe_ratio", 0.0)),
        )

        n_cfg, var_sharpes = _stats_registro_para_dsr(preparado.activo, int(estrategia.ID))
        equity_valores = (
            mejor.replay.equity_curve.tolist()
            if mejor.replay is not None and mejor.replay.equity_curve is not None
            else None
        )
        huella = preparado.huella_repro.digest[:12] if preparado.huella_repro else None

        resultado = ejecutar_validacion_completa(
            n_obs=n_obs,
            optimizar=optimizar,
            evaluar=evaluar,
            cpcv_grupos=int(cfg.VALIDACION_N_GRUPOS),
            cpcv_k=int(cfg.VALIDACION_K),
            embargo=float(cfg.VALIDACION_EMBARGO),
            duracion_trade=int(cfg.VALIDACION_DURACION_TRADE),
            wfa_activa=bool(cfg.VALIDACION_WFA_ACTIVA),
            wfa_ventanas=int(cfg.VALIDACION_WFA_VENTANAS),
            wfa_fraccion=float(cfg.VALIDACION_WFA_FRACCION),
            wfa_anchored=bool(cfg.VALIDACION_WFA_ANCHORED),
            sharpe_hat=float(mejor.metricas.get("sharpe_ratio", 0.0)),
            n_trades=int(mejor.metricas.get("total_trades", 0)),
            n_configuraciones=n_cfg,
            varianza_sharpe_trials=var_sharpes,
            sharpe_anual_objetivo=float(cfg.VALIDACION_SHARPE_ANUAL_OBJETIVO),
            retornos_mejor=retornos_mejor,
            capital_inicial=float(cfg.SALDO_INICIAL),
            bootstrap_iter=int(cfg.VALIDACION_BOOTSTRAP_ITER),
            bootstrap_bloque=int(cfg.VALIDACION_BOOTSTRAP_BLOQUE),
            bootstrap_seed=_optuna_seed(),
            regimen=regimen,
            nula_resumen=nula_resumen,
            cabecera={
                "estrategia": estrategia.NOMBRE,
                "activo": preparado.activo,
                "timeframe": timeframe,
                "salida": salida.tipo,
                "modo": getattr(cfg, "MODO", "investigacion"),
                "huella": huella,
            },
            is_extra={
                "mejor_score": float(mejor.score),
                "sharpe_is": float(mejor.metricas.get("sharpe_ratio", 0.0)),
                "total_trades_is": int(mejor.metricas.get("total_trades", 0)),
            },
            equity_valores=equity_valores,
            indice_holdout=None,
        )

        ruta = (run_dir / "informe_avanzado.html") if run_dir is not None else None
        generar_informe_institucional(resultado.datos_informe, ruta_salida=ruta)
        print(f"[VALIDACION] {contexto}: {resultado.veredicto.resumen()}")
    except Exception as exc:  # noqa: BLE001 - se informa y se continúa
        print(f"[VALIDACION] AVISO: validación OOS no completada ({contexto}): {exc}")


def _optimizar_combinacion(
    *,
    activo: str,
    timeframe: str,
    estrategia,
    salida_base: ExitConfig,
    ctx: ContextoCombinacion,
    timeframe_base: str,
    n_jobs: int,
    fecha_inicio: date,
    fecha_fin: date,
    perturbaciones: ConfiguracionPerturbaciones,
    resultados_base: Any,
) -> list[TrialResultado]:
    seed_activa = _seed_activa()
    multiobjetivo = bool(getattr(cfg, "OPTUNA_MULTIOBJETIVO", False))
    if multiobjetivo:
        # Única búsqueda, pero multiobjetivo: NSGA-II saca el frente de Pareto.
        sampler = optuna.samplers.NSGAIISampler(seed=_optuna_seed())
        study = optuna.create_study(directions=list(DIRECCIONES_PARETO), sampler=sampler)
    else:
        sampler = crear_sampler(cfg.OPTUNA_SAMPLER, _optuna_seed(), cfg.N_TRIALS)
        study = optuna.create_study(direction="maximize", sampler=sampler)
    resultados: list[TrialResultado] = []
    lock = Lock()

    # Sin perturbaciones, la estrategia ya esta vinculada por main() a buffers
    # estables. Con perturbaciones, cada trial crea su propio contexto y clone.

    def objective(trial: optuna.Trial) -> float:
        perturbacion_seed = seed_para_trial(
            perturbaciones,
            trial_numero=trial.number,
            activo=activo,
            timeframe=timeframe,
            estrategia_id=estrategia.ID,
            salida_tipo=salida_base.tipo,
        )
        ctx_trial = _ctx_para_trial(
            ctx=ctx,
            timeframe=timeframe,
            perturbaciones=perturbaciones,
            seed=perturbacion_seed,
        )
        estrategia_trial = _estrategia_para_trial(estrategia, perturbaciones)
        if perturbaciones.activa:
            estrategia_trial.bind(ctx_trial.arrays_tf, CacheIndicadores())

        params_estrategia = estrategia.espacio_busqueda(trial)
        salida_trial, params_salida = _salida_para_trial(salida_base, trial)
        parametros = {**params_estrategia, **params_salida}

        try:
            senales_tf = estrategia_trial.generar_senales(ctx_trial.df_tf, params_estrategia)
            conteo = integridad.verificar_senales(ctx_trial.df_tf, senales_tf)
            salidas_custom = (
                estrategia_trial.generar_salidas(ctx_trial.df_tf, params_estrategia)
                if salida_trial.tipo == "CUSTOM"
                else None
            )
            conteo_salidas = (
                integridad.verificar_salidas_custom(ctx_trial.df_tf, salidas_custom)
                if salidas_custom is not None
                else None
            )

            arrays_exec, senales_exec, salidas_exec = _preparar_ejecucion(
                ctx=ctx_trial,
                salida=salida_trial,
                senales_tf=senales_tf,
                salidas_custom=salidas_custom,
            )
            timeframe_ejecucion = _timeframe_ejecucion(
                ctx=ctx_trial,
                timeframe=timeframe,
                timeframe_base=timeframe_base,
            )

            metricas_obj = simular_metricas(
                arrays_exec,
                senales_exec,
                sim_cfg=_sim_config(salida_trial, ctx=ctx_trial),
                salidas_custom=salidas_exec,
            )
            metricas = calcular_metricas(metricas_obj, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
            score = calcular_score(metricas)
            score = _aplicar_penalizaciones_robustez(
                score, int(metricas.get("total_trades", 0)), len(params_estrategia)
            )
            trial.set_user_attr("metricas", metricas)
            trial.set_user_attr("conteo_senales", conteo)
            trial.set_user_attr("conteo_salidas", conteo_salidas)
            trial.set_user_attr("perturbacion_seed", perturbacion_seed)

            trial_resultado = TrialResultado(
                numero=trial.number,
                activo=activo,
                timeframe=timeframe,
                timeframe_ejecucion=timeframe_ejecucion,
                estrategia_id=int(estrategia.ID),
                estrategia_nombre=estrategia.NOMBRE,
                salida=salida_trial,
                parametros=parametros,
                score=float(score),
                metricas=metricas,
                conteo_senales=conteo,
                conteo_salidas=conteo_salidas,
                perturbacion_seed=perturbacion_seed,
            )
            with lock:
                resultados.append(trial_resultado)
            monitor.registrar(
                trial_number=trial.number,
                score=float(score),
                metricas=metricas,
                params=_params_para_monitor(parametros, salida_trial),
            )
            if multiobjetivo:
                return vector_pareto(metricas)
            return float(score)
        finally:
            if perturbaciones.activa:
                estrategia_trial.desvincular()

    with MonitorOptimizacion(
        activo=activo,
        timeframe=timeframe,
        estrategia=estrategia.NOMBRE,
        salida=salida_base.tipo,
        total_trials=cfg.N_TRIALS,
        sampler=cfg.OPTUNA_SAMPLER,
        n_jobs=n_jobs,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        perturbaciones=perturbaciones.activa,
        seed_activa=seed_activa,
        resultados_dir=resultados_base,
    ) as monitor:
        study.optimize(objective, n_trials=cfg.N_TRIALS, n_jobs=n_jobs)

    if len(resultados) != cfg.N_TRIALS:
        raise ValueError(
            f"[OPTUNA] Trials conservados incorrectos: {len(resultados)} != {cfg.N_TRIALS}."
        )

    if multiobjetivo:
        _marcar_pareto(study, resultados)
    else:
        _verificar_mejor_de_study(study, resultados)

    # Replay determinista SOLO de la configuración elegida (la que alimenta los
    # reportes): el mejor por score en modo escalar, o la meseta del frente de
    # Pareto en multiobjetivo. `_seleccionar_mejor` es determinista, así que el
    # mismo trial se reusa luego en `_ejecutar_combinacion`.
    top = [_seleccionar_mejor(resultados)]
    for trial_res in top:
        _replay_trial(
            trial_res=trial_res,
            ctx=ctx,
            estrategia=estrategia,
            fecha_inicio=fecha_inicio,
            fecha_fin=fecha_fin,
            timeframe=timeframe,
            perturbaciones=perturbaciones,
        )

    return resultados


def _replay_trial(
    *,
    trial_res: TrialResultado,
    ctx: ContextoCombinacion,
    estrategia,
    fecha_inicio: date,
    fecha_fin: date,
    timeframe: str,
    perturbaciones: ConfiguracionPerturbaciones,
) -> None:
    """Reconstruye trades + equity_curve para un trial que va a generar reporte.

    Verifica que los escalares clave (saldo_final, total_trades) coinciden
    con los registrados durante la fase Optuna — garantía de determinismo.
    """
    salida = trial_res.salida
    ctx_trial = _ctx_para_trial(
        ctx=ctx,
        timeframe=timeframe,
        perturbaciones=perturbaciones,
        seed=trial_res.perturbacion_seed,
    )
    estrategia_trial = _estrategia_para_trial(estrategia, perturbaciones)
    if perturbaciones.activa:
        estrategia_trial.bind(ctx_trial.arrays_tf, CacheIndicadores())

    params_estrategia = {
        k: v for k, v in trial_res.parametros.items()
        if not k.startswith("exit_") and not k.startswith("risk_")
    }
    try:
        senales_tf = estrategia_trial.generar_senales(ctx_trial.df_tf, params_estrategia)
        salidas_custom = (
            estrategia_trial.generar_salidas(ctx_trial.df_tf, params_estrategia)
            if salida.tipo == "CUSTOM"
            else None
        )
        arrays_exec, senales_exec, salidas_exec = _preparar_ejecucion(
            ctx=ctx_trial,
            salida=salida,
            senales_tf=senales_tf,
            salidas_custom=salidas_custom,
        )
        sim_result = simular_full(
            arrays_exec,
            senales_exec,
            sim_cfg=_sim_config(salida, ctx=ctx_trial),
            salidas_custom=salidas_exec,
        )
        indicadores = (
            estrategia_trial.indicadores_para_grafica(ctx_trial.df_tf, trial_res.parametros)
            if perturbaciones.activa
            else None
        )
    finally:
        if perturbaciones.activa:
            estrategia_trial.desvincular()

    metricas_obj = sim_result.metricas
    trades = sim_result.take_trades()  # consume el SimResult, libera RAM Rust
    equity_curve = trades.pop("equity_curve")

    # Verificación de determinismo y de integridad columnar.
    if int(metricas_obj.total_trades) != int(trial_res.metricas["total_trades"]):
        raise RuntimeError(
            f"[REPLAY] total_trades difiere entre Optuna y replay: "
            f"{trial_res.metricas['total_trades']} vs {metricas_obj.total_trades}."
        )
    if abs(float(metricas_obj.saldo_final) - float(trial_res.metricas["saldo_final"])) > 1e-6:
        raise RuntimeError("[REPLAY] saldo_final difiere entre Optuna y replay.")

    # Para la verificación, las señales deben venir como ndarray del mismo
    # buffer que se pasó al motor (idéntica fuente de verdad).
    senales_arr = senales_exec.to_numpy() if hasattr(senales_exec, "to_numpy") else senales_exec
    integridad.verificar_resultado(
        arrays_exec, senales_arr, trades, equity_curve, metricas_obj,
    )

    trial_res.replay = ReplayTrial(
        metricas_obj=metricas_obj,
        trades=trades,
        equity_curve=equity_curve,
        df_tf=ctx_trial.df_tf if perturbaciones.activa else None,
        indicadores=indicadores,
        perturbacion_seed=trial_res.perturbacion_seed,
    )


def _ctx_para_trial(
    *,
    ctx: ContextoCombinacion,
    timeframe: str,
    perturbaciones: ConfiguracionPerturbaciones,
    seed: int | None,
) -> ContextoCombinacion:
    if not perturbaciones.activa:
        return ctx

    df_base = aplicar_perturbaciones(ctx.df_base, perturbaciones, seed=seed)
    return _crear_contexto_perturbado_con_plan(ctx=ctx, df_base=df_base, timeframe=timeframe)


def _crear_contexto_perturbado_con_plan(
    *,
    ctx: ContextoCombinacion,
    df_base: pl.DataFrame,
    timeframe: str,
) -> ContextoCombinacion:
    arrays_base = construir_arrays_motor(df_base)
    if ctx.es_min_tf:
        return ContextoCombinacion(
            df_tf=df_base,
            df_base=df_base,
            arrays_tf=arrays_base,
            arrays_base=arrays_base,
            tf_to_base_idx=ctx.tf_to_base_idx,
            es_min_tf=True,
        )

    df_tf = _resamplear_perturbado_con_plan(df_base=df_base, ctx=ctx, timeframe=timeframe)
    return ContextoCombinacion(
        df_tf=df_tf,
        df_base=df_base,
        arrays_tf=construir_arrays_motor(df_tf),
        arrays_base=arrays_base,
        tf_to_base_idx=ctx.tf_to_base_idx,
        es_min_tf=False,
    )


def _resamplear_perturbado_con_plan(
    *,
    df_base: pl.DataFrame,
    ctx: ContextoCombinacion,
    timeframe: str,
) -> pl.DataFrame:
    del timeframe  # El plan temporal ya fue validado al crear `ctx`.
    starts = np.searchsorted(ctx.arrays_base.timestamps, ctx.arrays_tf.timestamps).astype(np.int64)
    ends = ctx.tf_to_base_idx
    _validar_plan_resampleo_perturbado(starts, ends, df_base.height)

    columnas: dict[str, object] = {"timestamp": ctx.df_tf["timestamp"]}
    for columna in df_base.columns:
        if columna == "timestamp":
            continue
        regla = regla_agregacion(columna)
        valores = df_base[columna].to_numpy()
        columnas[columna] = _agregar_columna_por_plan(valores, starts, ends, regla)
    return pl.DataFrame(columnas)


def _validar_plan_resampleo_perturbado(starts: np.ndarray, ends: np.ndarray, n_base: int) -> None:
    if starts.shape != ends.shape:
        raise ValueError("[PERT] Plan de resampleo inconsistente: starts y ends difieren.")
    if starts.size == 0:
        raise ValueError("[PERT] Plan de resampleo vacio para contexto perturbado.")
    if (starts < 0).any() or (ends < 0).any() or (starts >= n_base).any() or (ends >= n_base).any():
        raise ValueError("[PERT] Plan de resampleo fuera de rango.")
    if (ends < starts).any():
        raise ValueError("[PERT] Plan de resampleo con ventanas invertidas.")


def _agregar_columna_por_plan(
    valores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    regla: str,
) -> np.ndarray:
    valores = np.ascontiguousarray(valores)
    if regla == "first":
        return valores[starts]
    if regla == "last":
        return valores[ends]
    if regla == "max":
        return _segment_max_f64(valores.astype(np.float64, copy=False), starts, ends)
    if regla == "min":
        return _segment_min_f64(valores.astype(np.float64, copy=False), starts, ends)
    if regla == "sum":
        if np.issubdtype(valores.dtype, np.integer):
            return _segment_sum_i64(valores.astype(np.int64, copy=False), starts, ends)
        return _segment_sum_f64(valores.astype(np.float64, copy=False), starts, ends)
    raise ValueError(f"Regla de resampleo no soportada para '{regla}'.")


def _jit_cache(func):
    if njit is None:
        return func
    return njit(cache=True, nogil=True)(func)


@_jit_cache
def _segment_sum_f64(valores: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    out = np.empty(starts.shape[0], dtype=np.float64)
    for i in range(starts.shape[0]):
        total = 0.0
        for j in range(starts[i], ends[i] + 1):
            total += valores[j]
        out[i] = total
    return out


@_jit_cache
def _segment_sum_i64(valores: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    out = np.empty(starts.shape[0], dtype=np.int64)
    for i in range(starts.shape[0]):
        total = 0
        for j in range(starts[i], ends[i] + 1):
            total += valores[j]
        out[i] = total
    return out


@_jit_cache
def _segment_max_f64(valores: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    out = np.empty(starts.shape[0], dtype=np.float64)
    for i in range(starts.shape[0]):
        maximo = valores[starts[i]]
        for j in range(starts[i] + 1, ends[i] + 1):
            if valores[j] > maximo:
                maximo = valores[j]
        out[i] = maximo
    return out


@_jit_cache
def _segment_min_f64(valores: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    out = np.empty(starts.shape[0], dtype=np.float64)
    for i in range(starts.shape[0]):
        minimo = valores[starts[i]]
        for j in range(starts[i] + 1, ends[i] + 1):
            if valores[j] < minimo:
                minimo = valores[j]
        out[i] = minimo
    return out


def _estrategia_para_trial(estrategia, perturbaciones: ConfiguracionPerturbaciones):
    if not perturbaciones.activa:
        return estrategia
    return type(estrategia)()


def _seed_activa() -> bool:
    return bool(getattr(cfg, "USAR_SEED", True))


def _optuna_seed() -> int | None:
    if not _seed_activa():
        return None
    return getattr(cfg, "OPTUNA_SEED", None)


def _preparar_ejecucion(
    *,
    ctx: ContextoCombinacion,
    salida: ExitConfig,
    senales_tf,
    salidas_custom,
) -> tuple[ArraysMotor, Any, Any]:
    if ctx.es_min_tf:
        return ctx.arrays_tf, senales_tf, salidas_custom

    senales_base = proyectar_senales_a_base(
        senales_tf,
        ctx.tf_to_base_idx,
        ctx.df_base.height,
    )
    if salida.tipo == "CUSTOM":
        if salidas_custom is None:
            raise ValueError("[CUSTOM] Falta la serie de salidas custom para preparar ejecucion.")
        salidas_base = proyectar_senales_a_base(
            salidas_custom,
            ctx.tf_to_base_idx,
            ctx.df_base.height,
        )
        return ctx.arrays_base, senales_base, salidas_base

    return ctx.arrays_base, senales_base, None


def _timeframe_ejecucion(
    *,
    ctx: ContextoCombinacion,
    timeframe: str,
    timeframe_base: str,
) -> str:
    if ctx.es_min_tf:
        return str(timeframe)
    return str(timeframe_base)


def _exit_velas_para_motor(salida: ExitConfig, ctx: ContextoCombinacion | None) -> int:
    velas = int(salida.velas)
    if salida.tipo != "BARS":
        return velas
    return velas * _barras_base_por_vela_tf(ctx)


def _barras_base_por_vela_tf(ctx: ContextoCombinacion | None) -> int:
    if ctx is None or ctx.es_min_tf:
        return 1

    starts = np.searchsorted(ctx.arrays_base.timestamps, ctx.arrays_tf.timestamps).astype(np.int64)
    ends = np.asarray(ctx.tf_to_base_idx, dtype=np.int64)
    if starts.shape != ends.shape:
        raise ValueError("[BARS] Plan temporal inconsistente: starts y ends difieren.")
    if starts.size == 0:
        raise ValueError("[BARS] No hay velas de estrategia para calcular la duracion.")

    widths = ends - starts + 1
    if (widths <= 0).any():
        raise ValueError("[BARS] Plan temporal con ventanas invertidas o vacias.")

    unique = np.unique(widths)
    if unique.size != 1:
        raise ValueError(
            "[BARS] El timeframe de estrategia no proyecta ventanas uniformes al timeframe base."
        )
    return int(unique[0])


def _sim_config(
    salida: ExitConfig,
    *,
    ctx: ContextoCombinacion | None = None,
) -> SimConfigMotor:
    return SimConfigMotor(
        saldo_inicial=cfg.SALDO_INICIAL,
        saldo_por_trade=cfg.SALDO_USADO_POR_TRADE,
        apalancamiento=cfg.APALANCAMIENTO,
        saldo_minimo=cfg.SALDO_MINIMO_OPERATIVO,
        comision_pct=cfg.COMISION_PCT,
        comision_lados=cfg.COMISION_LADOS,
        exit_type=salida.tipo,
        exit_sl_pct=salida.sl_pct,
        exit_tp_pct=salida.tp_pct,
        exit_velas=_exit_velas_para_motor(salida, ctx),
        exit_trail_act_pct=salida.trail_act_pct,
        exit_trail_dist_pct=salida.trail_dist_pct,
    )


def _salida_para_trial(salida: ExitConfig, trial: optuna.Trial) -> tuple[ExitConfig, dict]:
    if not salida.optimizar:
        return salida, {}

    if salida.tipo == "FIXED":
        sl = trial.suggest_float("exit_sl_pct", float(salida.sl_min), float(salida.sl_max), step=1.0)
        tp = trial.suggest_float("exit_tp_pct", float(salida.tp_min), float(salida.tp_max), step=1.0)
        return (
            ExitConfig(tipo="FIXED", sl_pct=sl, tp_pct=tp, velas=0, optimizar=True),
            {"exit_sl_pct": sl, "exit_tp_pct": tp},
        )

    if salida.tipo == "BARS":
        velas = trial.suggest_int("exit_velas", int(salida.velas_min), int(salida.velas_max))
        params = {"exit_velas": velas}
        if salida.sl_emergencia:
            sl = trial.suggest_float("exit_sl_pct", float(salida.sl_min), float(salida.sl_max), step=1.0)
            params["exit_sl_pct"] = sl
        else:
            sl = 0.0
        return (
            ExitConfig(
                tipo="BARS",
                sl_pct=sl,
                tp_pct=0.0,
                velas=velas,
                sl_emergencia=salida.sl_emergencia,
                optimizar=True,
            ),
            params,
        )

    if salida.tipo == "TRAILING":
        sl = trial.suggest_float("exit_sl_pct", float(salida.sl_min), float(salida.sl_max), step=1.0)
        act = trial.suggest_float(
            "exit_trail_act_pct",
            float(salida.trail_act_min),
            float(salida.trail_act_max),
            step=1.0,
        )
        dist = trial.suggest_float(
            "exit_trail_dist_pct",
            float(salida.trail_dist_min),
            float(salida.trail_dist_max),
            step=1.0,
        )
        act, dist = _normalizar_trailing(act, dist)
        return (
            ExitConfig(
                tipo="TRAILING",
                sl_pct=sl,
                tp_pct=0.0,
                velas=0,
                trail_act_pct=act,
                trail_dist_pct=dist,
                optimizar=True,
            ),
            {"exit_sl_pct": sl, "exit_trail_act_pct": act, "exit_trail_dist_pct": dist},
        )

    if salida.tipo == "CUSTOM":
        sl = trial.suggest_float("exit_sl_pct", float(salida.sl_min), float(salida.sl_max), step=1.0)
        return (
            ExitConfig(tipo="CUSTOM", sl_pct=sl, tp_pct=0.0, velas=0, optimizar=True),
            {"exit_sl_pct": sl},
        )

    raise ValueError(f"Salida no soportada para Optuna: {salida.tipo}")


def _params_para_monitor(parametros: dict, salida: ExitConfig) -> dict:
    params = dict(parametros)
    params.update(
        {
            "__exit_type": salida.tipo,
            "__exit_sl_pct": salida.sl_pct,
            "__exit_tp_pct": salida.tp_pct,
            "__exit_velas": salida.velas,
            "__exit_sl_emergencia": salida.sl_emergencia,
            "__exit_trail_act_pct": salida.trail_act_pct,
            "__exit_trail_dist_pct": salida.trail_dist_pct,
        }
    )
    return params


def _salidas_a_ejecutar():
    exit_type = cfg.EXIT_TYPE
    if exit_type in {"FIXED", "ALL"}:
        from SALIDAS import fijo

        yield ExitConfig(
            tipo="FIXED",
            sl_pct=float(fijo.EXIT_SL_PCT),
            tp_pct=float(fijo.EXIT_TP_PCT),
            velas=0,
            optimizar=bool(getattr(fijo, "OPTIMIZAR_SALIDAS", False)),
            sl_min=float(getattr(fijo, "EXIT_SL_MIN", 5)),
            sl_max=float(getattr(fijo, "EXIT_SL_MAX", 50)),
            tp_min=float(getattr(fijo, "EXIT_TP_MIN", 10)),
            tp_max=float(getattr(fijo, "EXIT_TP_MAX", 120)),
        )

    if exit_type in {"BARS", "ALL"}:
        from SALIDAS import velas

        sl_emergencia = bool(getattr(velas, "USAR_SL_EMERGENCIA", True))
        yield ExitConfig(
            tipo="BARS",
            sl_pct=float(velas.EXIT_SL_PCT) if sl_emergencia else 0.0,
            tp_pct=0.0,
            velas=int(velas.EXIT_VELAS),
            sl_emergencia=sl_emergencia,
            optimizar=bool(getattr(velas, "OPTIMIZAR_SALIDAS", False)),
            sl_min=float(getattr(velas, "EXIT_SL_MIN", 5)),
            sl_max=float(getattr(velas, "EXIT_SL_MAX", 50)),
            velas_min=int(getattr(velas, "EXIT_VELAS_MIN", 6)),
            velas_max=int(getattr(velas, "EXIT_VELAS_MAX", 240)),
        )

    if exit_type in {"TRAILING", "ALL"}:
        from SALIDAS import trailing

        act, dist = _normalizar_trailing(
            float(trailing.EXIT_TRAIL_ACT_PCT),
            float(trailing.EXIT_TRAIL_DIST_PCT),
        )
        yield ExitConfig(
            tipo="TRAILING",
            sl_pct=float(trailing.EXIT_SL_PCT),
            tp_pct=0.0,
            velas=0,
            trail_act_pct=act,
            trail_dist_pct=dist,
            optimizar=bool(getattr(trailing, "OPTIMIZAR_SALIDAS", False)),
            sl_min=float(getattr(trailing, "EXIT_SL_MIN", 5)),
            sl_max=float(getattr(trailing, "EXIT_SL_MAX", 50)),
            trail_act_min=float(getattr(trailing, "EXIT_TRAIL_ACT_MIN", 10)),
            trail_act_max=float(getattr(trailing, "EXIT_TRAIL_ACT_MAX", 80)),
            trail_dist_min=float(getattr(trailing, "EXIT_TRAIL_DIST_MIN", 2)),
            trail_dist_max=float(getattr(trailing, "EXIT_TRAIL_DIST_MAX", 30)),
        )

    if exit_type in {"CUSTOM", "ALL"}:
        from SALIDAS import personalizada

        yield ExitConfig(
            tipo="CUSTOM",
            sl_pct=float(personalizada.EXIT_SL_PCT),
            tp_pct=0.0,
            velas=0,
            optimizar=bool(getattr(personalizada, "OPTIMIZAR_SALIDAS", False)),
            sl_min=float(getattr(personalizada, "EXIT_SL_MIN", 5)),
            sl_max=float(getattr(personalizada, "EXIT_SL_MAX", 50)),
        )


def _normalizar_trailing(act_pct: float, dist_pct: float) -> tuple[float, float]:
    act = abs(float(act_pct)) if float(act_pct) != 0.0 else 0.5
    dist = abs(float(dist_pct)) if float(dist_pct) != 0.0 else 0.25
    if dist >= act:
        act, dist = dist, act
    if act == dist:
        act += 0.5
    return float(act), float(dist)


def _verificar_mejor_de_study(study: optuna.Study, resultados: list[TrialResultado]) -> None:
    mejor_local = max(resultados, key=lambda t: t.score)
    if int(study.best_trial.number) != int(mejor_local.numero):
        raise ValueError("[OPTUNA] El mejor trial de Optuna no coincide con el conservado.")


def _marcar_pareto(study: optuna.Study, resultados: list[TrialResultado]) -> None:
    """Marca con `en_pareto=True` los trials del frente de Pareto (multiobjetivo)."""
    numeros = {int(t.number) for t in study.best_trials}
    if not numeros:
        raise ValueError("[OPTUNA] El frente de Pareto está vacío.")
    por_numero = {int(r.numero): r for r in resultados}
    marcados = 0
    for n in numeros:
        r = por_numero.get(n)
        if r is not None:
            r.en_pareto = True
            marcados += 1
    if marcados == 0:
        raise ValueError("[OPTUNA] Ningún trial del frente de Pareto está en los resultados.")


def _seleccionar_mejor(trials: list[TrialResultado]) -> TrialResultado:
    """Selecciona la configuración final.

    - Escalar (por defecto): el trial de mayor score.
    - Multiobjetivo: la MESETA del frente de Pareto (región de parámetros más
      estable por PSR), no el pico de score.
    """
    if bool(getattr(cfg, "OPTUNA_MULTIOBJETIVO", False)):
        candidatos = [t for t in trials if t.en_pareto] or list(trials)
        return seleccionar_meseta(
            candidatos,
            trials,
            valor=lambda t: float(t.metricas.get("psr", t.score)),
            parametros=lambda t: t.parametros,
            k=int(getattr(cfg, "MESETA_VECINOS", 7)),
        )
    return max(trials, key=lambda t: t.score)


def _normalizar_jobs(valor: int) -> int:
    cpus = os.cpu_count() or 1
    if valor == -1:
        return cpus
    if valor == -2:
        return max(1, cpus - 1)
    if valor < -2:
        return max(1, cpus + valor + 1)
    return max(1, int(valor))


def _normalizar_jobs_perturbaciones(n_jobs: int, *, max_jobs: int | None = None) -> int:
    if max_jobs is None:
        max_jobs = int(getattr(cfg, "PERTURBACIONES_MAX_JOBS", 4))
    return max(1, min(int(n_jobs), int(max_jobs)))


def _fecha_config(valor, nombre: str) -> date:
    if isinstance(valor, date):
        return valor
    try:
        return date.fromisoformat(str(valor))
    except ValueError as exc:
        raise ValueError(f"[CONFIG] {nombre} debe tener formato YYYY-MM-DD: {valor!r}.") from exc


def _es_mercado_24_7(activo: str) -> bool:
    return bool(getattr(cfg, "MERCADO_24_7", {}).get(activo, True))


def _como_lista(valor):
    return valor if isinstance(valor, list) else [valor]


def _columnas_requeridas(estrategias) -> set[str]:
    columnas: set[str] = set()
    for estrategia in estrategias:
        columnas.update(getattr(estrategia, "COLUMNAS_REQUERIDAS", set()))
    return columnas

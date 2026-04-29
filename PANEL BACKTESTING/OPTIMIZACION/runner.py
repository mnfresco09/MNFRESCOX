from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from datetime import date
from threading import Lock
from typing import Any, Optional

import numpy as np
import optuna

from CONFIGURACION import config as cfg
from CONFIGURACION.validador_config import validar as validar_config
from DATOS.cargador import cargar
from DATOS.perturbaciones import (
    ConfiguracionPerturbaciones,
    aplicar_perturbaciones,
    seed_para_trial,
)
from DATOS.resampleo import inferir_timeframe, resamplear
from DATOS.validador import validar as validar_datos
from MOTOR import simular_full, simular_metricas
from NUCLEO import integridad, paridad_riesgo
from NUCLEO.base_estrategia import CacheIndicadores
from NUCLEO.contexto import ArraysMotor, ContextoCombinacion, SimConfigMotor, crear_contexto
from NUCLEO.proyeccion import proyectar_senales_a_base
from NUCLEO.registro import cargar_estrategias, obtener_estrategia
from OPTIMIZACION.metricas import calcular_metricas
from OPTIMIZACION.puntuacion import calcular_score
from OPTIMIZACION.samplers import crear_sampler
from REPORTES.excel import MAX_DETALLES_EXCEL, generar_excel
from REPORTES.html import generar_htmls
from REPORTES.informe import generar_informe
from REPORTES.persistencia import guardar_optimizacion, preparar_resultados_combinacion
from REPORTES.rich import (
    MonitorOptimizacion,
    mostrar_aviso_perturbaciones,
    mostrar_fin_backtest,
    mostrar_huella_datos,
    mostrar_inicio_motor,
    mostrar_resumen_run,
)
from SALIDAS import paridad as paridad_salida


@dataclass(frozen=True)
class ExitConfig:
    tipo: str
    sl_pct: float
    tp_pct: float
    velas: int
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


def main() -> None:
    mostrar_inicio_motor()
    validar_config(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fecha_inicio = _fecha_config(cfg.FECHA_INICIO, "FECHA_INICIO")
    fecha_fin = _fecha_config(cfg.FECHA_FIN, "FECHA_FIN")

    registro = cargar_estrategias()
    estrategias = obtener_estrategia(registro, cfg.ESTRATEGIA_ID)
    columnas_requeridas = _columnas_requeridas(estrategias)
    activos = _como_lista(cfg.ACTIVOS)
    timeframes = _como_lista(cfg.TIMEFRAMES)
    salidas = list(_salidas_a_ejecutar())
    n_jobs = _normalizar_jobs(cfg.N_JOBS)
    perturbaciones = ConfiguracionPerturbaciones.desde_config(cfg)
    if perturbaciones.activa and n_jobs != 1:
        mostrar_aviso_perturbaciones(n_jobs_original=n_jobs, n_jobs_final=1)
        n_jobs = 1
    combinaciones_esperadas = len(activos) * len(timeframes) * len(estrategias) * len(salidas)
    combinaciones_ejecutadas: set[tuple[str, str, int, str]] = set()

    total_runs = 0
    for activo in activos:
        permitir_huecos = not _es_mercado_24_7(activo)
        df_base = cargar(activo, cfg)
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

        for timeframe in timeframes:
            df_tf = resamplear(df_base, timeframe)
            validar_datos(
                df_tf,
                f"{activo} {timeframe}",
                columnas_requeridas,
                timeframe=timeframe,
                permitir_huecos=permitir_huecos,
            )
            integridad.verificar_resampleo(df_base, df_tf, timeframe)
            huella_tf = integridad.huella_dataframe(f"{activo} {timeframe}", df_tf)
            mostrar_huella_datos(huella_tf)
            ctx = crear_contexto(df_base=df_base, df_tf=df_tf, timeframe=timeframe)

            for estrategia in estrategias:
                # Sin perturbaciones los buffers son estables para toda la
                # combinacion. Con perturbaciones se clona y vincula por trial.
                if not perturbaciones_activo.activa:
                    cache_indicadores = CacheIndicadores()
                    estrategia.bind(ctx.arrays_tf, cache_indicadores)
                try:
                    for salida in salidas:
                        clave = (str(activo), str(timeframe), int(estrategia.ID), str(salida.tipo))
                        if clave in combinaciones_ejecutadas:
                            raise ValueError(f"[RUN] Combinacion duplicada detectada: {clave}.")
                        combinaciones_ejecutadas.add(clave)

                        try:
                            resultados_base = preparar_resultados_combinacion(
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
                                timeframe_base=timeframe_base,
                                n_jobs=n_jobs,
                                fecha_inicio=fecha_inicio,
                                fecha_fin=fecha_fin,
                                perturbaciones=perturbaciones_activo,
                                resultados_base=resultados_base,
                            )
                        except Exception as exc:
                            contexto = (
                                f"{activo}/{timeframe}/{estrategia.NOMBRE}/{salida.tipo}"
                            )
                            raise RuntimeError(f"[RUN] Fallo en {contexto}: {exc}") from exc

                        mejor = max(trials, key=lambda t: t.score)
                        if mejor.replay is None:
                            raise RuntimeError("[RUN] El mejor trial no tiene replay materializado.")
                        run_dir = guardar_optimizacion(
                            carpeta_resultados=cfg.CARPETA_RESULTADOS,
                            activo=activo,
                            timeframe=timeframe,
                            estrategia_id=estrategia.ID,
                            estrategia_nombre=estrategia.NOMBRE,
                            salida=salida,
                            trials=trials,
                            mejor=mejor,
                            huella_base=huella_base,
                            huella_timeframe=huella_tf,
                            conteo_senales_mejor=mejor.conteo_senales,
                            conteo_salidas_mejor=mejor.conteo_salidas,
                            max_archivos=cfg.MAX_ARCHIVOS,
                        )

                        excel_path = None
                        if cfg.USAR_EXCEL:
                            excel_path = generar_excel(run_dir, trials, mejor)

                        html_paths = generar_htmls(
                            run_dir=run_dir,
                            df=ctx.df_tf,
                            df_indicadores=ctx.df_tf,
                            trials=trials,
                            estrategia=estrategia,
                            max_plots=cfg.MAX_PLOTS,
                            grafica_rango=cfg.GRAFICA_RANGO,
                            grafica_desde=cfg.GRAFICA_DESDE,
                            grafica_hasta=cfg.GRAFICA_HASTA,
                        )
                        informe_path = generar_informe(
                            run_dir=run_dir,
                            trials=trials,
                            estrategia=estrategia,
                            activo=activo,
                            timeframe=timeframe,
                            salida_tipo=salida.tipo,
                        )

                        mostrar_resumen_run(
                            mejor=mejor,
                            total_trials=cfg.N_TRIALS,
                            run_dir=run_dir,
                            excel_path=excel_path,
                            html_paths=html_paths,
                            informe_path=informe_path,
                        )
                        total_runs += 1
                        del trials, mejor, run_dir, excel_path, html_paths, informe_path
                        gc.collect()
                finally:
                    estrategia.desvincular()

            del df_tf, ctx
            gc.collect()

        del df_base
        gc.collect()

    if total_runs != combinaciones_esperadas:
        raise ValueError(
            f"[RUN] Runs ejecutados incorrectos: {total_runs} != {combinaciones_esperadas}."
        )

    mostrar_fin_backtest(total_runs)


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
        paridad_trial, params_paridad = paridad_riesgo.parametros_para_trial(
            trial,
            salida_trial.tipo,
            activa=_paridad_activa(),
            optimizar=bool(getattr(cfg, "OPTIMIZAR_PARIDAD_RIESGO", True)),
        )
        parametros = {**params_estrategia, **params_salida, **params_paridad}

        try:
            senales_tf = estrategia_trial.generar_señales(ctx_trial.df_tf, params_estrategia)
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
            risk_vol_exec = _preparar_volatilidad_paridad(
                ctx=ctx_trial,
                params=paridad_trial,
            )
            timeframe_ejecucion = _timeframe_ejecucion(
                ctx=ctx_trial,
                timeframe=timeframe,
                timeframe_base=timeframe_base,
            )

            metricas_obj = simular_metricas(
                arrays_exec,
                senales_exec,
                sim_cfg=_sim_config(salida_trial, paridad_trial, risk_vol_exec),
                salidas_custom=salidas_exec,
            )
            metricas = calcular_metricas(metricas_obj, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
            score = calcular_score(metricas)
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

    _verificar_mejor_de_study(study, resultados)

    # Replay determinista de los top-N para alimentar reportes (CSV / Excel / HTML).
    # El resto de trials sigue sin trades en memoria.
    n_replay = max(MAX_DETALLES_EXCEL, int(cfg.MAX_PLOTS), 1)
    top = sorted(resultados, key=lambda t: t.score, reverse=True)[:n_replay]
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
    paridad_replay = paridad_riesgo.params_desde_dict(
        trial_res.parametros,
        salida.tipo,
        activa=_paridad_activa(),
    )
    try:
        senales_tf = estrategia_trial.generar_señales(ctx_trial.df_tf, params_estrategia)
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
        risk_vol_exec = _preparar_volatilidad_paridad(
            ctx=ctx_trial,
            params=paridad_replay,
        )
        sim_result = simular_full(
            arrays_exec,
            senales_exec,
            sim_cfg=_sim_config(salida, paridad_replay, risk_vol_exec),
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
    df_tf = resamplear(df_base, timeframe)
    return crear_contexto(df_base=df_base, df_tf=df_tf, timeframe=timeframe)


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


def _paridad_activa() -> bool:
    return bool(getattr(cfg, "USAR_PARIDAD_RIESGO", False))


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


def _preparar_volatilidad_paridad(
    *,
    ctx: ContextoCombinacion,
    params: paridad_riesgo.ParametrosParidadRiesgo,
) -> np.ndarray | None:
    if not params.activa:
        return None

    vol_tf = paridad_riesgo.calcular_volatilidad_ewma(ctx.df_tf, params.vol_halflife)
    if ctx.es_min_tf:
        return _array_f64_contiguo(vol_tf)
    vol_base = paridad_riesgo.proyectar_volatilidad_a_base(
        vol_tf,
        ctx.tf_to_base_idx,
        ctx.df_base.height,
    )
    return _array_f64_contiguo(vol_base)


def _array_f64_contiguo(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def _timeframe_ejecucion(
    *,
    ctx: ContextoCombinacion,
    timeframe: str,
    timeframe_base: str,
) -> str:
    if ctx.es_min_tf:
        return str(timeframe)
    return str(timeframe_base)


def _sim_config(
    salida: ExitConfig,
    paridad: paridad_riesgo.ParametrosParidadRiesgo | None = None,
    risk_vol_ewma: np.ndarray | None = None,
) -> SimConfigMotor:
    paridad = paridad or paridad_riesgo.ParametrosParidadRiesgo(activa=False)
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
        exit_velas=salida.velas,
        exit_trail_act_pct=salida.trail_act_pct,
        exit_trail_dist_pct=salida.trail_dist_pct,
        paridad_riesgo=bool(paridad.activa),
        paridad_riesgo_max_pct=float(paridad.riesgo_max_pct),
        paridad_apalancamiento_min=float(paridad_salida.PARIDAD_APALANCAMIENTO_MIN),
        paridad_apalancamiento_max=float(paridad_salida.PARIDAD_APALANCAMIENTO_MAX),
        risk_vol_ewma=risk_vol_ewma,
        exit_sl_ewma_mult=float(paridad.sl_ewma_mult),
        exit_tp_ewma_mult=float(paridad.tp_ewma_mult),
        exit_trail_act_ewma_mult=float(paridad.trail_act_ewma_mult),
        exit_trail_dist_ewma_mult=float(paridad.trail_dist_ewma_mult),
        paridad_skip_bajo_min=bool(paridad_salida.SKIP_SI_APALANCAMIENTO_MENOR_MIN),
    )


def _salida_para_trial(salida: ExitConfig, trial: optuna.Trial) -> tuple[ExitConfig, dict]:
    if _paridad_activa():
        if salida.tipo == "BARS" and salida.optimizar:
            velas = trial.suggest_int("exit_velas", int(salida.velas_min), int(salida.velas_max))
            return (
                ExitConfig(
                    tipo="BARS",
                    sl_pct=salida.sl_pct,
                    tp_pct=0.0,
                    velas=velas,
                    optimizar=True,
                ),
                {"exit_velas": velas},
            )
        return salida, {}

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
        sl = trial.suggest_float("exit_sl_pct", float(salida.sl_min), float(salida.sl_max), step=1.0)
        velas = trial.suggest_int("exit_velas", int(salida.velas_min), int(salida.velas_max))
        return (
            ExitConfig(tipo="BARS", sl_pct=sl, tp_pct=0.0, velas=velas, optimizar=True),
            {"exit_sl_pct": sl, "exit_velas": velas},
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
            "__exit_trail_act_pct": salida.trail_act_pct,
            "__exit_trail_dist_pct": salida.trail_dist_pct,
            "__paridad_riesgo": _paridad_activa(),
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

        yield ExitConfig(
            tipo="BARS",
            sl_pct=float(velas.EXIT_SL_PCT),
            tp_pct=0.0,
            velas=int(velas.EXIT_VELAS),
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


def _normalizar_jobs(valor: int) -> int:
    cpus = os.cpu_count() or 1
    if valor == -1:
        return cpus
    if valor == -2:
        return max(1, cpus - 1)
    if valor < -2:
        return max(1, cpus + valor + 1)
    return max(1, int(valor))


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

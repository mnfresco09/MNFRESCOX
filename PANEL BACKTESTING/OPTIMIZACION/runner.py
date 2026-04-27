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
from DATOS.resampleo import inferir_timeframe, resamplear
from DATOS.validador import validar as validar_datos
from MOTOR import simular_full, simular_metricas
from NUCLEO import integridad
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
from REPORTES.persistencia import guardar_optimizacion
from REPORTES.rich import MonitorOptimizacion


@dataclass(frozen=True)
class ExitConfig:
    tipo: str
    sl_pct: float
    tp_pct: float
    velas: int
    optimizar: bool = False
    sl_min: float | None = None
    sl_max: float | None = None
    tp_min: float | None = None
    tp_max: float | None = None
    velas_min: int | None = None
    velas_max: int | None = None


@dataclass
class ReplayTrial:
    """Datos del replay de un trial top: trades en columnas numpy + equity."""
    metricas_obj: object  # struct Metricas Rust (para integridad)
    trades: dict[str, np.ndarray]
    equity_curve: np.ndarray


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
    replay: Optional[ReplayTrial] = None


def main() -> None:
    print("[RUN] Backtest motor v2 — buffers numpy, métricas en Rust, GIL liberado")
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
        _imprimir_huella(huella_base)

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
            _imprimir_huella(huella_tf)
            ctx = crear_contexto(df_base=df_base, df_tf=df_tf)

            for estrategia in estrategias:
                # bind cubre TODA la combinación: optimización + reportes
                # (los reportes HTML llaman a indicadores_para_grafica que
                # también necesita los buffers).
                cache_indicadores = CacheIndicadores()
                estrategia.bind(ctx.arrays_tf, cache_indicadores)
                try:
                    for salida in salidas:
                        clave = (str(activo), str(timeframe), int(estrategia.ID), str(salida.tipo))
                        if clave in combinaciones_ejecutadas:
                            raise ValueError(f"[RUN] Combinacion duplicada detectada: {clave}.")
                        combinaciones_ejecutadas.add(clave)

                        try:
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
                            df_exec=_df_para_reporte(ctx, salida),
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

                        _imprimir_resumen_run(mejor, run_dir, excel_path, html_paths, informe_path)
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

    print(f"[OK] Backtest completado. Combinaciones verificadas: {total_runs}")


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
) -> list[TrialResultado]:
    sampler = crear_sampler(cfg.OPTUNA_SAMPLER, cfg.OPTUNA_SEED, cfg.N_TRIALS)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    resultados: list[TrialResultado] = []
    lock = Lock()

    # Estrategia ya vinculada por main() a los buffers TF y al cache.

    def objective(trial: optuna.Trial) -> float:
        params_estrategia = estrategia.espacio_busqueda(trial)
        salida_trial, params_salida = _salida_para_trial(salida_base, trial)
        parametros = {**params_estrategia, **params_salida}

        senales_tf = estrategia.generar_señales(ctx.df_tf, params_estrategia)
        conteo = integridad.verificar_senales(ctx.df_tf, senales_tf)
        salidas_custom = (
            estrategia.generar_salidas(ctx.df_tf, params_estrategia)
            if salida_trial.tipo == "CUSTOM"
            else None
        )
        conteo_salidas = (
            integridad.verificar_salidas_custom(ctx.df_tf, salidas_custom)
            if salidas_custom is not None
            else None
        )

        arrays_exec, senales_exec, salidas_exec = _preparar_ejecucion(
            ctx=ctx,
            salida=salida_trial,
            senales_tf=senales_tf,
            salidas_custom=salidas_custom,
        )
        timeframe_ejecucion = _timeframe_ejecucion(
            ctx=ctx,
            salida=salida_trial,
            timeframe=timeframe,
            timeframe_base=timeframe_base,
        )

        metricas_obj = simular_metricas(
            arrays_exec,
            senales_exec,
            sim_cfg=_sim_config(salida_trial),
            salidas_custom=salidas_exec,
        )
        metricas = calcular_metricas(metricas_obj, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
        score = calcular_score(metricas)
        trial.set_user_attr("metricas", metricas)
        trial.set_user_attr("conteo_senales", conteo)
        trial.set_user_attr("conteo_salidas", conteo_salidas)

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

    with MonitorOptimizacion(
        activo=activo,
        timeframe=timeframe,
        estrategia=estrategia.NOMBRE,
        salida=salida_base.tipo,
        total_trials=cfg.N_TRIALS,
        sampler=cfg.OPTUNA_SAMPLER,
        n_jobs=n_jobs,
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
        )

    return resultados


def _replay_trial(
    *,
    trial_res: TrialResultado,
    ctx: ContextoCombinacion,
    estrategia,
    fecha_inicio: date,
    fecha_fin: date,
) -> None:
    """Reconstruye trades + equity_curve para un trial que va a generar reporte.

    Verifica que los escalares clave (saldo_final, total_trades) coinciden
    con los registrados durante la fase Optuna — garantía de determinismo.
    """
    salida = trial_res.salida
    params_estrategia = {
        k: v for k, v in trial_res.parametros.items()
        if not k.startswith("exit_")
    }
    senales_tf = estrategia.generar_señales(ctx.df_tf, params_estrategia)
    salidas_custom = (
        estrategia.generar_salidas(ctx.df_tf, params_estrategia)
        if salida.tipo == "CUSTOM"
        else None
    )
    arrays_exec, senales_exec, salidas_exec = _preparar_ejecucion(
        ctx=ctx,
        salida=salida,
        senales_tf=senales_tf,
        salidas_custom=salidas_custom,
    )
    sim_result = simular_full(
        arrays_exec,
        senales_exec,
        sim_cfg=_sim_config(salida),
        salidas_custom=salidas_exec,
    )
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
    )


def _preparar_ejecucion(
    *,
    ctx: ContextoCombinacion,
    salida: ExitConfig,
    senales_tf,
    salidas_custom,
) -> tuple[ArraysMotor, Any, Any]:
    if salida.tipo == "CUSTOM" or ctx.es_min_tf:
        return ctx.arrays_tf, senales_tf, salidas_custom

    senales_base = proyectar_senales_a_base(
        senales_tf,
        ctx.tf_to_base_idx,
        ctx.df_base.height,
    )
    return ctx.arrays_base, senales_base, None


def _df_para_reporte(ctx: ContextoCombinacion, salida: ExitConfig):
    if salida.tipo == "CUSTOM" or ctx.es_min_tf:
        return ctx.df_tf
    return ctx.df_base


def _timeframe_ejecucion(
    *,
    ctx: ContextoCombinacion,
    salida: ExitConfig,
    timeframe: str,
    timeframe_base: str,
) -> str:
    if salida.tipo == "CUSTOM" or ctx.es_min_tf:
        return str(timeframe)
    return str(timeframe_base)


def _sim_config(salida: ExitConfig) -> SimConfigMotor:
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
        sl = trial.suggest_float("exit_sl_pct", float(salida.sl_min), float(salida.sl_max), step=1.0)
        velas = trial.suggest_int("exit_velas", int(salida.velas_min), int(salida.velas_max))
        return (
            ExitConfig(tipo="BARS", sl_pct=sl, tp_pct=0.0, velas=velas, optimizar=True),
            {"exit_sl_pct": sl, "exit_velas": velas},
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


def _imprimir_huella(huella: integridad.HuellaDatos) -> None:
    print(
        "[DATOS] "
        f"{huella.etapa}: filas={huella.filas:,} "
        f"rango={huella.ts_inicio} -> {huella.ts_fin}"
    )


def _imprimir_resumen_run(mejor: TrialResultado, run_dir, excel_path, html_paths, informe_path) -> None:
    metricas = mejor.metricas
    excel_txt = str(excel_path) if excel_path is not None else "desactivado"
    print(
        "[RESULTADO] "
        f"{mejor.activo} {mejor.timeframe} {mejor.estrategia_nombre} {mejor.salida.tipo}: "
        f"trials={cfg.N_TRIALS:,} "
        f"mejor_trial={mejor.numero} "
        f"score={mejor.score:.6f} "
        f"roi={metricas['roi_total']:.2%} "
        f"expectancy={metricas['expectancy']:.2%} "
        f"win_rate={metricas['win_rate']:.2%} "
        f"profit_factor={metricas['profit_factor']:.4f} "
        f"sharpe={metricas['sharpe_ratio']:.4f} "
        f"max_dd={metricas['max_drawdown']:.2%} "
        f"trades_dia={metricas['trades_por_dia']:.4f} "
        f"trades={metricas['total_trades']:,} "
        f"excel={excel_txt} "
        f"htmls={len(html_paths)} "
        f"informe={informe_path} "
        f"dir={run_dir}"
    )

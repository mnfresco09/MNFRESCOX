from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from threading import Lock

import optuna

from CONFIGURACION import config as cfg
from CONFIGURACION.validador_config import validar as validar_config
from DATOS.cargador import cargar
from DATOS.resampleo import resamplear
from DATOS.validador import validar as validar_datos
from MOTOR import simular_dataframe
from NUCLEO import integridad
from NUCLEO.registro import cargar_estrategias, obtener_estrategia
from OPTIMIZACION.metricas import calcular_metricas
from OPTIMIZACION.puntuacion import calcular_score
from OPTIMIZACION.samplers import crear_sampler
from REPORTES.excel import generar_excel
from REPORTES.html import generar_htmls
from REPORTES.persistencia import guardar_optimizacion
from REPORTES.terminal import MonitorOptimizacion


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


@dataclass(frozen=True)
class TrialResultado:
    numero: int
    activo: str
    timeframe: str
    estrategia_id: int
    estrategia_nombre: str
    salida: ExitConfig
    parametros: dict
    resultado: object
    score: float
    metricas: dict
    conteo_senales: dict[int, int]
    conteo_salidas: dict[int, int] | None = None


def main() -> None:
    print("[RUN] Fase 7: salidas CUSTOM + guia de estrategias + auditoria reforzada")
    validar_config(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        validar_datos(
            df_base,
            activo,
            columnas_requeridas,
            timeframe="1m",
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

            for estrategia in estrategias:
                for salida in salidas:
                    clave = (str(activo), str(timeframe), int(estrategia.ID), str(salida.tipo))
                    if clave in combinaciones_ejecutadas:
                        raise ValueError(f"[FASE 7] Combinacion duplicada detectada: {clave}.")
                    combinaciones_ejecutadas.add(clave)

                    try:
                        trials = _optimizar_combinacion(
                            activo=activo,
                            timeframe=timeframe,
                            estrategia=estrategia,
                            salida_base=salida,
                            df_tf=df_tf,
                            huella_base=huella_base,
                            huella_tf=huella_tf,
                            n_jobs=n_jobs,
                        )
                    except Exception as exc:
                        contexto = (
                            f"{activo}/{timeframe}/{estrategia.NOMBRE}/"
                            f"{salida.tipo}"
                        )
                        raise RuntimeError(f"[FASE 7] Fallo en {contexto}: {exc}") from exc
                    mejor = max(trials, key=lambda t: t.score)
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
                        df=df_tf,
                        trials=trials,
                        max_plots=cfg.MAX_PLOTS,
                        grafica_rango=cfg.GRAFICA_RANGO,
                        grafica_desde=cfg.GRAFICA_DESDE,
                        grafica_hasta=cfg.GRAFICA_HASTA,
                    )

                    _imprimir_resumen_run(mejor, run_dir, excel_path, html_paths)
                    total_runs += 1
                    del trials, mejor, run_dir, excel_path, html_paths
                    gc.collect()

            del df_tf
            gc.collect()

        del df_base
        gc.collect()

    if total_runs != combinaciones_esperadas:
        raise ValueError(
            f"[FASE 7] Runs ejecutados incorrectos: {total_runs} != {combinaciones_esperadas}."
        )

    print(f"[OK] Fase 7 completada. Combinaciones verificadas: {total_runs}")


def _optimizar_combinacion(
    *,
    activo: str,
    timeframe: str,
    estrategia,
    salida_base: ExitConfig,
    df_tf,
    huella_base,
    huella_tf,
    n_jobs: int,
) -> list[TrialResultado]:
    sampler = crear_sampler(cfg.OPTUNA_SAMPLER, cfg.OPTUNA_SEED, cfg.N_TRIALS)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    resultados: list[TrialResultado] = []
    lock = Lock()

    def objective(trial: optuna.Trial) -> float:
        params_estrategia = estrategia.espacio_busqueda(trial)
        salida_trial, params_salida = _salida_para_trial(salida_base, trial)
        parametros = {**params_estrategia, **params_salida}

        senales = getattr(estrategia, "generar_señales")(df_tf, params_estrategia)
        conteo = integridad.verificar_senales(df_tf, senales)
        salidas_custom = _generar_salidas_custom(
            estrategia=estrategia,
            df_tf=df_tf,
            params_estrategia=params_estrategia,
            salida=salida_trial,
        )
        conteo_salidas = (
            integridad.verificar_salidas_custom(df_tf, salidas_custom)
            if salidas_custom is not None
            else None
        )
        resultado = simular_dataframe(
            df_tf,
            senales,
            saldo_inicial=cfg.SALDO_INICIAL,
            saldo_por_trade=cfg.SALDO_USADO_POR_TRADE,
            apalancamiento=cfg.APALANCAMIENTO,
            saldo_minimo=cfg.SALDO_MINIMO_OPERATIVO,
            comision_pct=cfg.COMISION_PCT,
            comision_lados=cfg.COMISION_LADOS,
            exit_type=salida_trial.tipo,
            exit_sl_pct=salida_trial.sl_pct,
            exit_tp_pct=salida_trial.tp_pct,
            exit_velas=salida_trial.velas,
            salidas_custom=salidas_custom,
        )
        integridad.verificar_resultado(df_tf, senales, resultado, salidas_custom)
        metricas = calcular_metricas(resultado)
        score = calcular_score(resultado)
        trial.set_user_attr("metricas", metricas)
        trial.set_user_attr("conteo_senales", conteo)
        trial.set_user_attr("conteo_salidas", conteo_salidas)

        trial_resultado = TrialResultado(
            numero=trial.number,
            activo=activo,
            timeframe=timeframe,
            estrategia_id=int(estrategia.ID),
            estrategia_nombre=estrategia.NOMBRE,
            salida=salida_trial,
            parametros=parametros,
            resultado=resultado,
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
            params=parametros,
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
    return resultados


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


def _generar_salidas_custom(
    *,
    estrategia,
    df_tf,
    params_estrategia: dict,
    salida: ExitConfig,
):
    if salida.tipo != "CUSTOM":
        return None
    return getattr(estrategia, "generar_salidas")(df_tf, params_estrategia)


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


def _imprimir_resumen_run(mejor: TrialResultado, run_dir, excel_path, html_paths) -> None:
    metricas = mejor.metricas
    excel_txt = str(excel_path) if excel_path is not None else "desactivado"
    print(
        "[RESULTADO] "
        f"{mejor.activo} {mejor.timeframe} {mejor.estrategia_nombre} {mejor.salida.tipo}: "
        f"trials={cfg.N_TRIALS:,} "
        f"mejor_trial={mejor.numero} "
        f"score={mejor.score:.6f} "
        f"roi={metricas['roi_total']:.2%} "
        f"win_rate={metricas['win_rate']:.2%} "
        f"profit_factor={metricas['profit_factor']:.4f} "
        f"sharpe={metricas['sharpe_ratio']:.4f} "
        f"max_dd={metricas['max_drawdown']:.2%} "
        f"trades={metricas['total_trades']:,} "
        f"excel={excel_txt} "
        f"htmls={len(html_paths)} "
        f"dir={run_dir}"
    )

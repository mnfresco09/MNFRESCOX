"""
debug_timing.py — Perfil de tiempos por etapa del pipeline.
Ejecuta UNA sola combinacion con N_TRIALS reducido para medir rapidamente.
Uso: python3 debug_timing.py
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

RAIZ = Path(__file__).resolve().parent
PANEL_DIR = RAIZ / "PANEL BACKTESTING"
sys.path.insert(0, str(PANEL_DIR))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from CONFIGURACION import config as cfg
from CONFIGURACION.validador_config import validar as validar_config
from DATOS.cargador import cargar
from DATOS.resampleo import inferir_timeframe, resamplear
from DATOS.validador import validar as validar_datos
from MOTOR import simular_dataframe
from NUCLEO import integridad
from NUCLEO.integridad import cachear_df
from NUCLEO.registro import cargar_estrategias, obtener_estrategia
from OPTIMIZACION.metricas import calcular_metricas
from OPTIMIZACION.puntuacion import calcular_score
from OPTIMIZACION.samplers import crear_sampler
from OPTIMIZACION.runner import (
    ExitConfig, _salida_para_trial, _generar_salidas_custom,
    _normalizar_jobs, _es_mercado_24_7, _como_lista, _columnas_requeridas,
    _salidas_a_ejecutar,
)

N_TRIALS_DEBUG = 8
ACTIVO_DEBUG   = _como_lista(cfg.ACTIVOS)[0]
TF_DEBUG       = _como_lista(cfg.TIMEFRAMES)[0]

_log: list[tuple[str, float]] = []

@contextmanager
def crono(etapa: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    _log.append((etapa, dt))
    print(f"  {dt:7.3f}s  {etapa}")

def _banner(titulo: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {titulo}")
    print(f"{'─'*60}")


def main():
    _banner("DEBUG TIMING — una combinación, N_TRIALS=" + str(N_TRIALS_DEBUG))

    with crono("validar_config"):
        validar_config(cfg)

    with crono("cargar_estrategias"):
        registro = cargar_estrategias()

    with crono("obtener_estrategia"):
        estrategias = obtener_estrategia(registro, cfg.ESTRATEGIA_ID)

    estrategia = estrategias[0]
    print(f"\n  Estrategia: {estrategia.NOMBRE}  |  Activo: {ACTIVO_DEBUG}  |  TF: {TF_DEBUG}")

    columnas_requeridas = _columnas_requeridas(estrategias)
    permitir_huecos = not _es_mercado_24_7(ACTIVO_DEBUG)

    with crono("cargar datos base (1m)"):
        df_base = cargar(ACTIVO_DEBUG, cfg)

    timeframe_base = inferir_timeframe(df_base)

    with crono("validar datos base"):
        validar_datos(df_base, ACTIVO_DEBUG, columnas_requeridas,
                      timeframe=timeframe_base, permitir_huecos=permitir_huecos)

    with crono("huella_dataframe base"):
        integridad.huella_dataframe(f"{ACTIVO_DEBUG} carga", df_base)

    with crono(f"resamplear a {TF_DEBUG}"):
        df_tf = resamplear(df_base, TF_DEBUG)

    with crono("validar datos resampleados"):
        validar_datos(df_tf, f"{ACTIVO_DEBUG} {TF_DEBUG}", columnas_requeridas,
                      timeframe=TF_DEBUG, permitir_huecos=permitir_huecos)

    with crono("verificar_resampleo"):
        integridad.verificar_resampleo(df_base, df_tf, TF_DEBUG)

    with crono("huella_dataframe tf"):
        integridad.huella_dataframe(f"{ACTIVO_DEBUG} {TF_DEBUG}", df_tf)

    with crono("cachear_df (precálculo arrays)"):
        cache = cachear_df(df_tf)

    print(f"\n  df_base: {df_base.height:,} filas  |  df_tf: {df_tf.height:,} filas")

    salida_base = next(_salidas_a_ejecutar())
    print(f"  Salida: {salida_base.tipo}\n")

    # ── Un trial representativo ───────────────────────────────────────────────
    _banner("Desglose interno de 1 trial")

    _trial_dummy = optuna.create_study().ask()

    with crono("espacio_busqueda"):
        params_estrategia = estrategia.espacio_busqueda(_trial_dummy)

    salida_trial, _ = _salida_para_trial(salida_base, _trial_dummy)

    with crono("generar_señales"):
        senales = getattr(estrategia, "generar_señales")(df_tf, params_estrategia)

    with crono("verificar_senales"):
        integridad.verificar_senales(df_tf, senales)

    with crono("simular_dataframe (Rust)"):
        resultado = simular_dataframe(
            df_tf, senales,
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
            salidas_custom=None,
        )

    print(f"    → {len(list(resultado.trades))} trades generados")

    with crono("verificar_resultado (con cache)"):
        integridad.verificar_resultado(df_tf, senales, resultado, None, _cache=cache)

    with crono("calcular_metricas"):
        calcular_metricas(resultado)

    # ── N_TRIALS en paralelo ──────────────────────────────────────────────────
    _banner(f"Optimización: {N_TRIALS_DEBUG} trials")

    sampler = crear_sampler(cfg.OPTUNA_SAMPLER, cfg.OPTUNA_SEED, N_TRIALS_DEBUG)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    tiempos: dict[str, list[float]] = {
        "senales": [], "verif_senales": [], "simulacion": [],
        "verif_resultado": [], "metricas": [],
    }

    def objective(trial):
        p = estrategia.espacio_busqueda(trial)
        st, _ = _salida_para_trial(salida_base, trial)

        t0 = time.perf_counter()
        s = getattr(estrategia, "generar_señales")(df_tf, p)
        tiempos["senales"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        integridad.verificar_senales(df_tf, s)
        tiempos["verif_senales"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        res = simular_dataframe(
            df_tf, s,
            saldo_inicial=cfg.SALDO_INICIAL,
            saldo_por_trade=cfg.SALDO_USADO_POR_TRADE,
            apalancamiento=cfg.APALANCAMIENTO,
            saldo_minimo=cfg.SALDO_MINIMO_OPERATIVO,
            comision_pct=cfg.COMISION_PCT,
            comision_lados=cfg.COMISION_LADOS,
            exit_type=st.tipo,
            exit_sl_pct=st.sl_pct,
            exit_tp_pct=st.tp_pct,
            exit_velas=st.velas,
            salidas_custom=None,
        )
        tiempos["simulacion"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        integridad.verificar_resultado(df_tf, s, res, None, _cache=cache)
        tiempos["verif_resultado"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        m = calcular_metricas(res)
        tiempos["metricas"].append(time.perf_counter() - t0)

        return float(calcular_score(m))

    n_jobs = _normalizar_jobs(cfg.N_JOBS)
    t_total = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS_DEBUG, n_jobs=n_jobs)
    t_total = time.perf_counter() - t_total

    # ── Resumen ───────────────────────────────────────────────────────────────
    _banner("RESUMEN — promedio por trial")

    total_media = sum(sum(v)/len(v) for v in tiempos.values() if v)
    rows = []
    for etapa, vals in tiempos.items():
        if not vals:
            continue
        media = sum(vals) / len(vals)
        rows.append((etapa, media, max(vals)))

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"\n  {'ETAPA':<25} {'MEDIA':>8}  {'MAX':>8}  {'%':>5}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*8}  {'─'*5}")
    for etapa, media, maximo in rows:
        pct = media / total_media * 100 if total_media else 0
        print(f"  {etapa:<25} {media*1000:>7.1f}ms  {maximo*1000:>7.1f}ms  {pct:>4.0f}%")

    print(f"\n  Suma media/trial: {total_media*1000:.1f}ms")
    print(f"  Tiempo total:     {t_total:.2f}s  ({N_TRIALS_DEBUG} trials, {n_jobs} workers)")
    print(f"  Throughput:       {N_TRIALS_DEBUG/t_total:.2f} trials/s\n")

    _banner("TIEMPOS PRE-OPTIMIZACIÓN")
    print(f"\n  {'ETAPA':<38} {'TIEMPO':>8}")
    print(f"  {'─'*38}  {'─'*8}")
    pre = 0.0
    for etapa, dt in _log:
        pre += dt
        print(f"  {etapa:<38} {dt:.3f}s")
    print(f"\n  TOTAL: {pre:.3f}s\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass

from CONFIGURACION import config as cfg
from CONFIGURACION.validador_config import validar as validar_config
from DATOS.cargador import cargar
from DATOS.resampleo import resamplear
from DATOS.validador import validar as validar_datos
from MOTOR import simular_dataframe
from NUCLEO import integridad
from NUCLEO.registro import cargar_estrategias, obtener_estrategia
from OPTIMIZACION import calcular_score
from REPORTES import guardar_resultado


@dataclass(frozen=True)
class ExitConfig:
    tipo: str
    sl_pct: float
    tp_pct: float
    velas: int


def main() -> None:
    print("[RUN] Backtest fijo end-to-end sin Optuna")
    validar_config(cfg)

    registro = cargar_estrategias()
    estrategias = obtener_estrategia(registro, cfg.ESTRATEGIA_ID)
    columnas_requeridas = _columnas_requeridas(estrategias)
    activos = _como_lista(cfg.ACTIVOS)
    timeframes = _como_lista(cfg.TIMEFRAMES)
    salidas = list(_salidas_a_ejecutar())

    total_resultados = 0
    for activo in activos:
        df_base = cargar(activo, cfg)
        validar_datos(df_base, activo, columnas_requeridas, timeframe="1m")
        huella_base = integridad.huella_dataframe(f"{activo} carga", df_base)
        _imprimir_huella(huella_base)

        for timeframe in timeframes:
            df_tf = resamplear(df_base, timeframe)
            validar_datos(df_tf, f"{activo} {timeframe}", columnas_requeridas, timeframe=timeframe)
            integridad.verificar_resampleo(df_base, df_tf, timeframe)
            huella_tf = integridad.huella_dataframe(f"{activo} {timeframe}", df_tf)
            _imprimir_huella(huella_tf)

            for estrategia in estrategias:
                params = estrategia.parametros_por_defecto()
                if not params:
                    raise ValueError(
                        f"La estrategia {estrategia.NOMBRE} no define parametros_por_defecto()."
                    )

                senales = getattr(estrategia, "generar_se\u00f1ales")(df_tf, params)
                conteo = integridad.verificar_senales(df_tf, senales)
                print(
                    "[SENALES] "
                    f"{activo} {timeframe} {estrategia.NOMBRE}: "
                    f"LONG={conteo[1]:,} SHORT={conteo[-1]:,} SIN={conteo[0]:,}"
                )

                for salida in salidas:
                    resultado = simular_dataframe(
                        df_tf,
                        senales,
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
                    integridad.verificar_resultado(df_tf, senales, resultado)
                    score = calcular_score(resultado)
                    run_dir = guardar_resultado(
                        carpeta_resultados=cfg.CARPETA_RESULTADOS,
                        activo=activo,
                        timeframe=timeframe,
                        estrategia_id=estrategia.ID,
                        estrategia_nombre=estrategia.NOMBRE,
                        parametros=params,
                        salida=salida,
                        resultado=resultado,
                        score=score,
                        huella_base=huella_base,
                        huella_timeframe=huella_tf,
                        conteo_senales=conteo,
                    )
                    _imprimir_resultado(
                        activo,
                        timeframe,
                        estrategia.NOMBRE,
                        params,
                        salida,
                        resultado,
                        score,
                        run_dir,
                    )
                    total_resultados += 1

    print(f"[OK] Backtest fijo completado. Resultados verificados: {total_resultados}")


def _salidas_a_ejecutar():
    exit_type = cfg.EXIT_TYPE
    if exit_type in {"FIXED", "ALL"}:
        from SALIDAS import fijo

        yield ExitConfig(
            tipo="FIXED",
            sl_pct=float(fijo.EXIT_SL_PCT),
            tp_pct=float(fijo.EXIT_TP_PCT),
            velas=0,
        )

    if exit_type in {"BARS", "ALL"}:
        from SALIDAS import velas

        yield ExitConfig(
            tipo="BARS",
            sl_pct=float(velas.EXIT_SL_PCT),
            tp_pct=0.0,
            velas=int(velas.EXIT_VELAS),
        )


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


def _imprimir_resultado(
    activo: str,
    timeframe: str,
    estrategia: str,
    params: dict,
    salida: ExitConfig,
    resultado,
    score: float,
    run_dir,
) -> None:
    print(
        "[RESULTADO] "
        f"{activo} {timeframe} {estrategia} {salida.tipo}: "
        f"trades={resultado.total_trades:,} "
        f"saldo_final={resultado.saldo_final:,.2f} "
        f"pnl={resultado.pnl_total:,.2f} "
        f"score={score:.6f} "
        f"roi={resultado.roi_total:.2%} "
        f"win_rate={resultado.win_rate:.2%} "
        f"max_dd={resultado.max_drawdown:.2%} "
        f"params={params} "
        f"dir={run_dir}"
    )

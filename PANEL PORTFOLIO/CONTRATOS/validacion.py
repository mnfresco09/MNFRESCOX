"""Construcción y validación de la configuración tipada (motor de riesgo)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from CONFIGURACION import _tecnico, config
from CONTRATOS.errores import ErrorConfiguracion
from CONTRATOS.modelos import (
    Configuracion,
    ParametrosRegimen,
    Restricciones,
    ViewBlackLitterman,
)

RAIZ_PANEL = Path(__file__).resolve().parents[1]
PERFILES_REGIMEN_VALIDOS = frozenset(_tecnico.PRESETS_REGIMEN)
IDIOMAS_REPORTE_VALIDOS = frozenset({"es", "it"})
OPTIMIZATION_ENGINES_VALIDOS = frozenset({"ALL", "MARKOWITZ", "CVAR", "NCO"})


def _validar_tickers(tickers: Sequence[str], activo_referencia: str) -> tuple[str, ...]:
    if any(not isinstance(t, str) for t in tickers):
        raise ErrorConfiguracion("TICKERS solo admite valores de texto.")
    normalizados = tuple(t.strip() for t in tickers)
    if len(normalizados) < 2:
        raise ErrorConfiguracion("TICKERS debe contener al menos dos activos.")
    if any(not t for t in normalizados):
        raise ErrorConfiguracion("TICKERS no admite valores vacíos.")
    if len(set(normalizados)) != len(normalizados):
        raise ErrorConfiguracion("TICKERS contiene activos duplicados.")
    if activo_referencia not in normalizados:
        raise ErrorConfiguracion("ACTIVO_REFERENCIA debe pertenecer a TICKERS.")
    return normalizados


def _validar_fechas(fecha_inicio: str, fecha_fin: str) -> tuple[str, str]:
    try:
        inicio = pd.Timestamp(fecha_inicio)
        fin = pd.Timestamp(fecha_fin)
    except (TypeError, ValueError) as exc:
        raise ErrorConfiguracion("FECHA_INICIO o FECHA_FIN no es válida.") from exc
    if inicio.tzinfo is not None or fin.tzinfo is not None:
        raise ErrorConfiguracion("Las fechas no deben tener zona horaria.")
    if inicio >= fin:
        raise ErrorConfiguracion("FECHA_INICIO debe ser anterior a FECHA_FIN.")
    return inicio.date().isoformat(), fin.date().isoformat()


def _validar_restricciones(
    n_activos: int,
    solo_largos: bool,
    peso_maximo: float | None,
    peso_minimo: float,
    turnover_maximo: float | None,
) -> Restricciones:
    if peso_maximo is not None:
        if not math.isfinite(peso_maximo) or not 0 < peso_maximo <= 1:
            raise ErrorConfiguracion("PESO_MAXIMO_POR_ACTIVO debe estar en (0, 1].")
        if n_activos * peso_maximo < 1 - 1e-12:
            raise ErrorConfiguracion(
                "La suma de topes por activo no permite alcanzar peso total 1."
            )
    elif not solo_largos:
        raise ErrorConfiguracion(
            "PESO_MAXIMO_POR_ACTIVO es obligatorio cuando se permiten cortos."
        )
    if not math.isfinite(peso_minimo) or peso_minimo < 0:
        raise ErrorConfiguracion("PESO_MINIMO_POR_ACTIVO debe ser >= 0.")
    if n_activos * peso_minimo > 1 + 1e-12:
        raise ErrorConfiguracion("La suma de suelos por activo supera 1.")
    if peso_maximo is not None and peso_minimo > peso_maximo:
        raise ErrorConfiguracion("PESO_MINIMO no puede superar PESO_MAXIMO.")
    if turnover_maximo is not None and (not math.isfinite(turnover_maximo) or turnover_maximo < 0):
        raise ErrorConfiguracion("TURNOVER_MAXIMO debe ser >= 0 o None.")
    return Restricciones(
        solo_largos=bool(solo_largos),
        peso_maximo=float(peso_maximo) if peso_maximo is not None else None,
        peso_minimo=float(peso_minimo),
        turnover_maximo=float(turnover_maximo) if turnover_maximo is not None else None,
    )


def _validar_percentiles_perfil(perc: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    if not perc:
        raise ErrorConfiguracion("PERCENTILES_PERFIL no puede estar vacío.")
    salida: list[tuple[str, float]] = []
    for nivel, p in perc.items():
        if not math.isfinite(p) or not 0 < p < 1:
            raise ErrorConfiguracion(f"Percentil del perfil '{nivel}' debe estar en (0, 1).")
        salida.append((str(nivel), float(p)))
    return tuple(salida)


def _validar_optimization_engine(engine: str) -> str:
    normalizado = str(engine).strip().upper()
    if normalizado not in OPTIMIZATION_ENGINES_VALIDOS:
        raise ErrorConfiguracion(
            f"OPTIMIZATION_ENGINE debe ser uno de {sorted(OPTIMIZATION_ENGINES_VALIDOS)}."
        )
    return normalizado


def _validar_views(
    views: Sequence[Mapping[str, object]],
    tickers: tuple[str, ...],
) -> tuple[ViewBlackLitterman, ...]:
    resultado: list[ViewBlackLitterman] = []
    for i, view in enumerate(views):
        activos_brutos = view.get("activos")
        if not isinstance(activos_brutos, Mapping) or not activos_brutos:
            raise ErrorConfiguracion(f"View Black-Litterman {i} debe definir activos.")
        activos: list[tuple[str, float]] = []
        for activo, coef_bruto in activos_brutos.items():
            if activo not in tickers:
                raise ErrorConfiguracion(f"View {i}: activo desconocido '{activo}'.")
            try:
                coef = float(coef_bruto)
            except (TypeError, ValueError) as exc:
                raise ErrorConfiguracion(f"View {i}: coeficiente no numérico.") from exc
            if not math.isfinite(coef):
                raise ErrorConfiguracion(f"View {i}: coeficiente no finito.")
            activos.append((activo, coef))
        if all(abs(c) <= 1e-15 for _, c in activos):
            raise ErrorConfiguracion(f"View {i}: todos los coeficientes son cero.")
        try:
            retorno = float(view.get("retorno_anual", math.nan))
            confianza = float(view.get("confianza", math.nan))
        except (TypeError, ValueError) as exc:
            raise ErrorConfiguracion(f"View {i}: retorno o confianza no numérico.") from exc
        if not math.isfinite(retorno):
            raise ErrorConfiguracion(f"View {i}: retorno_anual no finito.")
        if not math.isfinite(confianza) or not 0 < confianza <= 1:
            raise ErrorConfiguracion(f"View {i}: confianza debe estar en (0, 1].")
        resultado.append(ViewBlackLitterman(tuple(activos), retorno, confianza))
    return tuple(resultado)


def _regimen_desde_preset(nombre: str) -> ParametrosRegimen:
    if nombre not in PERFILES_REGIMEN_VALIDOS:
        raise ErrorConfiguracion(
            f"PERFIL_REGIMEN debe ser uno de {sorted(PERFILES_REGIMEN_VALIDOS)}."
        )
    p = _tecnico.PRESETS_REGIMEN[nombre]
    if not p["umbral_drawdown_crisis"] < p["umbral_drawdown_bajista"] < 0:
        raise ErrorConfiguracion("Umbrales de régimen: crisis < bajista < 0.")
    return ParametrosRegimen(
        drawdown_crisis=float(p["umbral_drawdown_crisis"]),
        drawdown_bajista=float(p["umbral_drawdown_bajista"]),
        ventana_volatilidad=int(p["ventana_volatilidad"]),
        ventana_media_larga=int(p["ventana_media_larga"]),
        ventana_pendiente=int(p["ventana_pendiente"]),
        percentil_volatilidad_crisis=float(p["percentil_volatilidad_crisis"]),
    )


def construir_configuracion(
    *,
    tickers: Sequence[str],
    activo_referencia: str,
    fecha_inicio: str,
    fecha_fin: str,
    solo_largos: bool = True,
    peso_maximo: float | None = 0.40,
    peso_minimo: float = 0.0,
    turnover_maximo: float | None = None,
    horizonte_dias: int = 21,
    capital_base: float = 1_000_000.0,
    tasa_libre_riesgo_anual: float = 0.0,
    nivel_confianza_95: float = 0.95,
    nivel_confianza_99: float = 0.99,
    percentiles_perfil: Mapping[str, float] | None = None,
    lambda_ewma: float = 0.94,
    shrinkage_retorno: float = 0.50,
    ewma_min_obs: int = 60,
    n_puntos_frontera: int = 120,
    n_carteras_factibles: int = 20_000,
    n_trayectorias_mc: int = 50_000,
    percentiles_fan: Sequence[int] = (5, 25, 50, 75, 95),
    semilla: int = 42,
    optimization_engine: str = "ALL",
    views_black_litterman: Sequence[Mapping[str, object]] = (),
    perfil_regimen: str = "estandar",
    min_retornos_analisis: int = 252,
    dias_anio: int = 252,
    idioma_reporte: str = "es",
    carpeta_historico: Path | None = None,
    carpeta_salidas: Path | None = None,
) -> Configuracion:
    """Valida los parámetros declarativos y devuelve una configuración inmutable."""
    tickers_v = _validar_tickers(tickers, activo_referencia)
    inicio_v, fin_v = _validar_fechas(fecha_inicio, fecha_fin)
    restricciones = _validar_restricciones(
        len(tickers_v), solo_largos, peso_maximo, peso_minimo, turnover_maximo
    )

    if horizonte_dias < 1:
        raise ErrorConfiguracion("HORIZONTE_DIAS debe ser >= 1.")
    if not math.isfinite(capital_base) or capital_base <= 0:
        raise ErrorConfiguracion("CAPITAL_BASE debe ser positivo.")
    if not math.isfinite(tasa_libre_riesgo_anual):
        raise ErrorConfiguracion("TASA_LIBRE_RIESGO_ANUAL debe ser finita.")
    for nombre, nivel in (("95", nivel_confianza_95), ("99", nivel_confianza_99)):
        if not 0 < nivel < 1:
            raise ErrorConfiguracion(f"NIVEL_CONFIANZA_{nombre} debe estar en (0, 1).")
    if nivel_confianza_95 >= nivel_confianza_99:
        raise ErrorConfiguracion("NIVEL_CONFIANZA_95 debe ser menor que NIVEL_CONFIANZA_99.")
    if not 0 < lambda_ewma < 1:
        raise ErrorConfiguracion("LAMBDA_EWMA debe estar en (0, 1).")
    if not 0 <= shrinkage_retorno <= 1:
        raise ErrorConfiguracion("SHRINKAGE_RETORNO debe estar en [0, 1].")
    if min(n_puntos_frontera, n_carteras_factibles, n_trayectorias_mc) <= 0:
        raise ErrorConfiguracion("Los tamaños de rejilla/simulación deben ser positivos.")
    if min_retornos_analisis < 252:
        raise ErrorConfiguracion("MIN_RETORNOS_ANALISIS debe ser al menos 252.")
    if dias_anio <= 0:
        raise ErrorConfiguracion("DIAS_ANIO debe ser positivo.")
    percentiles_fan_v = tuple(int(p) for p in percentiles_fan)
    if any(not 0 < p < 100 for p in percentiles_fan_v):
        raise ErrorConfiguracion("PERCENTILES_FAN deben estar en (0, 100).")

    perc_perfil = _validar_percentiles_perfil(
        percentiles_perfil if percentiles_perfil is not None
        else {"bajo": 0.20, "medio": 0.50, "alto": 0.80}
    )
    views = _validar_views(views_black_litterman, tickers_v)
    optimization_engine_v = _validar_optimization_engine(optimization_engine)
    parametros_regimen = _regimen_desde_preset(perfil_regimen)
    if idioma_reporte not in IDIOMAS_REPORTE_VALIDOS:
        raise ErrorConfiguracion(f"IDIOMA_REPORTE debe ser uno de {sorted(IDIOMAS_REPORTE_VALIDOS)}.")

    return Configuracion(
        tickers=tickers_v,
        fecha_inicio=inicio_v,
        fecha_fin=fin_v,
        activo_referencia=activo_referencia,
        restricciones=restricciones,
        horizonte_dias=int(horizonte_dias),
        capital_base=float(capital_base),
        tasa_libre_riesgo_anual=float(tasa_libre_riesgo_anual),
        dias_anio=int(dias_anio),
        nivel_confianza_95=float(nivel_confianza_95),
        nivel_confianza_99=float(nivel_confianza_99),
        percentiles_perfil=perc_perfil,
        lambda_ewma=float(lambda_ewma),
        shrinkage_retorno=float(shrinkage_retorno),
        ewma_min_obs=int(ewma_min_obs),
        n_puntos_frontera=int(n_puntos_frontera),
        n_carteras_factibles=int(n_carteras_factibles),
        n_trayectorias_mc=int(n_trayectorias_mc),
        percentiles_fan=percentiles_fan_v,
        semilla=int(semilla),
        optimization_engine=optimization_engine_v,
        views_black_litterman=views,
        parametros_regimen=parametros_regimen,
        min_retornos_analisis=int(min_retornos_analisis),
        idioma_reporte=idioma_reporte,
        carpeta_historico=(
            Path(carpeta_historico) if carpeta_historico is not None
            else RAIZ_PANEL / "HISTORICO"
        ),
        carpeta_salidas=(
            Path(carpeta_salidas) if carpeta_salidas is not None
            else RAIZ_PANEL / "SALIDAS"
        ),
    )


def cargar_configuracion() -> Configuracion:
    """Carga exclusivamente los parámetros declarados por el usuario en config.py."""
    return construir_configuracion(
        tickers=config.TICKERS,
        activo_referencia=config.ACTIVO_REFERENCIA,
        fecha_inicio=config.FECHA_INICIO,
        fecha_fin=config.FECHA_FIN,
        solo_largos=config.SOLO_LARGOS,
        peso_maximo=config.PESO_MAXIMO_POR_ACTIVO,
        peso_minimo=config.PESO_MINIMO_POR_ACTIVO,
        turnover_maximo=config.TURNOVER_MAXIMO,
        horizonte_dias=config.HORIZONTE_DIAS,
        capital_base=config.CAPITAL_BASE,
        tasa_libre_riesgo_anual=config.TASA_LIBRE_RIESGO_ANUAL,
        nivel_confianza_95=config.NIVEL_CONFIANZA_95,
        nivel_confianza_99=config.NIVEL_CONFIANZA_99,
        percentiles_perfil=config.PERCENTILES_PERFIL,
        lambda_ewma=_tecnico.LAMBDA_EWMA,
        shrinkage_retorno=_tecnico.SHRINKAGE_RETORNO,
        ewma_min_obs=_tecnico.EWMA_MIN_OBS,
        n_puntos_frontera=_tecnico.N_PUNTOS_FRONTERA,
        n_carteras_factibles=_tecnico.N_CARTERAS_FACTIBLES,
        n_trayectorias_mc=_tecnico.N_TRAYECTORIAS_MC,
        percentiles_fan=_tecnico.PERCENTILES_FAN,
        semilla=_tecnico.SEMILLA,
        optimization_engine=getattr(config, "OPTIMIZATION_ENGINE", "ALL"),
        views_black_litterman=config.VIEWS_BLACK_LITTERMAN,
        perfil_regimen=config.PERFIL_REGIMEN,
        min_retornos_analisis=_tecnico.MIN_RETORNOS_ANALISIS,
        dias_anio=_tecnico.DIAS_ANIO,
        idioma_reporte=config.IDIOMA_REPORTE,
    )

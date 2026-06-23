"""Construcción y validación de la configuración tipada."""

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
    VentanaStress,
    ViewBlackLitterman,
)

RAIZ_PANEL = Path(__file__).resolve().parents[1]
FRECUENCIAS_VALIDAS = frozenset({"M", "W", "Q"})
PERFILES_REGIMEN_VALIDOS = frozenset(_tecnico.PRESETS_REGIMEN)
PERFILES_RIESGO_VALIDOS = frozenset((*_tecnico.FRACCION_VOL_NIVEL, "personalizado"))
IDIOMAS_REPORTE_VALIDOS = frozenset({"es", "it"})


def _validar_perfil_riesgo(perfil: str, volatilidad_objetivo: float | None) -> str:
    if perfil not in PERFILES_RIESGO_VALIDOS:
        raise ErrorConfiguracion(
            f"PERFIL_RIESGO debe ser uno de {sorted(PERFILES_RIESGO_VALIDOS)}."
        )
    if perfil == "personalizado" and volatilidad_objetivo is None:
        raise ErrorConfiguracion(
            "PERFIL_RIESGO='personalizado' requiere VOLATILIDAD_OBJETIVO_ANUAL."
        )
    return perfil


def _validar_idioma_reporte(idioma: str) -> str:
    if idioma not in IDIOMAS_REPORTE_VALIDOS:
        raise ErrorConfiguracion(
            f"IDIOMA_REPORTE debe ser uno de {sorted(IDIOMAS_REPORTE_VALIDOS)}."
        )
    return idioma


def _validar_volatilidad_objetivo(volatilidad_objetivo: float | None) -> float | None:
    if volatilidad_objetivo is None:
        return None
    if not math.isfinite(volatilidad_objetivo) or volatilidad_objetivo <= 0:
        raise ErrorConfiguracion(
            "VOLATILIDAD_OBJETIVO_ANUAL debe ser un número positivo o None."
        )
    return float(volatilidad_objetivo)


def _regimen_desde_preset(nombre: str) -> ParametrosRegimen:
    if nombre not in PERFILES_REGIMEN_VALIDOS:
        raise ErrorConfiguracion(
            f"PERFIL_REGIMEN debe ser uno de {sorted(PERFILES_REGIMEN_VALIDOS)}."
        )
    p = _tecnico.PRESETS_REGIMEN[nombre]
    return _validar_regimen(
        p["umbral_drawdown_crisis"], p["umbral_drawdown_bajista"],
        p["ventana_volatilidad"], p["ventana_media_larga"], p["ventana_pendiente"],
        p["percentil_volatilidad_crisis"],
    )


def _validar_tickers(
    tickers: Sequence[str],
    activo_referencia: str,
) -> tuple[str, ...]:
    if any(not isinstance(ticker, str) for ticker in tickers):
        raise ErrorConfiguracion("TICKERS solo admite valores de texto.")
    normalizados = tuple(ticker.strip() for ticker in tickers)
    if len(normalizados) < 2:
        raise ErrorConfiguracion("TICKERS debe contener al menos dos activos.")
    if any(not ticker for ticker in normalizados):
        raise ErrorConfiguracion("TICKERS no admite valores vacíos.")
    if len(set(normalizados)) != len(normalizados):
        raise ErrorConfiguracion("TICKERS contiene activos duplicados.")
    if activo_referencia not in normalizados:
        raise ErrorConfiguracion(
            "ACTIVO_REFERENCIA debe pertenecer a TICKERS."
        )
    return normalizados


def _validar_limite(
    n_activos: int,
    solo_largos: bool,
    peso_maximo: float | None,
) -> Restricciones:
    if peso_maximo is None:
        if not solo_largos:
            raise ErrorConfiguracion(
                "PESO_MAXIMO_POR_ACTIVO es obligatorio cuando se permiten cortos."
            )
        return Restricciones(solo_largos=True, peso_maximo=None)
    if not math.isfinite(peso_maximo) or not 0 < peso_maximo <= 1:
        raise ErrorConfiguracion(
            "PESO_MAXIMO_POR_ACTIVO debe ser finito y pertenecer a (0, 1]."
        )
    if n_activos * peso_maximo < 1 - 1e-12:
        raise ErrorConfiguracion(
            "La suma de límites por activo es inviable para alcanzar peso total 1."
        )
    return Restricciones(
        solo_largos=solo_largos,
        peso_maximo=float(peso_maximo),
    )


def _validar_fechas(fecha_inicio: str, fecha_fin: str) -> tuple[str, str]:
    try:
        inicio = pd.Timestamp(fecha_inicio)
        fin = pd.Timestamp(fecha_fin)
    except (TypeError, ValueError) as exc:
        raise ErrorConfiguracion("FECHA_INICIO o FECHA_FIN no es válida.") from exc
    if inicio.tzinfo is not None or fin.tzinfo is not None:
        raise ErrorConfiguracion("Las fechas de configuración no deben tener zona horaria.")
    if inicio >= fin:
        raise ErrorConfiguracion("FECHA_INICIO debe ser anterior a FECHA_FIN.")
    return inicio.date().isoformat(), fin.date().isoformat()


def _validar_views(
    views: Sequence[Mapping[str, object]],
    tickers: tuple[str, ...],
) -> tuple[ViewBlackLitterman, ...]:
    resultado: list[ViewBlackLitterman] = []
    for indice, view in enumerate(views):
        activos_brutos = view.get("activos")
        if not isinstance(activos_brutos, Mapping) or not activos_brutos:
            raise ErrorConfiguracion(
                f"View Black-Litterman {indice} debe definir activos."
            )
        activos: list[tuple[str, float]] = []
        for activo, coeficiente_bruto in activos_brutos.items():
            if activo not in tickers:
                raise ErrorConfiguracion(
                    f"View Black-Litterman {indice}: activo desconocido '{activo}'."
                )
            try:
                coeficiente = float(coeficiente_bruto)
            except (TypeError, ValueError) as exc:
                raise ErrorConfiguracion(
                    f"View Black-Litterman {indice}: coeficiente no numérico."
                ) from exc
            if not math.isfinite(coeficiente):
                raise ErrorConfiguracion(
                    f"View Black-Litterman {indice}: coeficiente no finito."
                )
            activos.append((activo, coeficiente))
        if all(abs(coeficiente) <= 1e-15 for _, coeficiente in activos):
            raise ErrorConfiguracion(
                f"View Black-Litterman {indice}: todos los coeficientes son cero."
            )
        try:
            retorno = float(view.get("retorno_anual", math.nan))
            confianza = float(view.get("confianza", math.nan))
        except (TypeError, ValueError) as exc:
            raise ErrorConfiguracion(
                f"View Black-Litterman {indice}: retorno o confianza no numérico."
            ) from exc
        if not math.isfinite(retorno):
            raise ErrorConfiguracion(
                f"View Black-Litterman {indice}: retorno_anual no finito."
            )
        if not math.isfinite(confianza) or not 0 < confianza <= 1:
            raise ErrorConfiguracion(
                f"View Black-Litterman {indice}: confianza debe estar en (0, 1]."
            )
        resultado.append(
            ViewBlackLitterman(
                activos=tuple(activos),
                retorno_anual=retorno,
                confianza=confianza,
            )
        )
    return tuple(resultado)


def _validar_stress(
    ventanas: Mapping[str, tuple[str, str]],
) -> tuple[VentanaStress, ...]:
    if not ventanas:
        raise ErrorConfiguracion("VENTANAS_STRESS no puede estar vacío.")
    resultado: list[VentanaStress] = []
    for nombre, periodo in ventanas.items():
        if not nombre.strip() or len(periodo) != 2:
            raise ErrorConfiguracion("Cada ventana de stress requiere nombre e intervalo.")
        try:
            inicio = pd.Timestamp(periodo[0])
            fin = pd.Timestamp(periodo[1])
        except (TypeError, ValueError) as exc:
            raise ErrorConfiguracion(
                f"Ventana de stress '{nombre}' contiene fechas inválidas."
            ) from exc
        if inicio.tzinfo is not None or fin.tzinfo is not None:
            raise ErrorConfiguracion(
                f"Ventana de stress '{nombre}' no debe tener zona horaria."
            )
        if inicio >= fin:
            raise ErrorConfiguracion(
                f"Ventana de stress '{nombre}' debe tener inicio anterior al fin."
            )
        resultado.append(VentanaStress(nombre=nombre, inicio=inicio, fin=fin))
    return tuple(resultado)


def _validar_regimen(
    drawdown_crisis: float,
    drawdown_bajista: float,
    ventana_volatilidad: int,
    ventana_media_larga: int,
    ventana_pendiente: int,
    percentil_volatilidad_crisis: float,
) -> ParametrosRegimen:
    if not drawdown_crisis < drawdown_bajista < 0:
        raise ErrorConfiguracion(
            "Los umbrales deben cumplir crisis < bajista < 0."
        )
    if min(ventana_volatilidad, ventana_media_larga, ventana_pendiente) < 2:
        raise ErrorConfiguracion("Las ventanas de régimen deben ser al menos 2.")
    if not 0 < percentil_volatilidad_crisis < 1:
        raise ErrorConfiguracion(
            "PERCENTIL_VOLATILIDAD_CRISIS debe pertenecer a (0, 1)."
        )
    return ParametrosRegimen(
        drawdown_crisis=float(drawdown_crisis),
        drawdown_bajista=float(drawdown_bajista),
        ventana_volatilidad=int(ventana_volatilidad),
        ventana_media_larga=int(ventana_media_larga),
        ventana_pendiente=int(ventana_pendiente),
        percentil_volatilidad_crisis=float(percentil_volatilidad_crisis),
    )


def construir_configuracion(
    *,
    tickers: Sequence[str],
    activo_referencia: str,
    peso_maximo: float | None,
    fecha_inicio: str = "2019-01-01",
    fecha_fin: str = "2024-12-31",
    frecuencia_rebalanceo: str = "M",
    ventana_estimacion: int = 504,
    solo_largos: bool = True,
    perfil_riesgo: str = "moderado",
    volatilidad_objetivo: float | None = None,
    idioma_reporte: str = "es",
    tasa_libre_riesgo_anual: float = 0.0,
    dias_anio: int = 252,
    coste_transaccion_pb: float = 10.0,
    nivel_confianza: float = 0.95,
    min_retornos_analisis: int = 252,
    views_black_litterman: Sequence[Mapping[str, object]] = (),
    ventanas_stress: Mapping[str, tuple[str, str]] | None = None,
    perfil_regimen: str = "estandar",
    n_carteras_montecarlo: int = 20_000,
    semilla: int = 42,
    carpeta_historico: Path | None = None,
    carpeta_salidas: Path | None = None,
) -> Configuracion:
    """Valida datos declarativos y devuelve una configuración inmutable."""

    tickers_validados = _validar_tickers(tickers, activo_referencia)
    fecha_inicio_validada, fecha_fin_validada = _validar_fechas(
        fecha_inicio,
        fecha_fin,
    )
    if frecuencia_rebalanceo not in FRECUENCIAS_VALIDAS:
        raise ErrorConfiguracion(
            f"FRECUENCIA_REBALANCEO debe ser una de {sorted(FRECUENCIAS_VALIDAS)}."
        )
    if ventana_estimacion < 2:
        raise ErrorConfiguracion("VENTANA_ESTIMACION_DIAS debe ser al menos 2.")
    if dias_anio <= 0:
        raise ErrorConfiguracion("DIAS_ANIO debe ser positivo.")
    if min_retornos_analisis < 252:
        raise ErrorConfiguracion("MIN_RETORNOS_ANALISIS debe ser al menos 252.")
    if ventana_estimacion < min_retornos_analisis:
        raise ErrorConfiguracion(
            "VENTANA_ESTIMACION_DIAS no puede ser menor que MIN_RETORNOS_ANALISIS."
        )
    for nombre, valor in (
        ("TASA_LIBRE_RIESGO_ANUAL", tasa_libre_riesgo_anual),
        ("COSTE_TRANSACCION_PB", coste_transaccion_pb),
    ):
        if not math.isfinite(valor):
            raise ErrorConfiguracion(f"{nombre} debe ser finito.")
    if coste_transaccion_pb < 0:
        raise ErrorConfiguracion("COSTE_TRANSACCION_PB no puede ser negativo.")
    if not 0 < nivel_confianza < 1:
        raise ErrorConfiguracion("NIVEL_CONFIANZA debe pertenecer a (0, 1).")
    if n_carteras_montecarlo <= 0:
        raise ErrorConfiguracion("N_CARTERAS_MONTECARLO debe ser positivo.")

    restricciones = _validar_limite(
        len(tickers_validados),
        solo_largos,
        peso_maximo,
    )
    vol_objetivo_validada = _validar_volatilidad_objetivo(volatilidad_objetivo)
    perfil_riesgo_validado = _validar_perfil_riesgo(perfil_riesgo, vol_objetivo_validada)
    idioma_reporte_validado = _validar_idioma_reporte(idioma_reporte)
    views = _validar_views(views_black_litterman, tickers_validados)
    stress = _validar_stress(
        ventanas_stress
        if ventanas_stress is not None
        else {"crisis_2022": ("2022-01-03", "2022-10-12")}
    )
    parametros_regimen = _regimen_desde_preset(perfil_regimen)
    return Configuracion(
        tickers=tickers_validados,
        fecha_inicio=fecha_inicio_validada,
        fecha_fin=fecha_fin_validada,
        activo_referencia=activo_referencia,
        frecuencia_rebalanceo=frecuencia_rebalanceo,
        ventana_estimacion=int(ventana_estimacion),
        restricciones=restricciones,
        perfil_riesgo=perfil_riesgo_validado,
        volatilidad_objetivo=vol_objetivo_validada,
        idioma_reporte=idioma_reporte_validado,
        tasa_libre_riesgo_anual=float(tasa_libre_riesgo_anual),
        dias_anio=int(dias_anio),
        coste_transaccion_pb=float(coste_transaccion_pb),
        nivel_confianza=float(nivel_confianza),
        min_retornos_analisis=int(min_retornos_analisis),
        views_black_litterman=views,
        ventanas_stress=stress,
        parametros_regimen=parametros_regimen,
        n_carteras_montecarlo=int(n_carteras_montecarlo),
        semilla=int(semilla),
        carpeta_historico=(
            Path(carpeta_historico)
            if carpeta_historico is not None
            else RAIZ_PANEL / "HISTORICO"
        ),
        carpeta_salidas=(
            Path(carpeta_salidas)
            if carpeta_salidas is not None
            else RAIZ_PANEL / "SALIDAS"
        ),
    )


def cargar_configuracion() -> Configuracion:
    """Carga exclusivamente los parámetros declarados por el usuario."""

    return construir_configuracion(
        tickers=config.TICKERS,
        activo_referencia=config.ACTIVO_REFERENCIA,
        peso_maximo=config.PESO_MAXIMO_POR_ACTIVO,
        fecha_inicio=config.FECHA_INICIO,
        fecha_fin=config.FECHA_FIN,
        frecuencia_rebalanceo=config.FRECUENCIA_REBALANCEO,
        ventana_estimacion=config.VENTANA_ESTIMACION_DIAS,
        solo_largos=config.SOLO_LARGOS,
        perfil_riesgo=config.PERFIL_RIESGO,
        volatilidad_objetivo=config.VOLATILIDAD_OBJETIVO_ANUAL,
        idioma_reporte=config.IDIOMA_REPORTE,
        tasa_libre_riesgo_anual=config.TASA_LIBRE_RIESGO_ANUAL,
        dias_anio=_tecnico.DIAS_ANIO,
        coste_transaccion_pb=config.COSTE_TRANSACCION_PB,
        nivel_confianza=config.NIVEL_CONFIANZA,
        min_retornos_analisis=_tecnico.MIN_RETORNOS_ANALISIS,
        views_black_litterman=config.VIEWS_BLACK_LITTERMAN,
        ventanas_stress=config.VENTANAS_STRESS,
        perfil_regimen=config.PERFIL_REGIMEN,
        n_carteras_montecarlo=_tecnico.N_CARTERAS_MONTECARLO,
        semilla=_tecnico.SEMILLA,
    )

"""Modelos públicos e inmutables compartidos entre las capas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class Restricciones:
    """Límites comunes de una cartera."""

    solo_largos: bool
    peso_maximo: float | None


@dataclass(frozen=True)
class ViewBlackLitterman:
    """Opinión absoluta o relativa expresada sobre una combinación de activos."""

    activos: tuple[tuple[str, float], ...]
    retorno_anual: float
    confianza: float


@dataclass(frozen=True)
class VentanaStress:
    """Episodio histórico configurable."""

    nombre: str
    inicio: pd.Timestamp
    fin: pd.Timestamp


@dataclass(frozen=True)
class ParametrosRegimen:
    """Umbrales transparentes para clasificar regímenes."""

    drawdown_crisis: float
    drawdown_bajista: float
    ventana_volatilidad: int
    ventana_media_larga: int
    ventana_pendiente: int
    percentil_volatilidad_crisis: float


@dataclass(frozen=True)
class Configuracion:
    """Configuración validada consumida por el pipeline."""

    tickers: tuple[str, ...]
    fecha_inicio: str
    fecha_fin: str
    activo_referencia: str
    frecuencia_rebalanceo: str
    ventana_estimacion: int
    restricciones: Restricciones
    retorno_objetivo_anual: float
    tasa_libre_riesgo_anual: float
    dias_anio: int
    coste_transaccion_pb: float
    nivel_confianza: float
    min_retornos_analisis: int
    views_black_litterman: tuple[ViewBlackLitterman, ...]
    ventanas_stress: tuple[VentanaStress, ...]
    parametros_regimen: ParametrosRegimen
    n_carteras_montecarlo: int
    semilla: int
    carpeta_historico: Path
    carpeta_salidas: Path


@dataclass(frozen=True)
class ResumenActivo:
    """Diagnóstico persistido de un activo descargado."""

    ticker: str
    archivo: str
    filas: int
    fecha_inicio: pd.Timestamp
    fecha_fin: pd.Timestamp
    huecos_sospechosos: int
    hueco_max_dias: int


@dataclass(frozen=True)
class DatosAlineados:
    """Cierres y log-retornos sobre el calendario común."""

    activos: tuple[str, ...]
    cierres: pd.DataFrame
    log_retornos: pd.DataFrame


@dataclass(frozen=True)
class ResultadoPCA:
    varianza_explicada: pd.Series
    varianza_acumulada: pd.Series
    cargas: pd.DataFrame


@dataclass(frozen=True)
class ResultadoAnalisis:
    log_retornos: pd.DataFrame
    retornos_esperados: pd.Series
    covarianza: pd.DataFrame
    volatilidades: pd.Series
    correlacion_media: pd.DataFrame
    correlacion_cola: pd.DataFrame
    diferencia_correlacion_cola: pd.DataFrame
    observaciones_cola: int
    pca: ResultadoPCA
    regimenes: pd.Series


@dataclass(frozen=True)
class MetricasEstimadas:
    retorno_anual: float
    volatilidad_anual: float
    sharpe: float


@dataclass(frozen=True)
class ResultadoAsignacion:
    nombre: str
    pesos: pd.Series
    metricas: MetricasEstimadas
    estado_solver: str
    diagnostico: str
    advertencias: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResultadoMonteCarlo:
    pesos: pd.DataFrame
    metricas: pd.DataFrame


@dataclass(frozen=True)
class ResultadoFrontera:
    puntos: pd.DataFrame
    minima_varianza: ResultadoAsignacion
    maximo_sharpe: ResultadoAsignacion
    retorno_objetivo: ResultadoAsignacion


@dataclass(frozen=True)
class Rebalanceo:
    fecha: pd.Timestamp
    pesos: Mapping[str, pd.Series]
    rotacion: Mapping[str, float]
    coste: Mapping[str, float]


@dataclass(frozen=True)
class ResultadoWalkForward:
    retornos: pd.DataFrame
    equity: pd.DataFrame
    pesos_diarios: Mapping[str, pd.DataFrame]
    rebalanceos: tuple[Rebalanceo, ...]


@dataclass(frozen=True)
class MetricasCartera:
    retorno_anual: float
    volatilidad_anual: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    duracion_drawdown_dias: int
    fecha_recuperacion: pd.Timestamp | None
    var: float
    cvar: float


@dataclass(frozen=True)
class ResultadoStress:
    nombre: str
    evaluable: bool
    observaciones: int
    metricas: Mapping[str, MetricasCartera]


@dataclass(frozen=True)
class ResultadoRiesgo:
    walk_forward: ResultadoWalkForward
    metricas: Mapping[str, MetricasCartera]
    metricas_por_regimen: Mapping[str, pd.DataFrame]
    stress: Mapping[str, ResultadoStress]
    diversificacion_crisis: pd.DataFrame
    convexidad: pd.DataFrame


@dataclass(frozen=True)
class PaqueteReporte:
    configuracion: Configuracion
    datos: DatosAlineados
    analisis: ResultadoAnalisis
    analisis_actual: ResultadoAnalisis
    asignaciones: Mapping[str, ResultadoAsignacion]
    frontera: ResultadoFrontera
    monte_carlo: ResultadoMonteCarlo
    riesgo: ResultadoRiesgo
    objetivo: str = "comparar"


@dataclass(frozen=True)
class RutasReporte:
    html: Path
    pdf: Path
    excel: Path
    manifiesto: Path

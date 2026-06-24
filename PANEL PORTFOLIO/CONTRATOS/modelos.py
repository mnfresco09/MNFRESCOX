"""Contratos públicos e inmutables compartidos entre las capas.

Estos `dataclasses` tipados son el ÚNICO vehículo de información entre capas. El
módulo de reporting NO recalcula métricas: solo consume `ReportPayload`.

Flujo de contratos:
  Configuracion + PortfolioInput
    → MomentsResult        (estadística individual + doble lente de covarianza)
    → ResultadoOptimizacion (Strategy: MARKOWITZ / CVAR / NCO)
    → PortfolioCandidate×N (contrato común + MCR + score)
    → RiskForecast         (VaR/CVaR FHS y paramétrico, T+1)
    → SimulationSummary    (fan chart, prob. pérdida, CDaR — vía Rust)
    → ReportPayload        (todo agregado para el dashboard)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# ===========================================================================
#  CONFIGURACIÓN
# ===========================================================================
@dataclass(frozen=True)
class Restricciones:
    """Límites institucionales duros de una cartera."""

    solo_largos: bool
    peso_maximo: float | None
    peso_minimo: float = 0.0
    turnover_maximo: float | None = None


@dataclass(frozen=True)
class ViewBlackLitterman:
    """Opinión absoluta o relativa sobre una combinación de activos."""

    activos: tuple[tuple[str, float], ...]
    retorno_anual: float
    confianza: float


@dataclass(frozen=True)
class ParametrosRegimen:
    """Umbrales transparentes para clasificar el régimen de mercado."""

    drawdown_crisis: float
    drawdown_bajista: float
    ventana_volatilidad: int
    ventana_media_larga: int
    ventana_pendiente: int
    percentil_volatilidad_crisis: float


@dataclass(frozen=True)
class Configuracion:
    """Configuración validada e inmutable que consume el pipeline."""

    # Pregunta 1 — activos y periodo
    tickers: tuple[str, ...]
    fecha_inicio: str
    fecha_fin: str
    activo_referencia: str
    # Restricciones y decisión
    restricciones: Restricciones
    horizonte_dias: int
    capital_base: float
    # Convenciones matemáticas
    tasa_libre_riesgo_anual: float
    dias_anio: int
    nivel_confianza_95: float
    nivel_confianza_99: float
    # Perfiles dinámicos (percentiles de vol de la frontera)
    percentiles_perfil: tuple[tuple[str, float], ...]
    # Doble lente / estimadores
    lambda_ewma: float
    shrinkage_retorno: float
    ewma_min_obs: int
    # Frontera y simulación
    n_puntos_frontera: int
    n_carteras_factibles: int
    n_trayectorias_mc: int
    percentiles_fan: tuple[int, ...]
    semilla: int
    optimization_engine: str
    # Avanzado
    views_black_litterman: tuple[ViewBlackLitterman, ...]
    parametros_regimen: ParametrosRegimen
    min_retornos_analisis: int
    idioma_reporte: str
    # Rutas
    carpeta_historico: Path
    carpeta_salidas: Path


# ===========================================================================
#  DATOS
# ===========================================================================
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
class PortfolioInput:
    """Pregunta 1 — qué activos tengo. Insumo limpio del motor."""

    activos: tuple[str, ...]
    log_retornos: pd.DataFrame          # diarios, alineados
    cierres: pd.DataFrame
    capital_base: float
    horizonte_dias: int


# ===========================================================================
#  ESTADÍSTICA Y DOBLE LENTE DE COVARIANZA  (preguntas 1 y 2)
# ===========================================================================
@dataclass(frozen=True)
class EstadisticaActivo:
    """Estadística individual anualizada de un activo."""

    ticker: str
    retorno_medio: float        # media histórica cruda (anual)
    retorno_ajustado: float     # tras shrinkage conservador (anual)
    volatilidad: float          # estructural (anual)
    volatilidad_tactica: float  # EWMA T+1 (anual)
    asimetria: float
    curtosis: float


@dataclass(frozen=True)
class MomentsResult:
    """Pregunta 2 — cómo se relacionan. Doble lente de covarianza.

    - `cov_estructural`: Ledoit-Wolf, bien condicionada → para OPTIMIZAR.
    - `cov_tactica`: EWMA (RiskMetrics) → para el riesgo de MAÑANA (T+1).
    """

    activos: tuple[str, ...]
    retornos_ajustados: pd.Series          # μ tras shrinkage (anual)
    retornos_medios: pd.Series             # μ histórico crudo (anual)
    cov_estructural: pd.DataFrame          # Ledoit-Wolf (anual)
    cov_tactica: pd.DataFrame              # EWMA (anual)
    volatilidades: pd.Series               # estructural (anual)
    volatilidades_tacticas: pd.Series      # EWMA T+1 (anual)
    correlacion: pd.DataFrame
    correlacion_cola: pd.DataFrame
    shrinkage_cov: float                   # coef. Ledoit-Wolf
    shrinkage_retorno: float               # intensidad shrinkage de μ
    estadisticas: tuple[EstadisticaActivo, ...]
    fuente_tactica: str = "EWMA"           # "EWMA" o "GARCH" (con fallback)


# ===========================================================================
#  FRONTERA Y CANDIDATOS  (pregunta 3)
# ===========================================================================
@dataclass(frozen=True)
class MetricasEstimadas:
    retorno_anual: float
    volatilidad_anual: float
    sharpe: float


@dataclass(frozen=True)
class ResultadoFrontera:
    """Frontera eficiente restringida: el 100% de puntos factibles."""

    puntos: pd.DataFrame                   # [retorno, volatilidad, sharpe, peso·*]
    nube_factible: pd.DataFrame            # fondo de densidad riesgo-retorno
    minima_varianza_pesos: pd.Series
    maximo_sharpe_pesos: pd.Series


@dataclass(frozen=True)
class DescomposicionRiesgo:
    """Contribución marginal y total al riesgo por activo (MCR)."""

    mcr: pd.Series                         # contribución marginal al riesgo
    contribucion: pd.Series                # contribución total (∑ = vol cartera)
    contribucion_pct: pd.Series            # % de la varianza/vol total
    concentracion_hhi: float               # índice Herfindahl de los pesos


@dataclass(frozen=True)
class PortfolioCandidate:
    """Pregunta 3 — una cartera candidata por nivel de riesgo.

    `nivel` identifica perfil o motor. Las métricas de riesgo táctico
    (vol T+1, VaR, CDaR) y el score se rellenan tras el forecast.
    """

    nivel: str
    pesos: pd.Series
    retorno_esperado: float                # anual, geométrico para Strategy
    volatilidad_estructural: float         # anual, Ledoit-Wolf
    volatilidad_tactica: float             # anual, EWMA T+1
    sharpe: float
    descomposicion: DescomposicionRiesgo
    retorno_geometrico: float | None = None    # CAGR in-sample con pesos fijos
    motor_optimizacion: str = ""               # MARKOWITZ / CVAR / NCO
    r2_curva_capital: float | None = None      # diagnóstico, no driver principal
    k_ratio: float | None = None               # estabilidad de la curva, diagnóstico
    forecast: "RiskForecast | None" = None
    simulacion: "SimulationSummary | None" = None
    score: float | None = None
    detalle_score: tuple[tuple[str, float], ...] = ()
    # Métricas de exploración (pregunta 3, multi-lente).
    diversificacion: float | None = None       # ratio de diversificación (Choueifaty)
    starr: float | None = None                 # retorno excedente / CVaR99 (tail-aware)
    erc_concentracion: float | None = None     # HHI de las contribuciones al riesgo (ERC)
    clase_riesgo: str | None = None            # banda detectada: bajo / medio / alto


@dataclass(frozen=True)
class CriterioRanking:
    """Top-N carteras de la frontera bajo un criterio de análisis concreto."""

    clave: str
    nombre: str
    descripcion: str
    sentido: str                               # "max" o "min"
    top: tuple[PortfolioCandidate, ...]


@dataclass(frozen=True)
class ResultadoOptimizacion:
    """Salida única del selector Strategy de optimizadores."""

    frontera: ResultadoFrontera
    candidatos: tuple[PortfolioCandidate, ...]
    curva_top_sharpe: pd.DataFrame
    motores_ejecutados: tuple[str, ...]


# ===========================================================================
#  RIESGO PROSPECTIVO  (pregunta 4)
# ===========================================================================
@dataclass(frozen=True)
class RiskForecast:
    """Pregunta 4 — cuánto puedo perder MAÑANA (T+1).

    Convención de signo: VaR y CVaR son retornos NEGATIVOS (la pérdida en la
    cola). NUNCA se etiquetan como "pérdida máxima": son estimaciones bajo los
    supuestos del modelo (VaR 99% diario estimado).
    """

    horizonte_dias: int
    volatilidad_tactica_diaria: float
    # Histórico (distribución empírica realizada)
    var_hist_95: float
    var_hist_99: float
    cvar_hist_95: float
    cvar_hist_99: float
    # Paramétrico forecast (vol táctica × cuantil normal/t)
    var_param_95: float
    var_param_99: float
    cvar_param_95: float
    cvar_param_99: float
    # Filtered Historical Simulation (motor Rust / fallback)
    var_fhs_95: float
    var_fhs_99: float
    cvar_fhs_95: float
    cvar_fhs_99: float
    fuente_fhs: str = "rust"               # "rust" o "python_fallback"


@dataclass(frozen=True)
class SimulationSummary:
    """Pregunta 4 — cuánto puedo perder este MES (horizonte agregado).

    El motor NUNCA devuelve la matriz completa de trayectorias: solo las series
    de percentiles para el fan chart, la probabilidad de pérdida y el CDaR.
    """

    horizonte_dias: int
    percentiles: tuple[int, ...]
    # series[percentil] = curva de capital base 1 a lo largo del horizonte
    sendas_percentil: pd.DataFrame         # index=día, columns=percentiles
    prob_perdida: float                    # P(retorno horizonte < 0)
    cdar_30d: float                        # Conditional Drawdown at Risk
    retorno_mediano: float                 # percentil 50 a horizonte
    perdida_p5: float                      # cola baja (P5) a horizonte
    fuente: str = "rust"                   # "rust" o "python_fallback"


# ===========================================================================
#  RÉGIMEN Y AGREGADOS
# ===========================================================================
@dataclass(frozen=True)
class RegimenMercado:
    """Régimen actual del mercado (baja / alta volatilidad)."""

    etiqueta: str                          # "baja_volatilidad" / "alta_volatilidad" / "crisis"
    volatilidad_actual: float              # anualizada, ventana corta
    percentil_volatilidad: float           # posición de la vol actual en su historia
    correlacion_media_actual: float        # nivel medio de correlación reciente
    descripcion: str


@dataclass(frozen=True)
class Recomendacion:
    """Cartera ganadora y por qué (decisión ejecutiva)."""

    nivel: str
    criterio: str
    detalle: str


@dataclass(frozen=True)
class ReportPayload:
    """Contrato ÚNICO que consume el reporting. No se recalcula nada aquí."""

    configuracion: Configuracion
    entrada: PortfolioInput
    momentos: MomentsResult
    frontera: ResultadoFrontera
    candidatos: tuple[PortfolioCandidate, ...]   # bajo, medio, alto, max_sharpe
    regimen: RegimenMercado
    recomendada: PortfolioCandidate
    recomendacion: Recomendacion
    curva_top_sharpe: pd.DataFrame = field(default_factory=pd.DataFrame)
    frontera_degenerada: bool = False
    nota_frontera: str = ""
    # Exploración multi-criterio (frontera + nube clasificadas + leaderboard).
    clasificacion_frontera: pd.DataFrame = field(default_factory=pd.DataFrame)
    clasificacion_nube: pd.DataFrame = field(default_factory=pd.DataFrame)
    anclas: tuple[tuple[str, float], ...] = ()
    leaderboard: tuple[CriterioRanking, ...] = ()


@dataclass(frozen=True)
class RutasReporte:
    html: Path
    pdf: Path

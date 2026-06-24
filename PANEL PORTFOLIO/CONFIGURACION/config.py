"""Configuración del PANEL PORTFOLIO — SOLO datos y parámetros, cero lógica.

Motor de Riesgo Predictivo orientado a la decisión institucional (buy-side).
El panel responde a CUATRO preguntas y nada más:

  1. ¿Qué activos tengo?
  2. ¿Cómo se relacionan entre ellos?
  3. ¿Qué motor produce los pesos más robustos?
  4. ¿Cuánto puedo perder mañana o este mes bajo el régimen actual?

Este es el único archivo que editas. El resto del panel lee de aquí y nunca al
revés. Las constantes puramente técnicas (anualización, lambda EWMA, tamaño de
la simulación, semilla) viven en CONFIGURACION/_tecnico.py, fuera de la vista.

PRINCIPIO INSTITUCIONAL: el modo ALL compara motores Champion vs Challenger
(Markowitz, CVaR y NCO). Cuando se usan perfiles de frontera, NO se fijan
volatilidades absolutas (ej. "Bajo = 10% vol"): se derivan dinámicamente de los
percentiles de volatilidad del universo actual. Lo único que el usuario fija son
las restricciones operativas e institucionales mínimas.
"""

from __future__ import annotations

# ===========================================================================
#  BÁSICO  — lo que el usuario toca
# ===========================================================================

# --- Pregunta 1: ¿Qué activos tengo? (símbolos de Yahoo Finance) -----------
TICKERS: list[str] = ["SPY", "QQQ", "TLT", "GLD", "VNQ", "EEM"]

# Activo que representa "el mercado" (correlación de cola y detección de régimen).
# Debe pertenecer a TICKERS.
ACTIVO_REFERENCIA: str = "SPY"

# --- Periodo (datos diarios) -----------------------------------------------
FECHA_INICIO: str = "2015-01-01"
FECHA_FIN: str = "2026-06-22"

# --- Restricciones de cartera (institucionales, duras) ---------------------
SOLO_LARGOS: bool = True                         # True = sin posiciones cortas
PESO_MAXIMO_POR_ACTIVO: float | None = 0.66      # tope por activo; None = sin tope
PESO_MINIMO_POR_ACTIVO: float = 0.0              # suelo por activo (long-only)
TURNOVER_MAXIMO: float | None = None             # rotación máx. vs cartera previa; None = libre

# --- Horizonte de decisión y capital ---------------------------------------
HORIZONTE_DIAS: int = 21                          # horizonte del forecast (≈1 mes bursátil)
CAPITAL_BASE: float = 1_000_000.0                 # capital de referencia para € en riesgo

# --- Idioma del informe ----------------------------------------------------
# Opciones: "es" / "it". Se puede cambiar por prompt al ejecutar, sin tocar esto.
IDIOMA_REPORTE: str = "es"


# ===========================================================================
#  AVANZADO  — rara vez se toca
# ===========================================================================

# Tasa libre de riesgo anual EXPLÍCITA (para Sharpe y Black-Litterman).
TASA_LIBRE_RIESGO_ANUAL: float = 0.05

# Niveles de confianza para VaR / CVaR (forecast e histórico). Ambos se reportan.
NIVEL_CONFIANZA_95: float = 0.95
NIVEL_CONFIANZA_99: float = 0.99

# Percentiles de la distribución de volatilidad de la frontera eficiente que
# definen los perfiles DINÁMICOS. No son volatilidades absolutas: se calculan
# sobre el universo actual cada vez. Bajo=P20, Medio=P50, Alto=P80.
PERCENTILES_PERFIL: dict[str, float] = {
    "bajo": 0.20,
    "medio": 0.50,
    "alto": 0.80,
}

# Motor de optimización activo. ALL ejecuta Champion vs Challenger:
# MARKOWITZ (media-varianza actual), CVAR (Rockafellar-Uryasev) y NCO
# (Nested Clustered Optimization). Valores: "ALL", "MARKOWITZ", "CVAR", "NCO".
OPTIMIZATION_ENGINE: str = "ALL"

# Views de Black-Litterman. VACÍO = NO se calcula Black-Litterman; el estimador
# de retorno cae al shrinkage conservador (fallback institucional). Para
# activarlo, añade opiniones defendibles, p. ej.:
#   {"activos": {"BTEC.L": 1.0}, "retorno_anual": 0.12, "confianza": 0.6}
VIEWS_BLACK_LITTERMAN: list[dict] = []

# Sensibilidad del etiquetado de régimen: "conservador" / "estandar" / "sensible".
PERFIL_REGIMEN: str = "estandar"

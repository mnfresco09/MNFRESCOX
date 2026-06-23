"""Configuración del PANEL PORTFOLIO — SOLO datos y parámetros, cero lógica.

Este es el único archivo que editas. El resto del panel lee de aquí y nunca al
revés. Está dividido en dos bloques:

  • BÁSICO   → lo que normalmente tocas (activos, fechas, perfil de riesgo...).
  • AVANZADO → ajustes finos que rara vez se cambian.

Las constantes puramente técnicas (anualización, tamaño del Monte Carlo, etc.)
viven en CONFIGURACION/_tecnico.py, fuera de la vista.
"""

from __future__ import annotations

# ===========================================================================
#  BÁSICO  — lo que el usuario toca
# ===========================================================================

# --- Cesta de activos (símbolos de Yahoo Finance) --------------------------
TICKERS: list[str] = ["BTEC.L", "INRA.AS", "LOCK.L"]

# Activo que representa "el mercado" (correlación de cola y regímenes).
# Debe pertenecer a TICKERS.
ACTIVO_REFERENCIA: str = "BTEC.L"

# --- Periodo (datos diarios) -----------------------------------------------
FECHA_INICIO: str = "2019-08-01"
FECHA_FIN: str = "2026-06-20"

# --- Perfil de riesgo e idioma del informe ---------------------------------
# Perfil que domina la recomendación del informe. También se muestran todos los
# niveles para ver cómo cambian los pesos al subir el riesgo.
# Opciones: "conservador" / "moderado" / "agresivo" / "personalizado".
PERFIL_RIESGO: str = "moderado"

# Si PERFIL_RIESGO = "personalizado", fija aquí la volatilidad anual concreta
# buscada (ej. 0.10 = 10%). Para los otros perfiles puede quedarse en None.
VOLATILIDAD_OBJETIVO_ANUAL: float | None = None

# Idioma por defecto del reporte. Al ejecutar el análisis se puede cambiar por
# prompt sin tocar este archivo. Opciones: "es" / "it".
IDIOMA_REPORTE: str = "es"

# --- Restricciones de cartera ----------------------------------------------
SOLO_LARGOS: bool = True                        # True = sin posiciones cortas
PESO_MAXIMO_POR_ACTIVO: float | None = 0.40     # tope por activo; None = sin tope

# --- Rebalanceo y coste (para el backtest walk-forward) --------------------
FRECUENCIA_REBALANCEO: str = "M"                # "M" mensual, "W" semanal, "Q" trimestral
VENTANA_ESTIMACION_DIAS: int = 504              # historia usada para estimar (≈2 años)
COSTE_TRANSACCION_PB: float = 10.0              # coste proporcional sobre la rotación


# ===========================================================================
#  AVANZADO  — rara vez se toca
# ===========================================================================

# Tasa libre de riesgo anual (para Sharpe y Black-Litterman).
TASA_LIBRE_RIESGO_ANUAL: float = 0.0

# Nivel común para VaR, CVaR y Min-CVaR.
NIVEL_CONFIANZA: float = 0.95

# Views de Black-Litterman. VACÍO = NO se calcula Black-Litterman (sin views es
# idéntico a la equiponderada 1/N, así que no se duplica). Para activarlo, añade
# opiniones como:
#   {"activos": {"BTC-USD": 1.0}, "retorno_anual": 0.20, "confianza": 0.6}
VIEWS_BLACK_LITTERMAN: list[dict] = []

# Sensibilidad del etiquetado de regímenes: "conservador" / "estandar" / "sensible".
PERFIL_REGIMEN: str = "estandar"

# Episodios para el stress testing histórico (solo se evalúan si hay cobertura OOS).
VENTANAS_STRESS: dict[str, tuple[str, str]] = {
    "crisis_financiera_2008": ("2008-09-01", "2009-03-31"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "crisis_2022": ("2022-01-03", "2022-10-12"),
}

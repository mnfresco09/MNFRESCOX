"""Configuración del PANEL PORTFOLIO — SOLO datos y parámetros, cero lógica.

Este es el único archivo que edita el usuario. El resto del panel lee de aquí
y nunca al revés. No importa nada de otras capas.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CESTA DE ACTIVOS
# ---------------------------------------------------------------------------
# Tickers tal cual los entiende Yahoo Finance.
#   "BTC-USD" Bitcoin · "GC=F" oro · "^GSPC" S&P 500 · "BZ=F" Brent · "EURUSD=X"
TICKERS: list[str] = ["BTC-USD", "EMIM.AS", "QDV5.DE", "IUSN.DE", "IS3Q.DE", "VWCE.DE", "IS3S.DE"]

# Activo usado para correlación de cola y etiquetado de regímenes.
# Debe pertenecer siempre a TICKERS.
ACTIVO_REFERENCIA: str = "VWCE.DE"

# ---------------------------------------------------------------------------
# PERIODO (datos diarios)
# ---------------------------------------------------------------------------
FECHA_INICIO: str = "2019-08-01"
FECHA_FIN: str = "2026-06-20"

# ---------------------------------------------------------------------------
# REBALANCEO (para el backtest walk-forward)
# ---------------------------------------------------------------------------
# "M" = mensual (por defecto). Otras opciones: "W" semanal, "Q" trimestral.
FRECUENCIA_REBALANCEO: str = "M"
# Días de historia usados para ESTIMAR los pesos en cada paso del walk-forward.
# 504 ≈ 2 años de días bursátiles alineados.
VENTANA_ESTIMACION_DIAS: int = 504

# Coste proporcional aplicado a la rotación en cada rebalanceo.
COSTE_TRANSACCION_PB: float = 10.0

# ---------------------------------------------------------------------------
# RESTRICCIONES DE CARTERA
# ---------------------------------------------------------------------------
SOLO_LARGOS: bool = True          # True = sin posiciones cortas (pesos >= 0)
PESO_MAXIMO_POR_ACTIVO: float | None = 0.40   # tope por activo; None = sin tope

# ---------------------------------------------------------------------------
# RETORNO OBJETIVO (para la cartera de retorno-objetivo de Markowitz)
# ---------------------------------------------------------------------------
RETORNO_OBJETIVO_ANUAL: float = 0.15   # 15% anual

# Tasa libre de riesgo anual (para Sharpe y Black-Litterman).
TASA_LIBRE_RIESGO_ANUAL: float = 0.0

# ---------------------------------------------------------------------------
# ANUALIZACIÓN
# ---------------------------------------------------------------------------
# Los retornos se calculan sobre el calendario ALINEADO (intersección de fechas
# en que cotizan TODOS los activos), que queda limitado por el activo de menor
# frecuencia (la bolsa, ~252 días/año). Por eso se anualiza con 252, no con 365.
DIAS_ANIO: int = 252
MIN_RETORNOS_ANALISIS: int = 252

# Nivel común para VaR, CVaR y Min-CVaR.
NIVEL_CONFIANZA: float = 0.95

# ---------------------------------------------------------------------------
# BLACK-LITTERMAN — views opcionales
# ---------------------------------------------------------------------------
# Lista de "opiniones". Vacía = Black-Litterman cae limpiamente a la cartera de
# mercado (equilibrio implícito), sin views.
# Cada view es un dict:
#   {
#     "activos": {"BTC-USD": 1.0},   # combinación: +1 BTC (view absoluta)
#                                    # o {"^GSPC": 1.0, "BZ=F": -1.0} (relativa)
#     "retorno_anual": 0.20,         # retorno anual esperado de esa combinación
#     "confianza": 0.6,              # 0..1 (1 = certeza total en la view)
#   }
VIEWS_BLACK_LITTERMAN: list[dict] = []

# ---------------------------------------------------------------------------
# REGÍMENES DE MERCADO
# ---------------------------------------------------------------------------
UMBRAL_DRAWDOWN_CRISIS: float = -0.20
UMBRAL_DRAWDOWN_BAJISTA: float = -0.10
VENTANA_VOLATILIDAD: int = 20
VENTANA_MEDIA_LARGA: int = 200
VENTANA_PENDIENTE: int = 20
PERCENTIL_VOLATILIDAD_CRISIS: float = 0.90

# ---------------------------------------------------------------------------
# STRESS TESTING HISTÓRICO
# ---------------------------------------------------------------------------
# Cada episodio se evalúa solo sobre la cobertura walk-forward OOS disponible.
VENTANAS_STRESS: dict[str, tuple[str, str]] = {
    "crisis_financiera_2008": ("2008-09-01", "2009-03-31"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "crisis_2022": ("2022-01-03", "2022-10-12"),
}

# ---------------------------------------------------------------------------
# MONTE CARLO (nube de carteras del reporte)
# ---------------------------------------------------------------------------
N_CARTERAS_MONTECARLO: int = 20_000
SEMILLA: int = 42

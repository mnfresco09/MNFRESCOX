"""Constantes técnicas y presets — fuera de la vista del usuario.

El usuario normal NO necesita tocar esto. Aquí viven los valores que rara vez
cambian (anualización, lambda EWMA, tamaño de la simulación, semilla) y los
presets que traducen una etiqueta sencilla ("estandar") a los números concretos
que consume el motor de régimen.

Sigue siendo solo datos: cero lógica.
"""

from __future__ import annotations

# --- Constantes matemáticas / convenciones ---------------------------------
DIAS_ANIO: int = 252                # factor de anualización estandarizado
MIN_RETORNOS_ANALISIS: int = 252    # mínimo de observaciones para estimar
SEMILLA: int = 42                   # reproducibilidad (Rust acepta seed)

# --- Doble lente de covarianza ---------------------------------------------
# Lente ESTRUCTURAL: Ledoit-Wolf (para optimizar; estable fuera de muestra).
# Lente TÁCTICA: EWMA estilo RiskMetrics (para el riesgo de MAÑANA, T+1).
LAMBDA_EWMA: float = 0.94           # factor de decaimiento RiskMetrics diario
EWMA_MIN_OBS: int = 60              # observaciones mínimas para EWMA fiable

# --- Estimador de retorno (shrinkage conservador) --------------------------
# Encogimiento de la media histórica hacia la gran media (James-Stein-like).
# 0 = media cruda; 1 = todos los activos a la gran media. Conservador alto.
SHRINKAGE_RETORNO: float = 0.50

# --- Frontera eficiente -----------------------------------------------------
# "El 100% de opciones": rejilla densa de la frontera restringida.
N_PUNTOS_FRONTERA: int = 120
# Nube de carteras factibles (fondo del mapa riesgo-retorno; NO Monte Carlo de
# trayectorias). Se dibuja como densidad, no como miles de puntos sueltos.
N_CARTERAS_FACTIBLES: int = 20_000

# --- Simulación futura (motor Rust) ----------------------------------------
N_TRAYECTORIAS_MC: int = 50_000     # trayectorias de bootstrapping (Rust, rayon)
PERCENTILES_FAN: tuple[int, ...] = (5, 25, 50, 75, 95)  # fan chart

# --- Presets de régimen ----------------------------------------------------
# Empaquetan los parámetros del clasificador de régimen en una etiqueta. El
# usuario elige el nombre en config.py (PERFIL_REGIMEN); aquí están los números.
PRESETS_REGIMEN: dict[str, dict] = {
    "conservador": dict(  # marca crisis/bajista antes (más prudente)
        umbral_drawdown_crisis=-0.25, umbral_drawdown_bajista=-0.12,
        ventana_volatilidad=20, ventana_media_larga=200, ventana_pendiente=20,
        percentil_volatilidad_crisis=0.95,
    ),
    "estandar": dict(
        umbral_drawdown_crisis=-0.20, umbral_drawdown_bajista=-0.10,
        ventana_volatilidad=20, ventana_media_larga=200, ventana_pendiente=20,
        percentil_volatilidad_crisis=0.90,
    ),
    "sensible": dict(  # reacciona antes a tramos bajistas (más nervioso)
        umbral_drawdown_crisis=-0.15, umbral_drawdown_bajista=-0.07,
        ventana_volatilidad=15, ventana_media_larga=150, ventana_pendiente=15,
        percentil_volatilidad_crisis=0.85,
    ),
}

# --- Pesos del Score de cartera --------------------------------------------
# Combinan Sharpe ajustado con penalizaciones estandarizadas (z-scores) por
# VaR, drawdown, concentración (HHI) y turnover. Suman 1 en magnitud relativa.
PESOS_SCORE: dict[str, float] = {
    "sharpe": 1.00,          # recompensa (Sharpe táctico ajustado)
    "var": 0.60,             # penaliza VaR 99% forecast
    "cdar": 0.40,            # penaliza Conditional Drawdown at Risk
    "concentracion": 0.30,   # penaliza HHI (concentración de pesos)
    "turnover": 0.15,        # penaliza rotación vs cartera previa
}

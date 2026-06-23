"""Constantes técnicas y presets — fuera de la vista del usuario.

El usuario normal NO necesita tocar esto. Aquí viven los valores que rara vez
cambian (anualización, datos mínimos, tamaño del Monte Carlo, semilla) y los
presets que traducen una etiqueta sencilla ("estandar", "moderado") a los
números concretos que consume el motor.

Sigue siendo solo datos: cero lógica.
"""

from __future__ import annotations

# --- Constantes técnicas ---------------------------------------------------
DIAS_ANIO: int = 252
MIN_RETORNOS_ANALISIS: int = 252
N_CARTERAS_MONTECARLO: int = 20_000
SEMILLA: int = 42

# --- Presets de régimen ----------------------------------------------------
# Empaquetan los 6 parámetros sueltos en una etiqueta. El usuario elige el
# nombre en config.py (PERFIL_REGIMEN); aquí están los números.
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

# --- Perfiles de riesgo -----------------------------------------------------
# Fracción del rango de volatilidad de la frontera (0 = mínima varianza,
# 1 = máximo retorno) que define cada nivel. El informe los muestra todos.
FRACCION_VOL_NIVEL: dict[str, float] = {
    "conservador": 0.15,
    "moderado": 0.50,
    "agresivo": 0.85,
}

# Número de niveles escalonados de la tabla "pesos por nivel de riesgo".
N_NIVELES_RIESGO: int = 5

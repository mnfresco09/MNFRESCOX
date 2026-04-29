# ---------------------------------------------------------------------------
# Parámetros del tipo de salida TRAILING
# Stop Loss de seguridad + trailing stop activado por beneficio.
# Todos los porcentajes se expresan sobre el colateral, no sobre el precio.
# Se evalúa en el timeframe de ejecución más bajo disponible en HISTORICO.
# ---------------------------------------------------------------------------

# Stop Loss de seguridad. Si el precio lo toca antes de activar o ejecutar el
# trailing, el motor cierra por SL.
# Ejemplo: 25 -> cierra si la pérdida llega al 25% del colateral.
EXIT_SL_PCT = 25

# Beneficio mínimo sobre colateral para activar el trailing.
# Ejemplo: 30 -> el trailing empieza cuando el trade gana un 30% del colateral.
EXIT_TRAIL_ACT_PCT = 30

# Distancia del trailing respecto al mejor precio alcanzado tras activarse.
# Ejemplo: 6 -> el stop queda a una distancia equivalente al 6% del colateral.
EXIT_TRAIL_DIST_PCT = 6

# Si True, Optuna busca SL, activación y distancia óptimos.
OPTIMIZAR_SALIDAS = False

# Rangos usados solo cuando OPTIMIZAR_SALIDAS = True.
EXIT_SL_MIN = 5
EXIT_SL_MAX = 50
EXIT_TRAIL_ACT_MIN = 10
EXIT_TRAIL_ACT_MAX = 80
EXIT_TRAIL_DIST_MIN = 2
EXIT_TRAIL_DIST_MAX = 30

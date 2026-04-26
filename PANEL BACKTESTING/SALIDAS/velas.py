# ---------------------------------------------------------------------------
# Parámetros del tipo de salida BARS
# El trade se cierra después de un número fijo de velas.
# ---------------------------------------------------------------------------

# Número máximo de velas que puede durar un trade.
# Se mide en el timeframe de ejecucion mas bajo disponible en HISTORICO,
# no necesariamente en el timeframe de la estrategia.
# Cuando se alcanza, se cierra al precio de cierre de esa vela.
EXIT_VELAS = 48

# Stop Loss de seguridad activo en paralelo.
# Si el precio lo toca antes de llegar a EXIT_VELAS, cierra el trade.
# El primero en cumplirse tiene preferencia.
# Ejemplo: 20 → cierra si la pérdida llega al 20% del colateral.
EXIT_SL_PCT = 20

# Si True, Optuna también buscará el número óptimo de velas y el SL
# en lugar de usar los valores fijos definidos arriba.
OPTIMIZAR_SALIDAS = False

# Rangos usados solo cuando OPTIMIZAR_SALIDAS = True.
EXIT_VELAS_MIN = 6
EXIT_VELAS_MAX = 240
EXIT_SL_MIN = 5
EXIT_SL_MAX = 50

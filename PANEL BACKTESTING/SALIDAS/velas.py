# ---------------------------------------------------------------------------
# Parámetros del tipo de salida BARS
# El trade se cierra después de un número fijo de velas.
# ---------------------------------------------------------------------------

# Número máximo de velas que puede durar un trade.
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

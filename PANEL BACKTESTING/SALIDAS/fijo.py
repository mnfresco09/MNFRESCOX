# ---------------------------------------------------------------------------
# Parámetros del tipo de salida FIXED
# Stop Loss y Take Profit definidos como porcentaje del colateral.
# ---------------------------------------------------------------------------

# Porcentaje de pérdida sobre el colateral que cierra el trade.
# Ejemplo: 20 → el trade se cierra cuando pierde el 20% del colateral.
EXIT_SL_PCT = 20

# Porcentaje de ganancia sobre el colateral que cierra el trade.
# Ejemplo: 40 → el trade se cierra cuando gana el 40% del colateral.
EXIT_TP_PCT = 40

# Si True, Optuna también buscará los valores óptimos de SL y TP
# en lugar de usar los valores fijos definidos arriba.
OPTIMIZAR_SALIDAS = False

# Rangos usados solo cuando OPTIMIZAR_SALIDAS = True.
EXIT_SL_MIN = 5
EXIT_SL_MAX = 50
EXIT_TP_MIN = 10
EXIT_TP_MAX = 120

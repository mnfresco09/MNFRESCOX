# ---------------------------------------------------------------------------
# Parametros del tipo de salida CUSTOM
# La condicion de cierre la define cada estrategia en generar_salidas().
# ---------------------------------------------------------------------------

# Stop Loss de seguridad activo en paralelo.
# Si el precio toca el SL antes de la salida custom, el motor cierra por SL.
# Ejemplo: 20 -> cierra si la perdida llega al 20% del colateral.
EXIT_SL_PCT = 20

# Si True, Optuna buscara tambien el SL de seguridad optimo.
OPTIMIZAR_SALIDAS = False

# Rangos usados solo cuando OPTIMIZAR_SALIDAS = True.
EXIT_SL_MIN = 5
EXIT_SL_MAX = 50

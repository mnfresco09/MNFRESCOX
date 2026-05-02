# ---------------------------------------------------------------------------
# Parametros de PARIDAD DE RIESGO
# ---------------------------------------------------------------------------
# Este archivo contiene los valores editables de la salida dinamica por
# volatilidad. `NUCLEO/paridad_riesgo.py` solo calcula; no debe almacenar
# configuracion operativa.
#
# Cuando USAR_PARIDAD_RIESGO=True en CONFIGURACION/config.py:
# - El SL, TP y trailing dejan de ser porcentajes fijos de SALIDAS/fijo.py,
#   SALIDAS/velas.py o SALIDAS/trailing.py.
# - Optuna busca multiplos de volatilidad EWMA cuando
#   OPTIMIZAR_PARIDAD_RIESGO=True.
# - El motor calcula el apalancamiento por trade para que el SL represente
#   RIESGO_MAXIMO_PCT del colateral.
# ---------------------------------------------------------------------------

# Riesgo maximo de perdida por trade, expresado como % del colateral.
# Ejemplo: 5.0 significa que un SL debe perder aproximadamente el 5% del
# colateral usado en esa operacion.
RIESGO_MAXIMO_PCT = 20.0

# Rango que Optuna puede explorar para RIESGO_MAXIMO_PCT.
RIESGO_MAXIMO_MIN = 15.0
RIESGO_MAXIMO_MAX = 35.0

# Vida media de la volatilidad EWMA, medida en velas del timeframe de la
# estrategia. Mas bajo = reacciona antes; mas alto = volatilidad mas estable.
VOL_HALFLIFE = 50

# Rango que Optuna puede explorar para VOL_HALFLIFE.
VOL_HALFLIFE_MIN = 30
VOL_HALFLIFE_MAX = 75

# Multiplo de volatilidad EWMA usado para colocar el Stop Loss.
# Distancia SL real = volatilidad_ewma * SL_EWMA_MULT.
SL_EWMA_MULT = 2.0

# Rango que Optuna puede explorar para SL_EWMA_MULT.
SL_EWMA_MULT_MIN = 2.0
SL_EWMA_MULT_MAX = 6.0

# Multiplo de volatilidad EWMA usado como Take Profit en salida FIXED.
# Solo se usa cuando EXIT_TYPE="FIXED" y USAR_PARIDAD_RIESGO=True.
TP_EWMA_MULT = 4.0

# Rango que Optuna puede explorar para TP_EWMA_MULT.
TP_EWMA_MULT_MIN = 1.0
TP_EWMA_MULT_MAX = 20.0

# Multiplo de volatilidad EWMA necesario para activar el trailing.
# Solo se usa cuando EXIT_TYPE="TRAILING" y USAR_PARIDAD_RIESGO=True.
TRAIL_ACT_EWMA_MULT = 3.0

# Rango que Optuna puede explorar para TRAIL_ACT_EWMA_MULT.
TRAIL_ACT_EWMA_MULT_MIN = 6.0
TRAIL_ACT_EWMA_MULT_MAX = 18.0

# Distancia del trailing respecto al mejor precio alcanzado, expresada como
# multiplo de volatilidad EWMA. Debe quedar por debajo de TRAIL_ACT_EWMA_MULT.
TRAIL_DIST_EWMA_MULT = 1.0

# Rango que Optuna puede explorar para TRAIL_DIST_EWMA_MULT.
TRAIL_DIST_EWMA_MULT_MIN = 0.50
TRAIL_DIST_EWMA_MULT_MAX = 6.0

# Apalancamiento minimo permitido cuando la paridad calcula el tamano del
# trade. Si el apalancamiento bruto queda por debajo y
# SKIP_SI_APALANCAMIENTO_MENOR_MIN=True, el trade se descarta.
PARIDAD_APALANCAMIENTO_MIN = 0.1

# Apalancamiento maximo permitido. Si la formula pide mas, se limita a este
# valor para evitar exposicion excesiva.
PARIDAD_APALANCAMIENTO_MAX = 50.0

# True: descarta senales cuando la volatilidad exige un apalancamiento menor
# que PARIDAD_APALANCAMIENTO_MIN.
# False: fuerza el minimo aunque eso aumente el riesgo real sobre el colateral.
SKIP_SI_APALANCAMIENTO_MENOR_MIN = False

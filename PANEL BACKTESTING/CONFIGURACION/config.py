from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
CARPETA_HISTORICO = RAIZ / "HISTORICO"
CARPETA_RESULTADOS = RAIZ / "RESULTADOS"
# Fase 0 — registro de experimentos. Carpeta y base de datos SQLite donde queda
# la traza de cada trial de cada run (alimenta el conteo real de N para el DSR).
CARPETA_REGISTRO_EXPERIMENTOS = RAIZ / "REGISTRO_EXPERIMENTOS"
BD_EXPERIMENTOS = CARPETA_REGISTRO_EXPERIMENTOS / "experimentos.db"

# ---------------------------------------------------------------------------
# ACTIVOS Y DATOS "BTC", "GOLD", "BRENT", "EURUSD", "SP500"
# ---------------------------------------------------------------------------
ACTIVOS = ["BTC"]   # un activo o lista
FORMATO_DATOS = "parquet"          # "feather" | "parquet" | "csv"

# True = activo continuo 24/7: cualquier hueco temporal es error.
# False = activo con cierre de mercado: se permiten saltos entre sesiones
#         siempre conservando orden, duplicados, OHLC y trazabilidad.
# Nota: el dataset horario de BTC tiene algunos huecos puntuales, por eso va a
# False; si en el futuro se carga un BTC 1m sin huecos, puede volver a True.
MERCADO_24_7 = {
    "BTC": False,
    "GOLD": False,
    "BRENT": False,
    "EURUSD": False,
    "SP500": False,
}

# ---------------------------------------------------------------------------
# TIMEFRAMES
# ---------------------------------------------------------------------------
# El sistema resamplea hacia arriba desde el timeframe base disponible.
# La base actual de los datos es 1h, asi que el minimo usable es "1h".
# Opciones validas con base 1h: "1h" "4h" "1d".
TIMEFRAMES = ["1h"]

# ---------------------------------------------------------------------------
# FECHAS 2020-01-01 hasta 2025-12-31
# ---------------------------------------------------------------------------
FECHA_INICIO = "2021-01-01"
FECHA_FIN    = "2024-12-31"

# ---------------------------------------------------------------------------
# MODO DE OPERACIÓN Y HOLDOUT BLOQUEADO  (Fase 0 — la disciplina innegociable)
# ---------------------------------------------------------------------------
# El holdout bloqueado es el tramo MÁS RECIENTE de datos que NUNCA entra en
# ninguna optimización ni validación. Se evalúa UNA SOLA VEZ, al final, con los
# parámetros ya congelados (Puerta 6 del protocolo). Si "echas un vistazo" antes
# de tiempo, ese tramo queda quemado para siempre y toda la estadística de
# arriba se vuelve teatro.
#
# MODO controla FÍSICAMENTE qué datos llegan al pipeline:
#   "investigacion"   → el cargador EXCLUYE el holdout del DataFrame. Es
#                       imposible que el tramo bloqueado entre en optimización
#                       o validación. Es el modo de trabajo por defecto.
#   "veredicto_final" → desbloquea el dataset completo para el examen final.
#                       Úsalo SOLO cuando los parámetros ya están fijados y vas
#                       a disparar la única evaluación sobre el holdout.
MODO = "investigacion"   # "investigacion" | "veredicto_final"

# Inicio del holdout bloqueado (inclusive, AAAA-MM-DD). Todo lo anterior a esta
# fecha es TRAIN/VALIDATION; desde esta fecha (incluida) hasta FECHA_FIN es el
# holdout. Debe caer estrictamente dentro de (FECHA_INICIO, FECHA_FIN].
# Recomendado: los últimos 12-18 meses. Aquí, 2024 completo (~12 meses) sobre un
# histórico 2021-2024, dejando 3 años para TRAIN/VALIDATION.
HOLDOUT_INICIO = "2024-01-01"

# ---------------------------------------------------------------------------
# ESTRATEGIAS
# ---------------------------------------------------------------------------
# ID numérico, lista de IDs, o "all" para ejecutar todas.
ESTRATEGIA_ID = 1

# ---------------------------------------------------------------------------
# CAPITAL Y COMISIONES
# ---------------------------------------------------------------------------
SALDO_INICIAL          = 10_000     # Capital inicial en USD
SALDO_USADO_POR_TRADE  = 500        # Colateral por operación en USD
APALANCAMIENTO         = 8        # Multiplicador sobre el colateral
SALDO_MINIMO_OPERATIVO = 1_000      # El backtest para si el saldo cae aquí
COMISION_PCT           = 0.0005     # 0.05% por operación (ej. Binance taker)
COMISION_LADOS         = 2          # 1 = solo apertura | 2 = apertura y cierre

# ---------------------------------------------------------------------------
# SALIDAS
# ---------------------------------------------------------------------------
# "FIXED" → Stop Loss y Take Profit fijos       → parámetros en SALIDAS/fijo.py
# "BARS"  → Cierre por número máximo de velas   → parámetros en SALIDAS/velas.py
# "TRAILING" → SL de seguridad + trailing stop  → parámetros en SALIDAS/trailing.py
# "CUSTOM"→ Cierre por generar_salidas()         → parámetros en SALIDAS/personalizada.py
# "ALL"   → Ejecuta todos por separado y guarda cada resultado
EXIT_TYPE = "BARS"

# ---------------------------------------------------------------------------
# OPTIMIZACIÓN (OPTUNA)
# ---------------------------------------------------------------------------
# Potencias de 2 recomendadas para QMC: 64, 128, 256, 512
N_TRIALS = 3000

# "QMC"    → Exploración uniforme (Sobol). Recomendado para campañas grandes.
# "TPE"    → Guiado por resultados anteriores; su coste crece con el histórico.
# "HYBRID" → QMC primera mitad + TPE segunda mitad; útil sólo en campañas moderadas.
OPTUNA_SAMPLER = "QMC"

# ---------------------------------------------------------------------------
# FUNCIÓN DE SCORE (lo que Optuna maximiza)
# ---------------------------------------------------------------------------
# Métrica que define qué es un "buen" trial. Todas tienen base estadística:
# "PSR"    → Probabilistic Sharpe Ratio (recomendado). Probabilidad (0..1) de que
#            el Sharpe real sea > 0. Penaliza por sí solo las muestras pequeñas.
# "SHARPE" → Sharpe anualizado (rentabilidad ajustada por volatilidad).
# "CALMAR" → CAGR / Max Drawdown (rentabilidad anual frente al peor desplome).
# "ROI"    → Retorno total simple (sin ajuste de riesgo; el más básico).
FUNCION_SCORE = "PSR"

# Mínimo de operaciones para puntuar un trial (0 = sin mínimo).
# Por debajo, el score es 0: evita premiar estrategias con muy pocas
# operaciones, cuyas métricas son ruido estadístico. PSR ya penaliza la
# muestra pequeña, pero este filtro duro añade una garantía explícita.
MIN_TRADES_SCORE = 30

# Penalizaciones de robustez sobre el score (Fase 4). Con factor 0 no tienen
# efecto (comportamiento por defecto idéntico al actual). Súbelos para empujar
# la optimización hacia menos turnover y menos parámetros libres.
#   score_penalizado = score
#       - TURNOVER_FACTOR · max(0, n_trades - TURNOVER_OBJETIVO) / TURNOVER_OBJETIVO
#       - COMPLEJIDAD_FACTOR · n_parametros_libres
TURNOVER_OBJETIVO              = 100
PENALIZACION_TURNOVER_FACTOR   = 0.0
PENALIZACION_COMPLEJIDAD_FACTOR = 0.0

# Optimización multiobjetivo (Fase 4). Con False, una única búsqueda escalar
# (maximiza FUNCION_SCORE) y el "mejor" es el de mayor score. Con True, la única
# búsqueda usa NSGA-II y saca un FRENTE DE PARETO de (PSR ↑, max drawdown ↓,
# turnover ↓); entonces NO se elige el de score máximo, sino la configuración de
# MESETA: la que vive en la región más estable del espacio de parámetros (la más
# robusta), no el pico estrecho.
OPTUNA_MULTIOBJETIVO = True
# Nº de vecinos en el espacio de parámetros para medir la meseta al elegir del
# frente de Pareto (solo aplica con OPTUNA_MULTIOBJETIVO=True).
MESETA_VECINOS = 7

# True  = usa las semillas configuradas y permite reproducibilidad.
# False = ignora las semillas y cada ejecución explora caminos aleatorios.
USAR_SEED = False

# Entero obligatorio cuando USAR_SEED = True.
OPTUNA_SEED = 42

# ---------------------------------------------------------------------------
# VALIDACIÓN FUERA DE MUESTRA  (Fases 2-7: CPCV, WFA, DSR, robustez, informe)
# ---------------------------------------------------------------------------
# Tras la optimización in-sample de cada combinación, el sistema valida la mejor
# configuración fuera de muestra y emite un veredicto 🟢/🟡/🔴 con los umbrales
# fijados a priori, además del informe institucional unificado.
#
# Coste: CPCV reoptimiza por cada fold. Con VALIDACION_N_GRUPOS=6 y K=2 son 15
# folds; cada uno lanza VALIDACION_N_TRIALS optimizaciones. Mantén N_TRIALS de
# validación moderado. La validación NO es compatible con perturbaciones activas
# (se omite con aviso si PERTURBACIONES_ACTIVAS=True).
VALIDACION_ACTIVA       = True

VALIDACION_N_TRIALS     = 100      # trials de Optuna por fold (< N_TRIALS global)
VALIDACION_N_GRUPOS     = 6        # N grupos temporales de CPCV
VALIDACION_K            = 2        # k grupos como test → C(N,k) combinaciones
VALIDACION_EMBARGO      = 0.01     # fracción de velas de embargo tras cada test
VALIDACION_DURACION_TRADE = 1      # nº máx. de velas que abarca un trade (purge)

VALIDACION_WFA_ACTIVA   = True     # Walk-Forward como complemento de CPCV
VALIDACION_WFA_VENTANAS = 5
VALIDACION_WFA_FRACCION = 0.15     # tamaño de cada tramo de test (fracción)
#   Restricción: (VALIDACION_WFA_VENTANAS + 1) · VALIDACION_WFA_FRACCION <= 1.
VALIDACION_WFA_ANCHORED = False    # False=rolling, True=anchored

VALIDACION_BOOTSTRAP_ITER   = 10_000   # remuestreos del bootstrap de trades
VALIDACION_BOOTSTRAP_BLOQUE = 1        # 1=i.i.d.; >1=block bootstrap
VALIDACION_SHARPE_ANUAL_OBJETIVO = 1.0 # objetivo para el chequeo MinBTL

VALIDACION_NULA_ITER = 50   # nº de estrategias nulas (entradas aleatorias) para
#                             el control de laboratorio; 0 lo desactiva.

# ---------------------------------------------------------------------------
# PERTURBACIONES
# ---------------------------------------------------------------------------
# False = todos los trials ven el Parquet original.
# True  = cada trial ve un camino alternativo plausible, generado en memoria.
PERTURBACIONES_ACTIVAS = False
# Entero obligatorio cuando USAR_SEED = True y PERTURBACIONES_ACTIVAS = True.
PERTURBACIONES_SEED = 42

# Parametros de la tabla automatica de microestructura.
# Los valores de la tabla se calculan desde el Parquet cargado.
GRANULARIDAD_CUBOS = 0.005
PERCENTIL_TABLA = 0.10
# True valida todas las invariantes OHLCV/order-flow despues de cada trial.
# Usarlo para desarrollo/auditoria; en optimizacion grande añade coste O(n).
VALIDAR_PERTURBACIONES = False

# ---------------------------------------------------------------------------
# PARALELISMO
# ---------------------------------------------------------------------------
# -1 → todos los cores
# -2 → todos los cores menos uno (recomendado)
#  1 → secuencial (útil para depurar)
N_JOBS = -2

# Las perturbaciones procesan el histórico base completo por trial. Este tope
# evita saturar memoria/cache cuando N_JOBS apunta a todos los cores.
PERTURBACIONES_MAX_JOBS = 4

# ---------------------------------------------------------------------------
# RESULTADOS Y REPORTING
# ---------------------------------------------------------------------------
USAR_EXCEL  = True

# La carpeta final de cada combinación (ESTRATEGIA/TIMEFRAME/SALIDA/ACTIVO)
# contiene exactamente los ficheros del último run y se reemplaza al relanzar la
# misma combinación; no hay histórico ni rotación. La traza completa vive en la
# base de datos de experimentos (REGISTRO_EXPERIMENTOS/).

# "all"    → muestra todo el período del trial en el gráfico HTML
# "3m"     → muestra los últimos 3 meses
# "custom" → usa GRAFICA_DESDE y GRAFICA_HASTA
GRAFICA_RANGO = "12m"
GRAFICA_DESDE = "2024-01-01"    # Solo si GRAFICA_RANGO = "custom"
GRAFICA_HASTA = "2024-12-31"    # Solo si GRAFICA_RANGO = "custom"

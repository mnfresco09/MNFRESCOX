from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
CARPETA_HISTORICO = RAIZ / "HISTORICO"
CARPETA_RESULTADOS = RAIZ / "RESULTADOS"

# ---------------------------------------------------------------------------
# ACTIVOS Y DATOS
# ---------------------------------------------------------------------------
ACTIVOS = ["BTC"]          # Un activo o lista: ["BTC", "GOLD"]
FORMATO_DATOS = "feather"          # "feather" | "parquet" | "csv"

# True = activo continuo 24/7: cualquier hueco temporal es error.
# False = activo con cierre de mercado: se permiten saltos entre sesiones
#         siempre conservando orden, duplicados, OHLC y trazabilidad.
MERCADO_24_7 = {
    "BTC": True,
    "GOLD": False,
}

# ---------------------------------------------------------------------------
# TIMEFRAMES
# ---------------------------------------------------------------------------
# El sistema resamplea desde 1m automáticamente.
# Opciones: "1m" "5m" "15m" "30m" "1h" "4h" "1d"
TIMEFRAMES = ["15m"]

# ---------------------------------------------------------------------------
# FECHAS 2020-01-01 hasta 2025-12-31
# ---------------------------------------------------------------------------
FECHA_INICIO = "2022-01-01"
FECHA_FIN    = "2024-12-31"

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
APALANCAMIENTO         = 10         # Multiplicador sobre el colateral
SALDO_MINIMO_OPERATIVO = 1_000      # El backtest para si el saldo cae aquí
COMISION_PCT           = 0.0005     # 0.05% por operación (ej. Binance taker)
COMISION_LADOS         = 2          # 1 = solo apertura | 2 = apertura y cierre

# ---------------------------------------------------------------------------
# SALIDAS
# ---------------------------------------------------------------------------
# "FIXED" → Stop Loss y Take Profit fijos       → parámetros en SALIDAS/fijo.py
# "BARS"  → Cierre por número máximo de velas   → parámetros en SALIDAS/velas.py
# "CUSTOM"→ Cierre por generar_salidas()         → parámetros en SALIDAS/personalizada.py
# "ALL"   → Ejecuta todos por separado y guarda cada resultado
EXIT_TYPE = "FIXED"

# ---------------------------------------------------------------------------
# OPTIMIZACIÓN (OPTUNA)
# ---------------------------------------------------------------------------
# Potencias de 2 recomendadas para QMC: 64, 128, 256, 512
N_TRIALS = 64

# "QMC"    → Exploración uniforme (secuencias Sobol). Bueno para primera pasada.
# "TPE"    → Guiado por resultados anteriores. Bueno para refinar.
# "HYBRID" → QMC primera mitad + TPE segunda mitad. Recomendado por defecto.
OPTUNA_SAMPLER = "HYBRID"

# None → resultados distintos en cada ejecución
# Número entero → reproducibilidad exacta
OPTUNA_SEED = 42

# ---------------------------------------------------------------------------
# PARALELISMO
# ---------------------------------------------------------------------------
# -1 → todos los cores
# -2 → todos los cores menos uno (recomendado)
#  1 → secuencial (útil para depurar)
N_JOBS = -2

# ---------------------------------------------------------------------------
# RESULTADOS Y REPORTING
# ---------------------------------------------------------------------------
USAR_EXCEL  = True
MAX_PLOTS   = 5         # HTMLs a generar (los mejores N por score)
MAX_ARCHIVOS = 20       # Máximo de archivos por carpeta antes de rotar los más viejos

# "all"    → muestra todo el período del trial en el gráfico HTML
# "3m"     → muestra los últimos 3 meses
# "custom" → usa GRAFICA_DESDE y GRAFICA_HASTA
GRAFICA_RANGO = "3m"
GRAFICA_DESDE = "2024-01-01"    # Solo si GRAFICA_RANGO = "custom"
GRAFICA_HASTA = "2024-12-31"    # Solo si GRAFICA_RANGO = "custom"

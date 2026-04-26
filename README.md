# PLATAFORMA DE INVERSIÓN — MNFRESCO

Sistema modular para un trader individual que integra backtesting, seguimiento de capital, vigilancia de activos, análisis técnico y contexto de mercado.

**Regla de oro de la arquitectura:** cada panel vive en su propia carpeta, es 100 % independiente y no sabe nada de los demás. Si un panel falla, los otros siguen funcionando sin tocarlos.

---

## Orden de construcción

| Fase | Panel | Estado |
|------|-------|--------|
| 1–7 | PANEL BACKTESTING | **En construcción — prioridad absoluta** |
| 8 | PANEL_PORTFOLIO | Pendiente |
| 8 | PANEL_SCANNER | Pendiente |
| 8 | PANEL_ANALISIS | Pendiente |
| 8 | PANEL_NOTICIAS | Pendiente |

---

## Estructura de la raíz

```
MNFRESCO/
│
├── run.py                  ← Punto de entrada único. Ejecuta el backtesting completo.
├── README.md               ← Este archivo.
│
├── PANEL BACKTESTING/      ← Laboratorio de estrategias. Prioridad nº 1.
├── PANEL_PORTFOLIO/        ← Seguimiento del capital real. (Pendiente)
├── PANEL_SCANNER/          ← Vigilancia automática de activos. (Pendiente)
├── PANEL_ANALISIS/         ← Gráfico interactivo y decisiones de trading. (Pendiente)
└── PANEL_NOTICIAS/         ← Noticias, calendario y notas personales. (Pendiente)
```

---

## PANEL BACKTESTING

El laboratorio donde se prueban las estrategias en datos históricos antes de arriesgar dinero real. Le dices qué activo, qué período, qué estrategia y qué parámetros quieres probar, y el sistema simula qué habría pasado operando con esa estrategia. Lo hace miles de veces variando parámetros para encontrar cuáles funcionan mejor.

**Punto de entrada:** `run.py` en la raíz del proyecto.
**Configuración:** `PANEL BACKTESTING/CONFIGURACION/config.py` — el único archivo que el usuario edita.

### Árbol completo

```
PANEL BACKTESTING/
│
├── CONFIGURACION/                  ← Todo lo relacionado con la configuración de un run.
│   ├── config.py                   ← El ÚNICO archivo que el usuario edita.
│   │                                 Contiene: activo, fechas, timeframe, capital,
│   │                                 comisiones, tipo de salida y parámetros de Optuna.
│   └── validador_config.py         ← Comprueba config.py antes de arrancar el run:
│                                     tipos correctos, fechas coherentes, que el activo
│                                     existe en HISTORICO/, que EXIT_TYPE es válido, etc.
│                                     Si algo falla: para y explica exactamente qué.
│
├── NUCLEO/                         ← La base de la que depende todo lo demás.
│   ├── base_estrategia.py          ← Clase abstracta madre de todas las estrategias.
│   │                                 Define el contrato: ID, nombre, espacio de
│   │                                 búsqueda, señales y salidas CUSTOM. Incluye
│   │                                 helpers de indicadores (SMA, EMA, RSI, ATR,
│   │                                 Bollinger).
│   ├── tipos.py                    ← Tipos de datos compartidos internamente:
│   │                                 TradeResult, Signal, OHLCVFrame, etc.
│   └── registro.py                 ← Auto-descubrimiento de estrategias. Escanea
│                                     ESTRATEGIAS/ al arrancar y registra todo lo que
│                                     encuentra. No hay que registrar nada a mano.
│
├── DATOS/                          ← Todo lo relacionado con datos de precio.
│   │                                 Solo código. Los archivos reales van en HISTORICO/.
│   ├── cargador.py                 ← Lee archivos Parquet, CSV y Feather desde
│   │                                 HISTORICO/. Usa el menor timeframe disponible
│   │                                 como base y aplica el filtro de fechas de config.
│   ├── validador.py                ← Comprueba antes de cualquier run:
│   │                                 columnas mínimas (timestamp, open, high, low,
│   │                                 close), orden cronológico, huecos, duplicados
│   │                                 y columnas extra que requiere la estrategia.
│   │                                 Si algo falla: para y explica exactamente qué.
│   └── resampleo.py                ← Construye timeframes mayores desde el timeframe base.
│                                     Reglas: open=primera, high=max, low=min,
│                                     close=última, volume=suma. Solo permite ir
│                                     hacia timeframes más grandes, nunca más pequeños.
│
├── HISTORICO/                      ← Archivos de datos de mercado del usuario.
│   │                                 Formato preferido: Parquet.
│   │                                 Convención de nombres: ACTIVO_TIMEFRAME
│   │                                 Ejemplos: BTC_ohlcv_1m.feather, GOLD_ohlcv_5m.parquet
│   └── .gitkeep                    ← Carpeta vacía en repo. Los datos van fuera de git.
│
├── MOTOR/                          ← Motor de simulación escrito en Rust.
│   │                                 Se compila una vez (cargo build --release)
│   │                                 y se usa desde Python como una librería normal
│   │                                 gracias a PyO3.
│   ├── src/
│   │   ├── lib.rs                  ← Punto de entrada del crate. Define los bindings
│   │   │                             PyO3 que expone al Python: función simulate_trades().
│   │   ├── simulador.rs            ← Loop principal vela a vela. Detecta señales,
│   │   │                             abre trades, los sigue hasta la condición de salida,
│   │   │                             calcula resultado y actualiza saldo.
│   │   │                             REGLA CRÍTICA: la entrada ocurre siempre en el
│   │   │                             open de la vela N+1, nunca en la vela de la señal.
│   │   ├── tipos.rs                ← Structs Rust: TradeInput, TradeResult, SimConfig.
│   │   └── capital.rs              ← Gestión del capital: colateral por trade,
│   │                                 apalancamiento, cálculo de comisiones,
│   │                                 saldo mínimo operativo.
│   └── Cargo.toml                  ← Manifiesto del proyecto Rust. Define dependencias
│                                     (PyO3) y el tipo de crate (cdylib).
│
├── SALIDAS/                        ← Un archivo por tipo de salida. Añadir un nuevo
│   │                                 tipo = añadir un archivo aquí. Nada más.
│   ├── fijo.py                     ← Tipo FIXED: cierra cuando el precio toca el
│   │                                 Stop Loss o el Take Profit definidos como % del
│   │                                 colateral. Es el tipo por defecto.
│   ├── velas.py                    ← Tipo BARS: cierra después de N velas
│   │                                 independientemente del precio. Compatible con
│   │                                 Stop Loss como red de seguridad.
│   └── personalizada.py            ← Tipo CUSTOM: la estrategia define sus propias
│                                     condiciones de salida. El TP se desactiva;
│                                     el SL permanece como protección de emergencia.
│
├── OPTIMIZACION/                   ← Motor de búsqueda de parámetros con Optuna.
│   ├── runner.py                   ← Coordina el bucle de trials. Gestiona el
│   │                                 paralelismo (N_JOBS workers). Procesa activos
│   │                                 y timeframes en secuencial para controlar RAM.
│   ├── metricas.py                 ← Calcula métricas derivadas del resultado:
│   │                                 profit factor, Sharpe, drawdown, ROI, etc.
│   ├── samplers.py                 ← Crea el sampler según config:
│   │                                 QMC (exploración uniforme, secuencias Sobol),
│   │                                 TPE (guiado por resultados anteriores),
│   │                                 HYBRID (QMC primera mitad, TPE segunda mitad).
│   └── puntuacion.py               ← Función de scoring que Optuna maximiza.
│                                     Versión inicial: score = ROI.
│                                     Diseñada para sustituirse sin tocar nada más.
│
├── REPORTES/                       ← Un reporter por tipo de output. Independientes.
│   ├── rich.py                     ← Muestra con Rich el progreso trial a trial:
│   │                                 panel institucional con performance, finanzas,
│   │                                 parámetros y estado del mejor trial.
│   ├── excel.py                    ← Genera EXCEL/RESUMEN *.xlsx
│   │                                 con todos los trials y EXCEL/TRIAL * - *.xlsx
│   │                                 para los 5 mejores scores.
│   ├── html.py                     ← Genera GRAFICA/TRIAL * - *.html
│   │                                 para los mejores trials configurados.
│                                     Gráfico de velas + indicadores + entradas/salidas
│                                     con flechas. Usa Lightweight Charts (TradingView).
│                                     No necesita internet. Pesa < 500 KB por archivo.
│   └── persistencia.py             ← Guarda CSV/JSON, verifica que no se pierdan
│                                     trials/trades/equity en
│                                     RESULTADOS/ESTRATEGIA/SALIDA/ACTIVO/TF/DATOS/RUN_*
│                                     y rota runs antiguos.
│
├── ESTRATEGIAS/                    ← Las estrategias del usuario.
│   │                                 REGLA: un archivo = una estrategia.
│   │                                 El sistema las descubre automáticamente al arrancar.
│   │                                 No hay que registrar nada en ningún archivo central.
│   │                                 CONTRATO obligatorio de cada estrategia:
│   │                                   - ID numérico único
│   │                                   - Nombre
│   │                                   - espacio_busqueda(): define parámetros y rangos
│   │                                   - generar_señales(): recibe datos, devuelve señales
│   │                                   - generar_salidas(): obligatorio para CUSTOM
│   ├── GUIA_ESTRATEGIAS.md         ← Contrato serio para crear estrategias:
│   │                                 anti-lookahead, salidas custom, checklist y auditoría.
│   ├── rsi_reversion.py            ← Estrategia de reversión a la media por cruce de RSI.
│   └── ema_tendencia.py            ← Estrategia tendencial por cruce de EMAs.
│
└── RESULTADOS/                     ← Outputs de cada run.
    │                                 Estructura: ACTIVO / TIMEFRAME / ESTRATEGIA / EXIT_TYPE /
    │                                 Ejemplo: RESULTADOS/BTC/1h/RSI_REVERSION/FIXED/
    │                                 Cada carpeta contiene su Excel y sus HTMLs.
    │                                 El sistema elimina automáticamente los más antiguos
    │                                 cuando se supera MAX_ARCHIVOS.
    └── .gitkeep
```

---

## Parámetros de config.py

Todos los parámetros que el usuario puede tocar, agrupados por categoría:

| Categoría | Parámetro | Descripción |
|-----------|-----------|-------------|
| **Datos** | `ACTIVOS` | Activo o lista: `'BTC'` o `['BTC', 'GOLD']` |
| | `FORMATO_DATOS` | `'parquet'` (recomendado), `'csv'` o `'feather'` |
| | `MERCADO_24_7` | Dict por activo. `True` exige continuidad total; `False` permite cierres de mercado sin rellenar datos. |
| **Timeframes** | `TIMEFRAMES` | Timeframe o lista: `'1h'` o `['1h', '4h']` |
| **Fechas** | `FECHA_INICIO` | Inicio del período: `'AAAA-MM-DD'` |
| | `FECHA_FIN` | Fin del período: `'AAAA-MM-DD'` |
| **Estrategia** | `ESTRATEGIA_ID` | ID numérico, lista de IDs o `'all'` |
| **Capital** | `SALDO_INICIAL` | Capital inicial en dólares |
| | `SALDO_USADO_POR_TRADE` | Colateral por operación en dólares |
| | `APALANCAMIENTO` | Multiplicador sobre el colateral |
| | `SALDO_MINIMO_OPERATIVO` | Si el saldo cae aquí, el backtest para |
| | `COMISION_PCT` | Comisión como decimal: `0.0005` = 0.05% |
| | `COMISION_LADOS` | `1` = solo apertura. `2` = apertura y cierre |
| **Salidas** | `EXIT_TYPE` | `'FIXED'`, `'BARS'`, `'CUSTOM'` o `'ALL'` |
| | `SALIDAS/fijo.py` | Parámetros de `FIXED`: SL, TP y rangos opcionales |
| | `SALIDAS/velas.py` | Parámetros de `BARS`: velas máximas, SL y rangos opcionales |
| | `SALIDAS/personalizada.py` | Parámetros de `CUSTOM`: SL de seguridad y rango opcional |
| **Optuna** | `N_TRIALS` | Nº de combinaciones a probar. Usar potencias de 2 |
| | `OPTUNA_SAMPLER` | `'QMC'`, `'TPE'` o `'HYBRID'` (recomendado) |
| | `OPTUNA_SEED` | Semilla para reproducibilidad. `None` = aleatorio |
| **Paralelismo** | `N_JOBS` | Workers paralelos. `-2` = todos los cores menos uno |
| **Resultados** | `USAR_EXCEL` | `True` genera el Excel al terminar |
| | `MAX_PLOTS` | Nº de HTMLs a generar (los mejores N por score) |
| | `MAX_ARCHIVOS` | Máximo de archivos por carpeta antes de rotar |
| | `GRAFICA_RANGO` | `'all'`, `'3m'` (últimos 3 meses) o `'custom'` |

---

## Tecnología

| Componente | Herramienta | Por qué |
|------------|-------------|---------|
| Motor de simulación | **Rust + PyO3** | 100–1000x más rápido que Python en el loop vela a vela |
| Transformación de datos | **Polars** | 5–20x más rápido que Pandas. Columnar, multi-core, lazy |
| Optimización | **Optuna** | Estándar de facto. QMC, TPE, reproducible, SQLite nativo |
| Formato de datos | **Parquet** | 5–50x más rápido que CSV. Compresión eficiente en columnas |
| Terminal | **Rich** | Tablas y paneles en tiempo real sin complejidad de GUI |
| Resultados tabulares | **Excel** | Cualquiera sabe abrirlo, filtrarlo y añadir gráficos propios |
| Gráficos interactivos | **HTML + Lightweight Charts** | Standalone, sin servidor, < 500 KB, interactivo |

---

## Flujo de un trade (cómo funciona el motor)

1. La estrategia detecta señal en la **vela N**
2. El motor anota la señal pero **no entra todavía**
3. En la **vela N+1**, entra al precio de **apertura** — el único momento realista
4. El motor vigila la posición vela a vela hacia adelante
5. En cada vela comprueba: ¿stop loss tocado? ¿take profit? ¿señal de salida? ¿máximo de velas?
6. El primero en cumplirse cierra el trade al precio correspondiente
7. Se calcula resultado, se descuentan comisiones, se actualiza el saldo
8. Si hay señales mientras el trade está abierto, se ignoran

**Regla absoluta:** nunca se usa el precio de cierre de la vela actual para decidir entrar en esa misma vela. El sistema lo garantiza por construcción en el motor Rust, no por disciplina del usuario.

---

## Fases de construcción de PANEL BACKTESTING

| Fase | Objetivo | Criterio de avance |
|------|----------|--------------------|
| 1 | Carga + validación + resampleo funcionando | Comparar velas resampleadas manualmente |
| 2 | Estrategia genera señales correctas y verificables | Señales en velas correctas, no en las siguientes |
| 3 | Motor Rust compila y simulate_trades() funciona con SL/TP | Entradas siempre en open de vela N+1 |
| 4 | Trial completo de punta a punta sin Optuna | Resultados con sentido para parámetros fijos |
| 5 | Optuna + Excel + HTML generados automáticamente | `python run.py` lanza 128 trials y genera outputs |
| 6 | Multi-activo y multi-estrategia con limpieza de RAM | Carpetas de resultados independientes y correctas |
| 7 | Salidas personalizadas + gestión de errores + rotación | Módulo listo para uso diario intensivo |

# FASE 0 — Auditoría del módulo `PANEL PORTFOLIO`

> Estado del repo: rama `main`, árbol limpio. ~6.500 líneas Python, 8 capas, sin Rust propio.
> Objetivo de la auditoría: mapear qué es **core**, qué es **legacy/ruido** y qué **rompe** si se toca, antes de refactorizar hacia un motor de riesgo predictivo orientado a decisión.

---

## 1. Mapa de capas (arquitectura actual)

| Capa | Archivos clave | Rol | Veredicto |
|------|----------------|-----|-----------|
| `CONFIGURACION/` | `config.py`, `_tecnico.py` | Parámetros, presets | **Core con legacy** (perfiles fijos) |
| `CONTRATOS/` | `modelos.py` (249), `validacion.py` (376), `errores.py`, `rutas.py` | Dataclasses tipadas + validación | **Core** (base sólida ya existe) |
| `DATOS/` | `cargador.py`, `alineacion.py` | Carga parquet + log-returns alineados | **Core** |
| `DESCARGADOR/` | `descargador.py`, `cache.py` | Descarga Yahoo + caché | Core (auxiliar) |
| `ANALISIS/` | `momentos.py`, `correlacion.py`, `pca.py`, `regimenes.py`, `diversificacion.py` | Momentos, Ledoit-Wolf, PCA, regímenes | **Core parcial** (PCA = ruido) |
| `OPTIMIZACION/` | `markowitz.py`, `frontera.py`, `perfil_riesgo.py`, `hrp.py`, `risk_parity.py`, `max_diversificacion.py`, `cvar.py`, `black_litterman.py`, `montecarlo.py`, `asignadores.py` | 7 asignadores + frontera + nube MC | **Core + mucho legacy** |
| `RIESGO/` | `riesgo.py`, `metricas.py`, `perfil.py`, `walk_forward.py`, `stress.py`, `convexidad.py`, `regimenes_riesgo.py` | Walk-forward (mini-backtest), VaR hist, stress | **Mezcla crítica** (ver §4) |
| `REPORTES/` | `html.py`, `pdf.py`, `excel.py`, `graficos_plotly.py` (420), `graficos_mpl.py`, `narrativa.py`, `tablas.py`, `objetivo.py`, `i18n.py`, `formato.py` | Informe HTML/PDF/Excel | **Core con sobrecarga de ruido** |
| `TESTS/` | 7 archivos | Cobertura parcial | Core (ampliar) |
| Entrypoints | `arranque.py`, `ejecutar.py`, `orquestador.py` | Pipeline en 11 pasos | **Core** (reescalar) |

---

## 2. Implementaciones ya existentes (lo que NO hay que reinventar)

- **Ledoit-Wolf**: ✅ `ANALISIS/momentos.py::covarianza_ledoit_wolf`. Sólida — chequea simetría y definición positiva, anualiza ×252. Reutilizable como **matriz estructural**.
- **HRP**: ✅ `OPTIMIZACION/hrp.py` (clustering + bisección, sin invertir Σ).
- **Black-Litterman**: ✅ `OPTIMIZACION/black_litterman.py`. Pero **inactivo** (sin views → equivale a 1/N, por diseño no se calcula). Encaja con el fallback exigido (BL sin inputs → shrinkage/1N).
- **Frontera eficiente**: ✅ `OPTIMIZACION/markowitz.py::puntos_frontera` — SLSQP, 40 puntos, con pesos por punto y restricciones duras (cotas + suma=1). **Buena base** para la frontera restringida.
- **Risk parity, Máx diversificación, Min-CVaR**: ✅ implementados.
- **VaR/CVaR histórico**: ✅ `RIESGO/metricas.py::var_cvar_historico` (empírico, signo negativo correcto).

### Lo que NO existe (hay que construir)
- ❌ **EWMA / GARCH** (volatilidad táctica T+1) — no hay rastro.
- ❌ **FHS** (Filtered Historical Simulation).
- ❌ **MCR** (contribución marginal al riesgo por activo).
- ❌ **CDaR** (Conditional Drawdown at Risk).
- ❌ **VaR/CVaR forecast paramétrico** (sólo hay histórico).
- ❌ **Score de cartera** (`calcular_score_cartera`).
- ❌ **Motor Rust** propio (sólo existe el de `PANEL BACKTESTING/MOTOR`, prohibido tocar).
- ❌ **Fan chart** de percentiles (el MC actual es nube de puntos descriptiva).

---

## 3. Candidatos a ELIMINAR / DEPRECAR / mover a apéndice

| Elemento | Ubicación | Motivo | Acción |
|----------|-----------|--------|--------|
| Perfiles fijos por fracción de rango | `_tecnico.py::FRACCION_VOL_NIVEL` (0.15/0.50/0.85) | No son percentiles reales de la frontera; semi-hardcode | **Reemplazar** por P20/P50/P80 dinámicos |
| `PERFIL_RIESGO` + `VOLATILIDAD_OBJETIVO_ANUAL` estáticos | `config.py` 30-38 | Perfil estático absoluto; choca con perfiles dinámicos | **Eliminar/relegar** |
| Nube Monte Carlo (20.000 carteras aleatorias) | `OPTIMIZACION/montecarlo.py`, `fig_frontera` scatter | Telón de fondo decorativo; no responde a las 4 preguntas | **Deprecar** → reemplazar por frontera 100% + fan chart |
| PCA | `ANALISIS/pca.py`, `fig_pca` | Análisis descriptivo; no decide pesos ni pérdida | **Apéndice técnico** |
| Convexidad / captura alcista-bajista | `RIESGO/convexidad.py`, `fig_convexidad`, `objetivo=convexidad` | Métrica de backtest, no de riesgo prospectivo | **Apéndice** |
| Stress histórico 2008/2020/2022 | `RIESGO/stress.py`, `VENTANAS_STRESS` | Sólo evaluable si hay OOS; ruido en informe ejecutivo | **Apéndice** (opcional) |
| Tabla "espejismo" promesa vs realidad | `tablas.py::tabla_espejismo`, `_tabla_espejismo_html` | Trata retorno histórico como referencia | **Revisar**: mantener sólo el aviso, no la promesa |
| Equity / drawdown walk-forward como sección principal | `fig_equity`, `fig_drawdown`, sección "validacion" | Es backtest secuencial dentro de un panel que debe ser vectorial/probabilístico | **Mover a apéndice** |
| Excel `informe.xlsx` + `~$informe.xlsx` (lock) | `SALIDAS/`, `REPORTES/excel.py` | Salida pesada no pedida en el foco | **Opcional / apéndice** |
| `graficos_mpl.py` (PNGs duplicados de Plotly) | `REPORTES/graficos_mpl.py` | Duplica figuras para el PDF | Mantener mínimo (sólo fan chart + tablas) |
| 7 métodos de asignación en tabla maestra | `asignadores.py` | El foco son 4 carteras (Bajo/Medio/Alto/MaxSharpe), no 7 métodos | **Reducir** a 4 perfiles; resto → apéndice/diagnóstico |

---

## 4. Riesgos de ruptura del pipeline (orden de dependencias)

El pipeline es **fail-fast y unidireccional** (`orquestador.py::PASOS`). Romper un contrato rompe todo aguas abajo.

1. **`CONTRATOS/modelos.py` es el eje.** `PaqueteReporte` agrega TODO; los reportes leen de ahí sin recalcular (bien). **Riesgo alto**: añadir/quitar campos (ej. `RiskForecast`, `SimulationSummary`, MCR, score) obliga a tocar `orquestador`, `riesgo`, `perfil` y los 3 generadores de reporte a la vez. → Plan: ampliar dataclasses de forma aditiva primero, migrar consumidores después.

2. **`_paso_riesgo` pesa 22 s** (walk-forward completo). Es el cuello de botella y conceptualmente el "backtest dentro del portfolio". Sustituirlo por forecast vectorial (EWMA→FHS→MC Rust) **mejora foco y velocidad**, pero `riesgo.metricas` alimenta `objetivo.py::recomendar` y la tabla maestra → hay que migrar el scoring en paralelo.

3. **`validacion.py` (376 líneas)** valida config estricta. Cambiar `config.py` (quitar `PERFIL_RIESGO`, etc.) **romperá** la validación y los tests `test_configuracion.py` / `test_perfil_riesgo.py`. → Tocar config + validación + tests como una sola unidad atómica.

4. **`asignadores.py::metodos()`** define las claves que recorren `walk_forward`, `metricas`, `objetivo` y reportes. Reducir de 7 métodos a 4 perfiles propaga a `i18n.py` (nombres visibles) y a las tablas. → Cambio coordinado.

5. **Reportes (`html.py`, `pdf.py`) leen `figs[...]` por clave fija** (`fig_frontera`, `fig_equity`, etc.). Quitar una figura sin quitar su referencia lanza `KeyError`. → Editar `todas_las_figuras` + consumidores juntos.

6. **Aislamiento Rust**: el patrón de `PANEL BACKTESTING/MOTOR` (pyo3 0.28, numpy 0.28, `cargar_motor()` compila on-demand vía `cargo build --release`, carga `.dylib/.so`) es **replicable y excelente referencia**, pero está **prohibido importarlo**. El nuevo `MOTOR_RIESGO/` será un crate independiente con su propio `wrapper`/binding. **Riesgo**: requiere `cargo` instalado en el entorno de ejecución; prever fallback Python puro si no compila.

---

## 5. Diagnóstico de foco (las 4 preguntas)

| Pregunta foco | ¿Cubierta hoy? | Brecha |
|---------------|----------------|--------|
| ¿Qué activos tengo? | Parcial | OK datos; sobra ruido descriptivo |
| ¿Cómo se relacionan? | Sí (corr media/cola, Ledoit-Wolf) | Falta separar lente **estructural** vs **táctica** |
| ¿Qué pesos debo usar? | Sí (frontera + perfiles) | Perfiles estáticos → dinámicos; reducir a 4 |
| ¿Cuánto puedo perder mañana/mes? | **Débil** | Sólo VaR histórico in-sample. Falta EWMA T+1, FHS, MC forward, CDaR, VaR forecast |

**Conclusión:** la base de datos, contratos, Ledoit-Wolf y frontera son sólidos y reutilizables. El grueso del trabajo es (a) **podar** la capa descriptiva/backtest (PCA, convexidad, stress, nube MC, walk-forward como sección principal), (b) **construir** la capa prospectiva de riesgo (EWMA→FHS→MC Rust→CDaR→VaR forecast→MCR→score) y (c) **reenfocar** el informe a un dashboard de decisión con un único fan chart.

---

*Fin Fase 0. No se ha modificado código.*

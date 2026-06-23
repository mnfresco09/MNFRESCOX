# PANEL PORTFOLIO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el panel independiente de optimización de carteras definido en `ESPECIFICACION_DISENO.md`, con siete asignadores, backtest walk-forward OOS, análisis de regímenes y reportes offline.

**Architecture:** Un orquestador mínimo coordina capas con dependencias en un solo sentido. Las capas intercambian dataclasses inmutables de `CONTRATOS/`; cada algoritmo y salida vive en un módulo focalizado, sin acceso a helpers privados de otras carpetas. La ejecución se divide en cinco fases que se verifican y comunican antes de continuar.

**Tech Stack:** Python 3.12, numpy, pandas, scipy, scikit-learn, cvxpy, yfinance, pyarrow, plotly, xlsxwriter, pytest.

---

## Reglas de ejecución

- Ejecutar todos los comandos desde `PANEL PORTFOLIO/`.
- Crear y usar `.venv/` dentro del panel; no modificar el entorno compartido ni
  usar el Python 3.14 del sistema.
- No modificar archivos fuera de `PANEL PORTFOLIO/`.
- Aplicar TDD: prueba roja, implementación mínima, prueba verde, refactor y commit.
- No avanzar de fase hasta ejecutar su suite completa y la comprobación indicada.
- No mantener implementaciones antiguas junto a las nuevas: sustituir y eliminar.
- No confirmar `.DS_Store`, `__pycache__/`, históricos descargados ni salidas.

## Mapa final de archivos

```text
PANEL PORTFOLIO/
├── .gitignore
├── CONFIGURACION/
│   ├── __init__.py
│   └── config.py
├── CONTRATOS/
│   ├── __init__.py
│   ├── errores.py
│   ├── modelos.py
│   └── validacion.py
├── DESCARGADOR/
│   ├── __init__.py
│   └── descargador.py
├── DATOS/
│   ├── __init__.py
│   ├── alineacion.py
│   └── cargador.py
├── ANALISIS/
│   ├── __init__.py
│   ├── correlacion.py
│   ├── diversificacion.py
│   ├── momentos.py
│   ├── pca.py
│   ├── regimenes.py
│   └── servicio.py
├── OPTIMIZACION/
│   ├── __init__.py
│   ├── black_litterman.py
│   ├── comun.py
│   ├── frontera.py
│   ├── hrp.py
│   ├── markowitz.py
│   ├── min_cvar.py
│   ├── risk_parity.py
│   └── servicio.py
├── RIESGO/
│   ├── __init__.py
│   ├── metricas.py
│   ├── por_regimen.py
│   ├── servicio.py
│   ├── stress.py
│   └── walk_forward.py
├── REPORTES/
│   ├── __init__.py
│   ├── excel.py
│   ├── html.py
│   ├── manifiesto.py
│   └── servicio.py
├── HISTORICO/
│   └── .gitkeep
├── SALIDAS/
│   └── .gitkeep
├── TESTS/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_alineacion.py
│   ├── test_analisis.py
│   ├── test_black_litterman.py
│   ├── test_configuracion.py
│   ├── test_descargador.py
│   ├── test_frontera.py
│   ├── test_hrp.py
│   ├── test_integracion.py
│   ├── test_markowitz.py
│   ├── test_metricas.py
│   ├── test_min_cvar.py
│   ├── test_regimenes.py
│   ├── test_reportes.py
│   ├── test_risk_parity.py
│   ├── test_stress.py
│   └── test_walk_forward.py
├── ESPECIFICACION_DISENO.md
├── PLAN_IMPLEMENTACION.md
├── ejecutar.py
└── requirements.txt
```

## Cobertura de la especificación

- Tasks 1–5: configuración, contratos, Yahoo, publicación segura, históricos,
  intersección de calendarios y log-retornos.
- Tasks 6–7: Ledoit-Wolf, correlación media, correlación condicional de cola,
  PCA, diversificación y regímenes transparentes.
- Tasks 8–13: Markowitz máximo Sharpe, Markowitz retorno objetivo, mínima
  varianza, Risk Parity, HRP, Min-CVaR, Black-Litterman, frontera y Monte Carlo.
- Tasks 14–16: métricas históricas, walk-forward OOS, costes, métricas por
  régimen, diversificación en crisis y stress.
- Tasks 17–19: HTML offline, Excel, manifiesto, CLI, auditoría de aislamiento y
  verificación final.

# Fase 1 — Cimientos, descarga y datos

### Task 1: Higiene, dependencias y configuración

**Files:**
- Create: `PANEL PORTFOLIO/.gitignore`
- Create: `PANEL PORTFOLIO/requirements.txt`
- Modify: `PANEL PORTFOLIO/CONFIGURACION/config.py`
- Create: `PANEL PORTFOLIO/TESTS/__init__.py`
- Create: `PANEL PORTFOLIO/TESTS/test_configuracion.py`

- [ ] **Step 1: Escribir la prueba roja de configuración declarativa**

```python
from CONFIGURACION import config


def test_configuracion_contiene_parametros_obligatorios():
    assert len(config.TICKERS) >= 2
    assert config.ACTIVO_REFERENCIA in config.TICKERS
    assert config.NIVEL_CONFIANZA == 0.95
    assert config.COSTE_TRANSACCION_PB >= 0
    assert config.UMBRAL_DRAWDOWN_CRISIS < config.UMBRAL_DRAWDOWN_BAJISTA < 0
    assert set(config.VENTANAS_STRESS) == {
        "crisis_financiera_2008",
        "covid_2020",
        "crisis_2022",
    }
```

- [ ] **Step 2: Ejecutar la prueba y confirmar el fallo**

Run: `../.venv/bin/python -m pytest TESTS/test_configuracion.py -v`

Expected: FAIL por ausencia de `ACTIVO_REFERENCIA`.

- [ ] **Step 3: Completar parámetros y dependencias**

Añadir en `config.py`, sin funciones ni imports:

```python
ACTIVO_REFERENCIA: str = "^GSPC"
COSTE_TRANSACCION_PB: float = 10.0
NIVEL_CONFIANZA: float = 0.95
MIN_RETORNOS_ANALISIS: int = 252
UMBRAL_DRAWDOWN_CRISIS: float = -0.20
UMBRAL_DRAWDOWN_BAJISTA: float = -0.10
VENTANA_VOLATILIDAD: int = 20
VENTANA_MEDIA_LARGA: int = 200
VENTANA_PENDIENTE: int = 20
PERCENTIL_VOLATILIDAD_CRISIS: float = 0.90
VENTANAS_STRESS: dict[str, tuple[str, str]] = {
    "crisis_financiera_2008": ("2008-09-01", "2009-03-31"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "crisis_2022": ("2022-01-03", "2022-10-12"),
}
```

`requirements.txt`:

```text
numpy>=2.0,<3
pandas>=2.2,<3
scipy>=1.14,<2
scikit-learn>=1.5,<2
cvxpy>=1.6,<2
yfinance>=0.2.54,<1
pyarrow>=18,<23
plotly>=6,<7
xlsxwriter>=3.2,<4
pytest>=8,<10
```

`.gitignore`:

```text
.DS_Store
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
HISTORICO/*.parquet
SALIDAS/*
!SALIDAS/.gitkeep
```

- [ ] **Step 4: Crear el entorno local e instalar dependencias**

Run: `../.venv/bin/python -m venv .venv`

Expected: se crea `.venv/` dentro de `PANEL PORTFOLIO/`.

Run: `.venv/bin/python -m pip install -r requirements.txt`

Expected: instalación completa sin modificar `../.venv`.

- [ ] **Step 5: Limpiar artefactos generados dentro del panel**

Run: `find . -name __pycache__ -type d -prune -exec rm -rf {} +`

Expected: no quedan directorios `__pycache__`.

Run: `find . -name .DS_Store -delete`

Expected: no queda `.DS_Store`.

- [ ] **Step 6: Ejecutar la prueba verde**

Run: `.venv/bin/python -m pytest TESTS/test_configuracion.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt CONFIGURACION TESTS
git commit -m "feat(portfolio): define configuration and dependencies"
```

### Task 2: Errores, contratos y validación

**Files:**
- Create: `PANEL PORTFOLIO/CONTRATOS/__init__.py`
- Create: `PANEL PORTFOLIO/CONTRATOS/errores.py`
- Create: `PANEL PORTFOLIO/CONTRATOS/modelos.py`
- Create: `PANEL PORTFOLIO/CONTRATOS/validacion.py`
- Modify: `PANEL PORTFOLIO/TESTS/test_configuracion.py`

- [ ] **Step 1: Escribir pruebas rojas de validación**

```python
import pytest

from CONTRATOS.errores import ErrorConfiguracion
from CONTRATOS.validacion import construir_configuracion


def test_rechaza_activo_referencia_fuera_de_la_cesta():
    with pytest.raises(ErrorConfiguracion, match="ACTIVO_REFERENCIA"):
        construir_configuracion(
            tickers=("AAA", "BBB"),
            activo_referencia="SPY",
            peso_maximo=0.60,
        )


def test_rechaza_limite_long_only_inviable():
    with pytest.raises(ErrorConfiguracion, match="inviable"):
        construir_configuracion(
            tickers=("AAA", "BBB", "CCC"),
            activo_referencia="AAA",
            peso_maximo=0.30,
        )
```

- [ ] **Step 2: Ejecutar y confirmar el fallo**

Run: `.venv/bin/python -m pytest TESTS/test_configuracion.py -v`

Expected: FAIL por ausencia de `CONTRATOS`.

- [ ] **Step 3: Implementar excepciones y contratos base**

`errores.py`:

```python
class ErrorPanelPortfolio(RuntimeError):
    def __init__(self, etapa: str, mensaje: str) -> None:
        self.etapa = etapa
        super().__init__(f"[{etapa}] {mensaje}")


class ErrorConfiguracion(ErrorPanelPortfolio):
    def __init__(self, mensaje: str) -> None:
        super().__init__("CONFIGURACION", mensaje)


class ErrorDatos(ErrorPanelPortfolio):
    pass


class ErrorOptimizacion(ErrorPanelPortfolio):
    pass


class ErrorReporte(ErrorPanelPortfolio):
    pass
```

`modelos.py` contendrá dataclasses congeladas. Los contratos se fijan aquí para
que las capas posteriores no inventen estructuras incompatibles:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class Restricciones:
    solo_largos: bool
    peso_maximo: float | None


@dataclass(frozen=True)
class ViewBlackLitterman:
    activos: tuple[tuple[str, float], ...]
    retorno_anual: float
    confianza: float


@dataclass(frozen=True)
class VentanaStress:
    nombre: str
    inicio: pd.Timestamp
    fin: pd.Timestamp


@dataclass(frozen=True)
class ParametrosRegimen:
    drawdown_crisis: float
    drawdown_bajista: float
    ventana_volatilidad: int
    ventana_media_larga: int
    ventana_pendiente: int
    percentil_volatilidad_crisis: float


@dataclass(frozen=True)
class Configuracion:
    tickers: tuple[str, ...]
    fecha_inicio: str
    fecha_fin: str
    activo_referencia: str
    frecuencia_rebalanceo: str
    ventana_estimacion: int
    restricciones: Restricciones
    retorno_objetivo_anual: float
    tasa_libre_riesgo_anual: float
    dias_anio: int
    coste_transaccion_pb: float
    nivel_confianza: float
    min_retornos_analisis: int
    views_black_litterman: tuple[ViewBlackLitterman, ...]
    ventanas_stress: tuple[VentanaStress, ...]
    parametros_regimen: ParametrosRegimen
    n_carteras_montecarlo: int
    semilla: int
    carpeta_historico: Path
    carpeta_salidas: Path


@dataclass(frozen=True)
class ResumenActivo:
    ticker: str
    archivo: str
    filas: int
    fecha_inicio: pd.Timestamp
    fecha_fin: pd.Timestamp
    huecos_sospechosos: int
    hueco_max_dias: int


@dataclass(frozen=True)
class DatosAlineados:
    activos: tuple[str, ...]
    cierres: pd.DataFrame
    log_retornos: pd.DataFrame


@dataclass(frozen=True)
class ResultadoPCA:
    varianza_explicada: pd.Series
    varianza_acumulada: pd.Series
    cargas: pd.DataFrame


@dataclass(frozen=True)
class ResultadoAnalisis:
    log_retornos: pd.DataFrame
    retornos_esperados: pd.Series
    covarianza: pd.DataFrame
    volatilidades: pd.Series
    correlacion_media: pd.DataFrame
    correlacion_cola: pd.DataFrame
    diferencia_correlacion_cola: pd.DataFrame
    observaciones_cola: int
    pca: ResultadoPCA
    regimenes: pd.Series


@dataclass(frozen=True)
class MetricasEstimadas:
    retorno_anual: float
    volatilidad_anual: float
    sharpe: float


@dataclass(frozen=True)
class ResultadoAsignacion:
    nombre: str
    pesos: pd.Series
    metricas: MetricasEstimadas
    estado_solver: str
    diagnostico: str
    advertencias: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResultadoMonteCarlo:
    pesos: pd.DataFrame
    metricas: pd.DataFrame


@dataclass(frozen=True)
class ResultadoFrontera:
    puntos: pd.DataFrame
    minima_varianza: ResultadoAsignacion
    maximo_sharpe: ResultadoAsignacion
    retorno_objetivo: ResultadoAsignacion


@dataclass(frozen=True)
class Rebalanceo:
    fecha: pd.Timestamp
    pesos: Mapping[str, pd.Series]
    rotacion: Mapping[str, float]
    coste: Mapping[str, float]


@dataclass(frozen=True)
class ResultadoWalkForward:
    retornos: pd.DataFrame
    equity: pd.DataFrame
    pesos_diarios: Mapping[str, pd.DataFrame]
    rebalanceos: tuple[Rebalanceo, ...]


@dataclass(frozen=True)
class MetricasCartera:
    retorno_anual: float
    volatilidad_anual: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    duracion_drawdown_dias: int
    fecha_recuperacion: pd.Timestamp | None
    var: float
    cvar: float


@dataclass(frozen=True)
class ResultadoStress:
    nombre: str
    evaluable: bool
    observaciones: int
    metricas: Mapping[str, MetricasCartera]


@dataclass(frozen=True)
class ResultadoRiesgo:
    walk_forward: ResultadoWalkForward
    metricas: Mapping[str, MetricasCartera]
    metricas_por_regimen: Mapping[str, pd.DataFrame]
    stress: Mapping[str, ResultadoStress]
    diversificacion_crisis: pd.DataFrame


@dataclass(frozen=True)
class PaqueteReporte:
    configuracion: Configuracion
    datos: DatosAlineados
    analisis: ResultadoAnalisis
    analisis_actual: ResultadoAnalisis
    asignaciones: Mapping[str, ResultadoAsignacion]
    frontera: ResultadoFrontera
    monte_carlo: ResultadoMonteCarlo
    riesgo: ResultadoRiesgo


@dataclass(frozen=True)
class RutasReporte:
    html: Path
    excel: Path
    manifiesto: Path
```

- [ ] **Step 4: Implementar validación explícita**

`construir_configuracion` aceptará todos los campos con valores por defecto
equivalentes a `CONFIGURACION/config.py`, normalizará tickers a tupla y validará:

```python
def _validar_limite(n_activos: int, solo_largos: bool, peso_maximo: float | None) -> None:
    if peso_maximo is None:
        return
    if not 0 < peso_maximo <= 1:
        raise ErrorConfiguracion("PESO_MAXIMO_POR_ACTIVO debe estar en (0, 1].")
    if solo_largos and n_activos * peso_maximo < 1 - 1e-12:
        raise ErrorConfiguracion("La suma de límites long-only es inviable.")
```

La función `cargar_configuracion()` importará exclusivamente datos de
`CONFIGURACION.config` y devolverá `Configuracion`.

- [ ] **Step 5: Ejecutar pruebas**

Run: `.venv/bin/python -m pytest TESTS/test_configuracion.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add CONTRATOS TESTS/test_configuracion.py
git commit -m "feat(portfolio): add typed contracts and validation"
```

### Task 3: Descarga validada y reemplazo atómico

**Files:**
- Replace: `PANEL PORTFOLIO/DESCARGADOR/descargador.py`
- Create: `PANEL PORTFOLIO/TESTS/test_descargador.py`

- [ ] **Step 1: Escribir pruebas rojas de descarga**

```python
from pathlib import Path

import pandas as pd
import pytest

from CONTRATOS.errores import ErrorDatos
from DESCARGADOR.descargador import descargar_cesta


def test_no_reemplaza_historicos_si_un_activo_falla(tmp_path: Path):
    previo = tmp_path / "AAA_1d.parquet"
    pd.DataFrame({"fecha": [pd.Timestamp("2024-01-01")], "cierre": [10.0]}).to_parquet(previo)

    def proveedor(ticker: str, inicio: str, fin: str):
        if ticker == "BBB":
            return pd.Series(dtype=float)
        return pd.Series([11.0], index=pd.to_datetime(["2024-01-02"]), name="cierre")

    with pytest.raises(ErrorDatos, match="BBB"):
        descargar_cesta(("AAA", "BBB"), "2024-01-01", "2024-02-01", tmp_path, proveedor)

    conservado = pd.read_parquet(previo)
    assert conservado["cierre"].tolist() == [10.0]
```

- [ ] **Step 2: Ejecutar y confirmar el fallo**

Run: `.venv/bin/python -m pytest TESTS/test_descargador.py -v`

Expected: FAIL porque `descargar_cesta` no existe.

- [ ] **Step 3: Implementar proveedor Yahoo y validación**

La API pública será:

```python
def descargar_cesta(
    tickers: tuple[str, ...],
    fecha_inicio: str,
    fecha_fin: str,
    carpeta_historico: Path,
    proveedor: ProveedorCierres = descargar_desde_yahoo,
) -> tuple[ResumenActivo, ...]:
    carpeta_historico.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=carpeta_historico.parent) as temporal:
        carpeta_temporal = Path(temporal)
        preparados = tuple(
            preparar_activo(
                ticker,
                proveedor(ticker, fecha_inicio, fecha_fin),
                carpeta_temporal,
            )
            for ticker in tickers
        )
        publicar_preparados(
            preparados,
            carpeta_historico,
            carpeta_temporal / "respaldos",
        )
        return tuple(preparado.resumen for preparado in preparados)
```

Cada serie debe tener índice `DatetimeIndex`, fechas únicas y crecientes, al
menos dos filas, valores finitos y positivos. `ResumenActivo` incluirá ticker,
archivo, filas, fecha inicial/final, huecos sospechosos y hueco máximo.

- [ ] **Step 4: Implementar publicación atómica de la cesta**

Usar `TemporaryDirectory(dir=carpeta_historico.parent)`. Escribir y releer todos
los Parquet temporales. Antes de publicar, copiar los definitivos existentes a
la carpeta temporal. Si cualquier reemplazo falla, restaurar los respaldos y
eliminar los definitivos nuevos:

```python
def publicar_preparados(preparados, carpeta_historico, carpeta_respaldos):
    carpeta_respaldos.mkdir()
    respaldos = {}
    publicados = []
    try:
        for preparado in preparados:
            definitivo = carpeta_historico / preparado.resumen.archivo
            if definitivo.exists():
                respaldo = carpeta_respaldos / definitivo.name
                shutil.copy2(definitivo, respaldo)
                respaldos[definitivo] = respaldo
        for preparado in preparados:
            definitivo = carpeta_historico / preparado.resumen.archivo
            preparado.ruta_temporal.replace(definitivo)
            publicados.append(definitivo)
    except Exception:
        for definitivo in publicados:
            if definitivo not in respaldos:
                definitivo.unlink(missing_ok=True)
        for definitivo, respaldo in respaldos.items():
            shutil.copy2(respaldo, definitivo)
        raise
```

Ante cualquier excepción, elevar `ErrorDatos("DESCARGADOR", mensaje)` y dejar
intactos los archivos definitivos.

- [ ] **Step 5: Ejecutar pruebas**

Run: `.venv/bin/python -m pytest TESTS/test_descargador.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add DESCARGADOR TESTS/test_descargador.py
git commit -m "feat(portfolio): download and publish validated histories atomically"
```

### Task 4: Carga, alineación y log-retornos

**Files:**
- Replace: `PANEL PORTFOLIO/DATOS/cargador.py`
- Create: `PANEL PORTFOLIO/DATOS/alineacion.py`
- Create: `PANEL PORTFOLIO/TESTS/test_alineacion.py`
- Create: `PANEL PORTFOLIO/TESTS/conftest.py`

- [ ] **Step 1: Escribir pruebas rojas de intersección**

```python
import numpy as np
import pandas as pd

from DATOS.alineacion import alinear_y_calcular_retornos


def test_intersecta_calendarios_sin_forward_fill():
    cierres = {
        "BTC": pd.Series([100, 110, 121, 133.1], index=pd.to_datetime(
            ["2024-01-05", "2024-01-06", "2024-01-08", "2024-01-09"]
        )),
        "SPY": pd.Series([200, 220, 242], index=pd.to_datetime(
            ["2024-01-05", "2024-01-08", "2024-01-09"]
        )),
    }
    resultado = alinear_y_calcular_retornos(cierres, min_retornos=2)
    assert resultado.cierres.index.tolist() == list(pd.to_datetime(
        ["2024-01-05", "2024-01-08", "2024-01-09"]
    ))
    np.testing.assert_allclose(resultado.log_retornos.to_numpy(), np.log(1.1))
```

- [ ] **Step 2: Ejecutar y confirmar el fallo**

Run: `.venv/bin/python -m pytest TESTS/test_alineacion.py -v`

Expected: FAIL por ausencia de `DATOS.alineacion`.

- [ ] **Step 3: Implementar carga estricta**

`cargar_cierres` rechazará archivos ausentes, esquemas diferentes de
`fecha/cierre`, duplicados, orden descendente, nulos y precios no positivos. No
eliminará ni corregirá filas.

- [ ] **Step 4: Implementar alineación**

```python
def alinear_y_calcular_retornos(
    cierres: Mapping[str, pd.Series],
    min_retornos: int,
) -> DatosAlineados:
    tabla = pd.concat(cierres, axis=1, join="inner").sort_index()
    if tabla.isna().any().any():
        raise ErrorDatos("DATOS", "La intersección contiene valores nulos.")
    log_retornos = np.log(tabla / tabla.shift(1)).iloc[1:]
    if len(log_retornos) < min_retornos:
        raise ErrorDatos(
            "DATOS",
            f"Retornos alineados insuficientes: {len(log_retornos)} < {min_retornos}.",
        )
    return DatosAlineados(
        activos=tuple(tabla.columns),
        cierres=tabla,
        log_retornos=log_retornos,
    )


def recortar_datos(datos: DatosAlineados, n_retornos: int) -> DatosAlineados:
    if len(datos.log_retornos) < n_retornos:
        raise ErrorDatos(
            "DATOS",
            f"Ventana actual insuficiente: {len(datos.log_retornos)} < {n_retornos}.",
        )
    log_retornos = datos.log_retornos.iloc[-n_retornos:]
    cierres = datos.cierres.loc[
        datos.cierres.index >= datos.cierres.index[-(n_retornos + 1)]
    ]
    return DatosAlineados(datos.activos, cierres, log_retornos)
```

- [ ] **Step 5: Ejecutar suite de fase**

Run: `.venv/bin/python -m pytest TESTS/test_configuracion.py TESTS/test_descargador.py TESTS/test_alineacion.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add DATOS TESTS
git commit -m "feat(portfolio): align calendars and compute log returns"
```

### Task 5: Comando de descarga y verificación real

**Files:**
- Create: `PANEL PORTFOLIO/ejecutar.py`
- Modify: `PANEL PORTFOLIO/CONTRATOS/modelos.py`
- Create: `PANEL PORTFOLIO/HISTORICO/.gitkeep`
- Create: `PANEL PORTFOLIO/SALIDAS/.gitkeep`
- Create: `PANEL PORTFOLIO/TESTS/test_integracion.py`

- [ ] **Step 1: Escribir prueba roja del comando**

```python
from ejecutar import main


def test_comando_desconocido_devuelve_error(capsys):
    assert main(["desconocido"]) == 2
    assert "descargar" in capsys.readouterr().err
```

- [ ] **Step 2: Ejecutar y confirmar el fallo**

Run: `.venv/bin/python -m pytest TESTS/test_integracion.py -v`

Expected: FAIL porque `ejecutar.py` no existe.

- [ ] **Step 3: Implementar CLI mínima**

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="PANEL PORTFOLIO")
    parser.add_argument("comando", choices=("descargar", "analizar"))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        configuracion = cargar_configuracion()
        if args.comando == "descargar":
            resumenes = descargar_cesta(
                configuracion.tickers,
                configuracion.fecha_inicio,
                configuracion.fecha_fin,
                configuracion.carpeta_historico,
            )
            imprimir_resumen(resumenes)
            return 0
        return ejecutar_analisis(configuracion)
    except ErrorPanelPortfolio as exc:
        print(str(exc), file=sys.stderr)
        return 1
```

Durante esta fase, `ejecutar_analisis` elevará un error explícito:
`[ANALISIS] La fase de análisis todavía no está instalada.` No devolverá datos
parciales.

- [ ] **Step 4: Ejecutar pruebas locales**

Run: `.venv/bin/python -m pytest TESTS/test_configuracion.py TESTS/test_descargador.py TESTS/test_alineacion.py TESTS/test_integracion.py -v`

Expected: PASS.

- [ ] **Step 5: Verificar dependencias del panel**

Run: `.venv/bin/python -m pip check`

Expected: `No broken requirements found.`

- [ ] **Step 6: Ejecutar descarga real**

Run: `.venv/bin/python ejecutar.py descargar`

Expected: código 0 y una fila de resumen por ticker con filas, fecha inicial,
fecha final, huecos y archivo.

- [ ] **Step 7: Validar históricos y alineación real**

Run:

```bash
.venv/bin/python -c "from CONTRATOS.validacion import cargar_configuracion; from DATOS.cargador import cargar_cierres; from DATOS.alineacion import alinear_y_calcular_retornos; c=cargar_configuracion(); d=alinear_y_calcular_retornos(cargar_cierres(c.tickers,c.carpeta_historico),c.min_retornos_analisis); print(d.cierres.shape, d.cierres.index.min(), d.cierres.index.max())"
```

Expected: cinco columnas, al menos 253 cierres comunes, sin errores.

- [ ] **Step 8: Commit y checkpoint**

```bash
git add .
git commit -m "feat(portfolio): complete validated data foundation"
```

Comunicar archivos creados, pruebas ejecutadas, cobertura real por activo,
dimensión de la intersección y cualquier hueco sospechoso. No comenzar Fase 2
sin esta comprobación.

# Fase 2 — Análisis estadístico

### Task 6: Momentos Ledoit-Wolf, correlación y PCA

**Files:**
- Create: `PANEL PORTFOLIO/ANALISIS/momentos.py`
- Create: `PANEL PORTFOLIO/ANALISIS/correlacion.py`
- Create: `PANEL PORTFOLIO/ANALISIS/pca.py`
- Create: `PANEL PORTFOLIO/TESTS/test_analisis.py`
- Verify: `PANEL PORTFOLIO/CONTRATOS/modelos.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_covarianza_ledoit_wolf_es_simetrica(retornos):
    resultado = estimar_momentos(retornos, dias_anio=252)
    np.testing.assert_allclose(resultado.covarianza, resultado.covarianza.T)
    assert np.linalg.eigvalsh(resultado.covarianza).min() >= -1e-10


def test_correlacion_cola_usa_peor_decil(retornos):
    resultado = correlacion_condicional(retornos, "A", cuantil=0.10)
    umbral = retornos["A"].quantile(0.10)
    assert resultado.observaciones == int((retornos["A"] <= umbral).sum())


def test_pca_varianza_suma_uno(retornos):
    resultado = calcular_pca(retornos)
    assert resultado.varianza_explicada.sum() == pytest.approx(1.0)
```

- [ ] **Step 2: Ejecutar y confirmar fallos**

Run: `.venv/bin/python -m pytest TESTS/test_analisis.py -v`

Expected: FAIL por módulos ausentes.

- [ ] **Step 3: Implementar momentos**

Usar `LedoitWolf().fit(log_retornos.to_numpy())`; anualizar media y covarianza
con `dias_anio`. Conservar índices y columnas originales en Series/DataFrames.

- [ ] **Step 4: Implementar correlación y PCA**

La cola será `retornos[referencia] <= quantile(0.10)`. PCA estandarizará cada
columna con `StandardScaler`, ejecutará `PCA()` y devolverá varianza explicada,
acumulada y cargas.

- [ ] **Step 5: Ejecutar pruebas y commit**

Run: `.venv/bin/python -m pytest TESTS/test_analisis.py -v`

Expected: PASS.

```bash
git add ANALISIS TESTS/test_analisis.py
git commit -m "feat(portfolio): estimate robust moments correlations and pca"
```

### Task 7: Diversificación y regímenes sin look-ahead

**Files:**
- Create: `PANEL PORTFOLIO/ANALISIS/diversificacion.py`
- Create: `PANEL PORTFOLIO/ANALISIS/regimenes.py`
- Create: `PANEL PORTFOLIO/ANALISIS/servicio.py`
- Create: `PANEL PORTFOLIO/TESTS/test_regimenes.py`
- Modify: `PANEL PORTFOLIO/TESTS/test_analisis.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_numero_efectivo_apuestas_igual_riesgo():
    cov = np.eye(4)
    pesos = np.full(4, 0.25)
    assert numero_efectivo_apuestas(pesos, cov) == pytest.approx(4.0)


def test_regimenes_no_cambian_al_agregar_futuro(serie_referencia):
    base = etiquetar_regimenes(serie_referencia.iloc[:-20], parametros)
    completo = etiquetar_regimenes(serie_referencia, parametros)
    pd.testing.assert_series_equal(base, completo.loc[base.index])
```

- [ ] **Step 2: Ejecutar y confirmar fallos**

Run: `.venv/bin/python -m pytest TESTS/test_analisis.py TESTS/test_regimenes.py -v`

Expected: FAIL por funciones ausentes.

- [ ] **Step 3: Implementar diversificación**

```python
def diversification_ratio(pesos, covarianza):
    volatilidades = np.sqrt(np.diag(covarianza))
    volatilidad_cartera = np.sqrt(pesos @ covarianza @ pesos)
    return float(pesos @ volatilidades / volatilidad_cartera)


def numero_efectivo_apuestas(pesos, covarianza):
    marginal = covarianza @ pesos
    contribuciones = pesos * marginal
    normalizadas = contribuciones / contribuciones.sum()
    return float(1.0 / np.square(normalizadas).sum())
```

- [ ] **Step 4: Implementar regímenes expansivos**

Calcular máximo acumulado, drawdown, media móvil 200, pendiente a 20,
volatilidad rolling 20 y percentil expansivo 90. Aplicar prioridad:
`crisis`, `bajista`, `alcista`, `lateral`; las primeras 199 observaciones serán
`sin_clasificar`.

- [ ] **Step 5: Crear servicio agregado**

`analizar_datos(datos, configuracion) -> ResultadoAnalisis` llamará momentos,
correlaciones, PCA y regímenes y devolverá un contrato único. El orquestador lo
usará una vez sobre toda la muestra para diagnóstico y otra sobre las últimas
`VENTANA_ESTIMACION_DIAS` observaciones para la asignación actual.

- [ ] **Step 6: Ejecutar suite de fase y commit**

Run: `.venv/bin/python -m pytest TESTS/test_analisis.py TESTS/test_regimenes.py -v`

Expected: PASS.

```bash
git add ANALISIS TESTS
git commit -m "feat(portfolio): add diversification and transparent regimes"
```

Comunicar resultados sobre los históricos reales: shrinkage Ledoit-Wolf,
observaciones de cola, varianza PCA y recuento por régimen.

# Fase 3 — Siete métodos de optimización

### Task 8: Restricciones comunes y Markowitz

**Files:**
- Create: `PANEL PORTFOLIO/OPTIMIZACION/comun.py`
- Create: `PANEL PORTFOLIO/OPTIMIZACION/markowitz.py`
- Create: `PANEL PORTFOLIO/TESTS/test_markowitz.py`
- Verify: `PANEL PORTFOLIO/CONTRATOS/modelos.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
@pytest.mark.parametrize("funcion", [minima_varianza, maximo_sharpe])
def test_markowitz_respeta_restricciones(funcion, analisis):
    resultado = funcion(analisis, restricciones_long_only)
    assert resultado.pesos.sum() == pytest.approx(1.0)
    assert (resultado.pesos >= -1e-9).all()
    assert (resultado.pesos <= 0.40 + 1e-9).all()


def test_retorno_objetivo_inviable_informa_rango(analisis):
    with pytest.raises(ErrorOptimizacion, match="rango alcanzable"):
        retorno_objetivo(analisis, restricciones_long_only, objetivo=9.0)
```

- [ ] **Step 2: Ejecutar y confirmar fallos**

Run: `.venv/bin/python -m pytest TESTS/test_markowitz.py -v`

Expected: FAIL por módulos ausentes.

- [ ] **Step 3: Implementar utilidades comunes**

Crear límites SLSQP, restricción `sum(pesos)=1`, validación de pesos y cálculo
de retorno, volatilidad y Sharpe. Para cortos usar `(-maximo, maximo)`.

- [ ] **Step 4: Implementar tres carteras**

Mínima varianza minimizará `w @ cov @ w`; máximo Sharpe minimizará el Sharpe
negativo; retorno objetivo minimizará varianza con igualdad
`w @ retornos = objetivo`. Calcular previamente retornos mínimo y máximo con
programación lineal sobre los límites.

- [ ] **Step 5: Ejecutar y commit**

Run: `.venv/bin/python -m pytest TESTS/test_markowitz.py -v`

Expected: PASS.

```bash
git add OPTIMIZACION TESTS/test_markowitz.py
git commit -m "feat(portfolio): add constrained markowitz allocators"
```

### Task 9: Risk Parity

**Files:**
- Create: `PANEL PORTFOLIO/OPTIMIZACION/risk_parity.py`
- Create: `PANEL PORTFOLIO/TESTS/test_risk_parity.py`

- [ ] **Step 1: Escribir prueba roja**

```python
def test_risk_parity_iguala_contribuciones(covarianza):
    resultado = optimizar_risk_parity(covarianza, activos, peso_maximo=0.60)
    rc = contribuciones_riesgo(resultado.pesos.to_numpy(), covarianza)
    np.testing.assert_allclose(rc / rc.sum(), np.full(len(activos), 1 / len(activos)), atol=1e-4)
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_risk_parity.py -v`

Expected: FAIL por módulo ausente.

- [ ] **Step 3: Implementar**

Minimizar la suma de diferencias cuadráticas entre contribuciones
normalizadas y `1/n`, con pesos long-only, suma uno y peso máximo. Rechazar
matrices no simétricas o resultados sin convergencia.

- [ ] **Step 4: Ejecutar y commit**

Run: `.venv/bin/python -m pytest TESTS/test_risk_parity.py -v`

Expected: PASS.

```bash
git add OPTIMIZACION/risk_parity.py TESTS/test_risk_parity.py
git commit -m "feat(portfolio): add equal risk contribution allocator"
```

### Task 10: HRP

**Files:**
- Create: `PANEL PORTFOLIO/OPTIMIZACION/hrp.py`
- Create: `PANEL PORTFOLIO/TESTS/test_hrp.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_hrp_no_invierte_covarianza(monkeypatch, covarianza):
    monkeypatch.setattr(np.linalg, "inv", lambda *_: (_ for _ in ()).throw(AssertionError()))
    resultado = optimizar_hrp(covarianza, correlacion, activos, peso_maximo=0.60)
    assert resultado.pesos.sum() == pytest.approx(1.0)


def test_hrp_proyecta_limite():
    resultado = optimizar_hrp(cov, corr, activos, peso_maximo=0.35)
    assert resultado.pesos.max() <= 0.35 + 1e-9
    assert resultado.proyeccion_aplicada
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_hrp.py -v`

Expected: FAIL por módulo ausente.

- [ ] **Step 3: Implementar HRP**

Usar `scipy.cluster.hierarchy.linkage(squareform(distancia), method="single")`,
orden de hojas, varianza de clúster con pesos inversos a diagonal y bisección
recursiva. Si excede el límite, resolver una proyección cuadrática acotada con
SLSQP.

- [ ] **Step 4: Ejecutar y commit**

Run: `.venv/bin/python -m pytest TESTS/test_hrp.py -v`

Expected: PASS.

```bash
git add OPTIMIZACION/hrp.py TESTS/test_hrp.py
git commit -m "feat(portfolio): add hierarchical risk parity"
```

### Task 11: Min-CVaR

**Files:**
- Create: `PANEL PORTFOLIO/OPTIMIZACION/min_cvar.py`
- Create: `PANEL PORTFOLIO/TESTS/test_min_cvar.py`

- [ ] **Step 1: Escribir prueba roja**

```python
def test_min_cvar_prefiere_activo_con_menor_cola():
    retornos = pd.DataFrame({"estable": [0.01, 0.0, -0.01, 0.0], "cola": [0.03, 0.03, -0.40, 0.03]})
    resultado = optimizar_min_cvar(retornos, restricciones, nivel_confianza=0.95)
    assert resultado.pesos["estable"] > resultado.pesos["cola"]
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_min_cvar.py -v`

Expected: FAIL por módulo ausente.

- [ ] **Step 3: Implementar LP Rockafellar-Uryasev**

Con `w`, `alpha` y `u >= 0`, minimizar:

```python
alpha + cp.sum(u) / ((1 - nivel_confianza) * n_escenarios)
```

sujeto a `u >= -R @ w - alpha`, suma uno y límites. Exigir estado
`OPTIMAL` u `OPTIMAL_INACCURATE` y validar pesos.

- [ ] **Step 4: Ejecutar y commit**

Run: `.venv/bin/python -m pytest TESTS/test_min_cvar.py -v`

Expected: PASS.

```bash
git add OPTIMIZACION/min_cvar.py TESTS/test_min_cvar.py
git commit -m "feat(portfolio): add historical min cvar allocator"
```

### Task 12: Black-Litterman

**Files:**
- Create: `PANEL PORTFOLIO/OPTIMIZACION/black_litterman.py`
- Create: `PANEL PORTFOLIO/TESTS/test_black_litterman.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_black_litterman_sin_views_devuelve_equilibrio():
    resultado = optimizar_black_litterman(analisis, restricciones, views=())
    np.testing.assert_allclose(resultado.pesos, np.full(4, 0.25), atol=1e-8)


def test_view_absoluta_aumenta_peso_del_activo():
    view = ({"activos": {"A": 1.0}, "retorno_anual": 0.30, "confianza": 0.80},)
    resultado = optimizar_black_litterman(analisis, restricciones, views=view)
    assert resultado.pesos["A"] > 0.25
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_black_litterman.py -v`

Expected: FAIL por módulo ausente.

- [ ] **Step 3: Implementar posterior y asignación**

Construir `P`, `Q` y `Omega` diagonal mediante confianza. Sin views devolver
exactamente el prior equiponderado. Con views calcular media y covarianza
posteriores y llamar al optimizador de máximo Sharpe común.

- [ ] **Step 4: Ejecutar y commit**

Run: `.venv/bin/python -m pytest TESTS/test_black_litterman.py -v`

Expected: PASS.

```bash
git add OPTIMIZACION/black_litterman.py TESTS/test_black_litterman.py
git commit -m "feat(portfolio): add black litterman allocator"
```

### Task 13: Frontera, Monte Carlo y servicio de siete métodos

**Files:**
- Create: `PANEL PORTFOLIO/OPTIMIZACION/frontera.py`
- Create: `PANEL PORTFOLIO/OPTIMIZACION/servicio.py`
- Create: `PANEL PORTFOLIO/TESTS/test_frontera.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_servicio_devuelve_siete_metodos(analisis, configuracion):
    resultados = optimizar_todos(analisis, configuracion)
    assert tuple(resultados) == (
        "markowitz_max_sharpe",
        "markowitz_retorno_objetivo",
        "minima_varianza",
        "risk_parity",
        "hrp",
        "min_cvar",
        "black_litterman",
    )


def test_monte_carlo_respeta_limites(resultado_montecarlo):
    assert np.allclose(resultado_montecarlo.pesos.sum(axis=1), 1.0)
    assert resultado_montecarlo.pesos.max().max() <= 0.40 + 1e-9
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_frontera.py -v`

Expected: FAIL por módulos ausentes.

- [ ] **Step 3: Implementar frontera y Monte Carlo**

Resolver 50 objetivos espaciados dentro del rango alcanzable. Generar carteras
con semilla fija y proyección al simplex acotado; calcular retorno,
volatilidad y Sharpe para cada fila.

- [ ] **Step 4: Implementar servicio**

`optimizar_todos` validará primero la factibilidad del retorno objetivo y
después llamará los siete métodos en orden fijo, devolviendo un `OrderedDict`.
Cualquier excepción se envolverá con método y fecha de estimación, sin
continuar.

- [ ] **Step 5: Ejecutar suite de fase y commit**

Run:

```bash
.venv/bin/python -m pytest TESTS/test_markowitz.py TESTS/test_risk_parity.py \
  TESTS/test_hrp.py TESTS/test_min_cvar.py TESTS/test_black_litterman.py \
  TESTS/test_frontera.py -v
```

Expected: PASS.

```bash
git add OPTIMIZACION TESTS
git commit -m "feat(portfolio): complete seven allocation methods"
```

Ejecutar los siete métodos sobre la cesta real, imprimir suma, mínimo y máximo
de pesos y detenerse si cualquiera incumple tolerancias.

# Fase 4 — Riesgo y walk-forward

### Task 14: Métricas históricas

**Files:**
- Create: `PANEL PORTFOLIO/RIESGO/metricas.py`
- Create: `PANEL PORTFOLIO/TESTS/test_metricas.py`
- Verify: `PANEL PORTFOLIO/CONTRATOS/modelos.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_var_cvar_historicos():
    retornos = pd.Series([-0.10, -0.04, 0.01, 0.02])
    var, cvar = var_cvar_historicos(retornos, nivel=0.75)
    assert var == pytest.approx(0.055)
    assert cvar == pytest.approx(0.10)


def test_drawdown_y_recuperacion():
    equity = pd.Series([1.0, 1.2, 0.8, 1.21], index=pd.date_range("2024-01-01", periods=4))
    resultado = analizar_drawdown(equity)
    assert resultado.max_drawdown == pytest.approx(0.8 / 1.2 - 1)
    assert resultado.fecha_recuperacion == equity.index[-1]
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_metricas.py -v`

Expected: FAIL por módulo ausente.

- [ ] **Step 3: Implementar métricas**

Usar retornos simples OOS. VaR será el negativo del cuantil inferior; CVaR el
negativo de la media de retornos bajo ese cuantil. Calcular retorno geométrico
anual, volatilidad, Sharpe, Sortino, Calmar, drawdown y recuperación.

- [ ] **Step 4: Ejecutar y commit**

Run: `.venv/bin/python -m pytest TESTS/test_metricas.py -v`

Expected: PASS.

```bash
git add RIESGO/metricas.py TESTS/test_metricas.py
git commit -m "feat(portfolio): add historical portfolio risk metrics"
```

### Task 15: Walk-forward OOS con costes

**Files:**
- Create: `PANEL PORTFOLIO/RIESGO/walk_forward.py`
- Create: `PANEL PORTFOLIO/TESTS/test_walk_forward.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_walk_forward_no_entrega_datos_futuros_al_optimizador(datos):
    ventanas = []

    def optimizador(retornos_estimacion, fecha):
        ventanas.append((retornos_estimacion.index.max(), fecha))
        return {"metodo": pd.Series([0.5, 0.5], index=retornos_estimacion.columns)}

    ejecutar_walk_forward(datos, configuracion, optimizador)
    assert all(max_estimacion < fecha for max_estimacion, fecha in ventanas)


def test_coste_se_descuenta_en_primer_dia_oos():
    resultado = ejecutar_walk_forward(datos, config_coste, optimizador_cambio_total)
    primer_rebalanceo = resultado.rebalanceos[1]
    esperado = primer_rebalanceo.rotacion * config_coste.coste_transaccion_pb / 10_000
    assert primer_rebalanceo.coste == pytest.approx(esperado)
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_walk_forward.py -v`

Expected: FAIL por módulo ausente.

- [ ] **Step 3: Implementar calendario y simulación**

Agrupar por `to_period(frecuencia)`, tomar la primera fecha de cada periodo como
rebalanceo, exigir ventana completa anterior y aplicar pesos hasta el siguiente
rebalanceo. Convertir retornos con `np.expm1`, componer equity y descontar coste
en el primer día del tramo. La primera entrada comparará los pesos objetivo con
una cartera inicial en efectivo de pesos cero, por lo que también pagará coste;
los rebalanceos posteriores compararán contra los pesos objetivo anteriores,
tal como define la especificación.

- [ ] **Step 4: Ejecutar y commit**

Run: `.venv/bin/python -m pytest TESTS/test_walk_forward.py -v`

Expected: PASS.

```bash
git add RIESGO/walk_forward.py TESTS/test_walk_forward.py
git commit -m "feat(portfolio): add out of sample walk forward engine"
```

### Task 16: Métricas por régimen, stress y servicio

**Files:**
- Create: `PANEL PORTFOLIO/RIESGO/por_regimen.py`
- Create: `PANEL PORTFOLIO/RIESGO/stress.py`
- Create: `PANEL PORTFOLIO/RIESGO/servicio.py`
- Create: `PANEL PORTFOLIO/TESTS/test_stress.py`
- Modify: `PANEL PORTFOLIO/TESTS/test_regimenes.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_stress_sin_cobertura_es_no_evaluable(retornos_oos):
    resultado = evaluar_stress(retornos_oos, {"2008": ("2008-09-01", "2009-03-31")})
    assert not resultado["2008"].evaluable
    assert resultado["2008"].observaciones == 0


def test_metricas_por_regimen_respetan_indices(retornos_oos, etiquetas):
    resultado = metricas_por_regimen(retornos_oos, etiquetas, dias_anio=252)
    assert resultado["crisis"].observaciones == int((etiquetas == "crisis").sum())
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_stress.py TESTS/test_regimenes.py -v`

Expected: FAIL por módulos ausentes.

- [ ] **Step 3: Implementar régimen y stress**

Intersectar índices de retornos y etiquetas. Stress será evaluable con cinco o
más retornos; calculará retorno acumulado, volatilidad, peor día y drawdown.
Para diversificación de crisis, restringir los retornos de activos al régimen
`crisis`, estimar su covarianza y usar el promedio temporal de los pesos diarios
OOS de cada método en esas mismas fechas. Sobre ambos se calcularán
diversification ratio y número efectivo de apuestas.

- [ ] **Step 4: Implementar servicio**

`evaluar_riesgo` combinará walk-forward, métricas generales, métricas por
régimen, stress y diagnósticos de diversificación.

- [ ] **Step 5: Ejecutar suite de fase y commit**

Run:

```bash
.venv/bin/python -m pytest TESTS/test_metricas.py TESTS/test_walk_forward.py \
  TESTS/test_regimenes.py TESTS/test_stress.py -v
```

Expected: PASS.

```bash
git add RIESGO TESTS
git commit -m "feat(portfolio): evaluate regimes stress and walk forward risk"
```

Comunicar fecha inicial OOS, número de rebalanceos, costes acumulados y
ventanas de stress evaluables antes de comenzar reportes.

# Fase 5 — Reportes y pipeline completo

### Task 17: Manifiesto, Excel y HTML offline

**Files:**
- Create: `PANEL PORTFOLIO/REPORTES/manifiesto.py`
- Create: `PANEL PORTFOLIO/REPORTES/excel.py`
- Create: `PANEL PORTFOLIO/REPORTES/html.py`
- Create: `PANEL PORTFOLIO/REPORTES/servicio.py`
- Create: `PANEL PORTFOLIO/TESTS/test_reportes.py`

- [ ] **Step 1: Escribir pruebas rojas**

```python
def test_html_es_autonomo(tmp_path, paquete_reporte):
    ruta = generar_html(paquete_reporte, tmp_path / "informe.html")
    contenido = ruta.read_text(encoding="utf-8")
    assert "cdn.plot.ly" not in contenido
    assert "El comportamiento pasado" in contenido
    assert "Resultados walk-forward OOS" in contenido


def test_excel_contiene_hojas_obligatorias(tmp_path, paquete_reporte):
    ruta = generar_excel(paquete_reporte, tmp_path / "informe.xlsx")
    hojas = pd.ExcelFile(ruta).sheet_names
    assert hojas == [
        "Configuracion", "Cobertura", "Pesos actuales", "Metricas OOS",
        "Pesos walk-forward", "Rotacion y costes", "Regimenes", "Stress",
        "Correlaciones", "PCA", "Diagnosticos",
    ]
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_reportes.py -v`

Expected: FAIL por módulos ausentes.

- [ ] **Step 3: Implementar manifiesto y Excel**

Serializar dataclasses, fechas, numpy y pandas a JSON determinista. Escribir
hojas con `pd.ExcelWriter(engine="xlsxwriter")`, congelar cabeceras y aplicar
formatos de porcentaje y fechas.

- [ ] **Step 4: Implementar HTML**

Construir secciones en el orden aprobado. Generar gráficos Plotly con
`include_plotlyjs=True` una sola vez y `False` en los siguientes. Incluir tablas
de pesos estimados y métricas OOS separadas y la advertencia metodológica
literal.

- [ ] **Step 5: Ejecutar y commit**

Run: `.venv/bin/python -m pytest TESTS/test_reportes.py -v`

Expected: PASS.

```bash
git add REPORTES TESTS/test_reportes.py
git commit -m "feat(portfolio): generate offline html excel and manifest reports"
```

### Task 18: Pipeline `analizar` e integración

**Files:**
- Modify: `PANEL PORTFOLIO/ejecutar.py`
- Modify: `PANEL PORTFOLIO/TESTS/test_integracion.py`

- [ ] **Step 1: Escribir prueba roja end-to-end sintética**

```python
def test_pipeline_analizar_genera_tres_salidas(tmp_path, configuracion_sintetica):
    resultado = ejecutar_analisis(configuracion_sintetica)
    assert resultado.html.exists()
    assert resultado.excel.exists()
    assert resultado.manifiesto.exists()
```

- [ ] **Step 2: Ejecutar fallo**

Run: `.venv/bin/python -m pytest TESTS/test_integracion.py -v`

Expected: FAIL porque `ejecutar_analisis` todavía detiene la ejecución.

- [ ] **Step 3: Conectar capas**

```python
def ejecutar_analisis(configuracion: Configuracion) -> RutasReporte:
    cierres = cargar_cierres(configuracion.tickers, configuracion.carpeta_historico)
    datos = alinear_y_calcular_retornos(cierres, configuracion.min_retornos_analisis)
    analisis = analizar_datos(datos, configuracion)
    datos_actuales = recortar_datos(datos, configuracion.ventana_estimacion)
    analisis_actual = analizar_datos(datos_actuales, configuracion)
    asignaciones = optimizar_todos(analisis_actual, configuracion)
    riesgo = evaluar_riesgo(datos, analisis, configuracion)
    paquete = construir_paquete_reporte(
        configuracion,
        datos,
        analisis,
        analisis_actual,
        asignaciones,
        riesgo,
    )
    return generar_reportes(paquete, configuracion.carpeta_salidas)
```

- [ ] **Step 4: Ejecutar suite completa**

Run: `.venv/bin/python -m pytest TESTS -v`

Expected: PASS sin tests omitidos.

- [ ] **Step 5: Ejecutar análisis real**

Run: `.venv/bin/python ejecutar.py analizar`

Expected: código 0 y rutas existentes para HTML, Excel y JSON.

- [ ] **Step 6: Verificar offline y contenido**

Run:

```bash
.venv/bin/python -c "from pathlib import Path; p=max(Path('SALIDAS').glob('*/informe.html')); s=p.read_text(); assert 'cdn.plot.ly' not in s; assert 'walk-forward OOS' in s; print(p, len(s))"
```

Expected: ruta del HTML y tamaño mayor de 1 MB por Plotly embebido.

- [ ] **Step 7: Auditoría de aislamiento y legacy**

Run: `rg -n "PANEL BACKTESTING|sys\\.path|forward_fill|ffill\\(" . --glob '*.py'`

Expected: ninguna coincidencia.

Run: `find . -name __pycache__ -o -name .DS_Store`

Expected: ninguna salida después de limpiar artefactos.

Run: `git status --short -- .`

Expected: solo históricos y salidas ignorados; ningún archivo provisional.

- [ ] **Step 8: Commit final**

```bash
git add .
git commit -m "feat(portfolio): complete professional portfolio analysis pipeline"
```

### Task 19: Verificación final y entrega

**Files:**
- Verify only: `PANEL PORTFOLIO/`

- [ ] **Step 1: Ejecutar compilación**

Run: `.venv/bin/python -m compileall -q .`

Expected: código 0.

- [ ] **Step 2: Ejecutar pruebas finales**

Run: `.venv/bin/python -m pytest TESTS -q`

Expected: todas las pruebas pasan.

- [ ] **Step 3: Ejecutar ambos comandos**

Run: `.venv/bin/python ejecutar.py descargar`

Expected: código 0 y resumen completo.

Run: `.venv/bin/python ejecutar.py analizar`

Expected: código 0 y tres salidas.

- [ ] **Step 4: Inspeccionar salidas**

Comprobar que HTML y Excel contienen siete métodos, tres carteras destacadas de
frontera, curvas OOS, correlación media y de cola, PCA, regímenes, stress,
costes y advertencia de no garantía.

- [ ] **Step 5: Comunicar evidencia**

Entregar:

- lista de archivos principales;
- número de pruebas y resultado;
- cobertura real descargada;
- dimensión del calendario intersectado;
- restricciones verificadas para los siete métodos;
- periodo OOS, rebalanceos y costes;
- ventanas stress evaluables;
- rutas de HTML, Excel y manifiesto;
- hash del commit final.

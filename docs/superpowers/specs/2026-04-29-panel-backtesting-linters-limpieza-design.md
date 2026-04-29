# Diseno: linters y limpieza del panel de backtesting

## Alcance

El trabajo se limita a `PANEL BACKTESTING/**`, `run.py` y los archivos de configuracion necesarios en la raiz del repositorio. No incluye `DESCARGADOR/`, scripts de `github/`, PDFs ni paneles pendientes.

El repositorio ya contiene cambios locales sin commitear en archivos del panel. La implementacion debe preservar esos cambios: no se deben revertir, sobrescribir ni reordenar modulos completos sin una razon tecnica concreta.

## Objetivo

Instalar una base profesional de revision de codigo para el panel de backtesting y usarla para limpiar malas practicas reales, codigo obsoleto evidente y problemas detectables automaticamente, manteniendo la logica de backtesting estable.

## Lenguajes detectados

- Python: codigo principal del panel, tests y `run.py`.
- Rust: motor en `PANEL BACKTESTING/MOTOR`.
- C/C++: no hay codigo propio dentro del alcance; los archivos encontrados pertenecen a `.venv` o artefactos de dependencias. No se configurara tooling C/C++ en esta fase.

## Tooling propuesto

### Python

Configurar `pyproject.toml` en la raiz con:

- `ruff` como linter y formatter.
- `pytest` para ejecutar los tests existentes en `PANEL BACKTESTING/TESTS`.
- Exclusiones para `.venv`, `__pycache__`, `PANEL BACKTESTING/MOTOR/target`, `PANEL BACKTESTING/RESULTADOS` y `PANEL BACKTESTING/HISTORICO`.
- Reglas iniciales conservadoras para evitar una reescritura masiva. La primera fase debe priorizar errores objetivos: imports, variables no usadas, sintaxis, patrones peligrosos y estilo automatizable.

Si falta tooling en `.venv`, instalar dependencias de desarrollo alli y reflejarlo en un archivo de dependencias de desarrollo o en extras del `pyproject.toml`.

### Rust

Usar las herramientas nativas ya disponibles:

- `cargo fmt --check` y `cargo fmt`.
- `cargo clippy -- -D warnings`.
- `cargo test` si existen tests o si el crate compila sus checks de forma suficiente para validar cambios.

La configuracion Rust debe permanecer dentro de `PANEL BACKTESTING/MOTOR` salvo que haga falta un comando raiz para orquestar la revision completa.

## Estrategia de limpieza

La limpieza se hara de forma incremental y verificable:

1. Configurar herramientas y comandos reproducibles.
2. Ejecutar linters/checks para obtener una lista objetiva de problemas.
3. Corregir primero fallos automaticos y de bajo riesgo.
4. Revisar manualmente advertencias que puedan afectar a comportamiento, especialmente en estrategias, gestion de riesgo, salidas y motor.
5. Mantener o mejorar los tests existentes antes de tocar logica sensible.

No se aceptan cambios cosmeticos extensos que oculten la revision funcional. Cualquier refactor debe tener una razon concreta: reducir duplicacion real, eliminar codigo muerto, simplificar una ruta confusa o corregir una mala practica detectada.

## Validacion

La implementacion se considerara lista cuando pasen, o cuando quede documentado claramente por que no se pudieron ejecutar:

- `ruff check` sobre `run.py` y `PANEL BACKTESTING`.
- `ruff format --check` o formato aplicado con `ruff format`.
- Tests Python del panel con el interprete de `.venv`.
- `cargo fmt --check` en `PANEL BACKTESTING/MOTOR`.
- `cargo clippy -- -D warnings` en `PANEL BACKTESTING/MOTOR`.

## Riesgos y mitigaciones

- **Cambios locales existentes:** tocar solo lo necesario y revisar diffs antes de editar archivos ya modificados.
- **Nombre de carpeta con espacio:** configurar comandos y paths con comillas o desde el directorio correcto.
- **Dependencias faltantes:** instalar solo dependencias de desarrollo necesarias en `.venv`.
- **Logica de trading sensible:** evitar reescrituras amplias; cualquier cambio de comportamiento debe estar cubierto por tests.
- **Reglas demasiado estrictas al inicio:** empezar con una configuracion razonable y endurecerla despues de estabilizar el codigo.

## Resultado esperado

Al terminar, el panel tendra una forma unica y repetible de revisar calidad de codigo para Python y Rust. La primera limpieza dejara el codigo en mejor estado sin introducir deuda nueva ni romper el flujo de ejecucion desde `run.py`.

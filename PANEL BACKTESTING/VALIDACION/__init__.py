"""Motor de validación fuera de muestra (Fase 2 + veredicto de Fase 3).

Convierte el optimizador de curvas en una máquina de estimación honesta:
  - `cpcv`         → Combinatorial Purged Cross-Validation (López de Prado).
  - `wfa`          → Walk-Forward Analysis (rolling y anchored) + efficiency.
  - `distribucion` → estadísticos de la distribución de métricas OOS.
  - `orquestador`  → ejecuta CPCV/WFA con callbacks de optimización/evaluación
                     inyectados (desacopla la lógica del motor).
  - `veredicto`    → umbrales a priori y semáforo 🟢/🟡/🔴 (Fase 3/4).

El núcleo combinatorio es matemática pura sobre índices/NumPy: no importa Polars
ni el motor, por lo que es verificable de forma aislada. El acoplamiento con
Optuna y el motor Rust se hace por inyección de funciones en `orquestador`.
"""

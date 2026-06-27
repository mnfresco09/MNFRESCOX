"""Batería de robustez (Fase 5): responde a "¿y si tuve suerte?" desde varios
ángulos.

Una curva de equity es UNA realización de un proceso estocástico. Estos módulos
cuantifican cuánta de la rentabilidad fue habilidad y cuánta fue el orden
afortunado de los trades o el azar del periodo:

  - `bootstrap`    → remuestrea la secuencia de trades → distribución de equity
                     final, max drawdown y Sharpe (las métricas dependientes del
                     camino NECESITAN esto).
  - `regimen`      → rendimiento separado por régimen (alcista/bajista/lateral).
  - `nula`         → estrategia nula / línea base de ruido procesada por la misma
                     maquinaria: el control de laboratorio que separa el rigor del
                     autoengaño.
  - `sensibilidad` → sensibilidad a la fecha de inicio.

El criterio de "supervivencia a 2× costes" del documento NO está aquí: depende
del modelo de costes de ejecución, fuera de este alcance.
"""

"""Sizing por riesgo y evaluación de cartera (Fase 6).

  - `sizing`  → volatility targeting (riesgo constante en vez de notional
                constante) y Kelly fraccional como referencia de sizing óptimo.
  - `cartera` → evaluación a nivel de cartera: contribución marginal al Sharpe
                considerando correlaciones. Dos estrategias mediocres
                descorrelacionadas pueden ser una cartera excelente; dos buenas
                correlacionadas son redundantes.

Matemática pura sobre NumPy: no toca el motor. Es la capa que se aplica ANTES del
motor (sizing) y DESPUÉS (evaluación de cartera).
"""

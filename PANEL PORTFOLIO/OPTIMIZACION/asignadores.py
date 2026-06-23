"""Fachada de OPTIMIZACION: ejecuta los 7 métodos de asignación.

Entrada: el análisis de la VENTANA ACTUAL (momentos + covarianza Ledoit-Wolf +
log-retornos como escenarios) y la configuración. Salida: un dict
{nombre_método: ResultadoAsignacion}, cada uno ya validado (pesos suman 1 y
respetan restricciones).

Los 7 métodos:
  1. Markowitz (máx Sharpe)   2. Mínima varianza   3. Risk parity
  4. HRP                      5. Min-CVaR          6. Black-Litterman
  7. Máxima diversificación
"""

from __future__ import annotations

import pandas as pd

from CONTRATOS.modelos import Configuracion, ResultadoAnalisis, ResultadoAsignacion

from ANALISIS.momentos import covarianza_ledoit_wolf, retornos_esperados

from . import markowitz, risk_parity as rp_mod
from .black_litterman import black_litterman
from .comun import aplicar_tope_y_renormalizar, asignacion
from .cvar import min_cvar
from .hrp import hrp
from .max_diversificacion import max_diversificacion

# Nombres canónicos de los 6 métodos (orden estable para tablas y reportes).
METODOS = (
    "Markowitz (máx Sharpe)",
    "Mínima varianza",
    "Risk parity",
    "HRP",
    "Min-CVaR",
    "Black-Litterman",
    "Máxima diversificación",
)


def calcular_pesos(
    log_retornos: pd.DataFrame,
    configuracion: Configuracion,
) -> dict[str, pd.Series]:
    """Pesos de los 6 métodos para una ventana de retornos (sin empaquetar métricas).

    Es la versión ligera que usa el walk-forward en cada rebalanceo: estima
    momentos y covarianza Ledoit-Wolf sobre la ventana y resuelve los 6 métodos.
    """
    restr = configuracion.restricciones
    rf = configuracion.tasa_libre_riesgo_anual
    mu = retornos_esperados(log_retornos, configuracion.dias_anio)
    cov, _ = covarianza_ledoit_wolf(log_retornos, configuracion.dias_anio)
    pesos_hrp, _ = aplicar_tope_y_renormalizar(hrp(cov), restr, "HRP")
    pesos_bl, _ = black_litterman(mu, cov, restr, configuracion)
    return {
        "Markowitz (máx Sharpe)": markowitz.maximo_sharpe(mu, cov, rf, restr),
        "Mínima varianza": markowitz.minima_varianza(cov, restr),
        "Risk parity": rp_mod.risk_parity(cov, restr),
        "HRP": pesos_hrp,
        "Min-CVaR": min_cvar(log_retornos, restr, configuracion.nivel_confianza),
        "Black-Litterman": pesos_bl,
        "Máxima diversificación": max_diversificacion(cov, restr),
    }


def asignar_todos(
    analisis: ResultadoAnalisis,
    configuracion: Configuracion,
) -> dict[str, ResultadoAsignacion]:
    mu = analisis.retornos_esperados
    cov = analisis.covarianza
    restr = configuracion.restricciones
    rf = configuracion.tasa_libre_riesgo_anual
    resultados: dict[str, ResultadoAsignacion] = {}

    # 1. Markowitz máximo Sharpe.
    resultados["Markowitz (máx Sharpe)"] = asignacion(
        "Markowitz (máx Sharpe)",
        markowitz.maximo_sharpe(mu, cov, rf, restr),
        mu, cov, rf, restr, "SLSQP",
        "Cartera tangente de máximo Sharpe sobre covarianza Ledoit-Wolf.",
    )

    # 2. Mínima varianza global.
    resultados["Mínima varianza"] = asignacion(
        "Mínima varianza",
        markowitz.minima_varianza(cov, restr),
        mu, cov, rf, restr, "SLSQP",
        "Cartera de mínima varianza global.",
    )

    # 3. Risk parity (contribuciones de riesgo iguales).
    resultados["Risk parity"] = asignacion(
        "Risk parity",
        rp_mod.risk_parity(cov, restr),
        mu, cov, rf, restr, "SLSQP",
        "Igual contribución al riesgo por activo (vía optimización).",
    )

    # 4. HRP (con tope aplicado fuera del algoritmo).
    pesos_hrp, avisos_hrp = aplicar_tope_y_renormalizar(hrp(cov), restr, "HRP")
    resultados["HRP"] = asignacion(
        "HRP",
        pesos_hrp,
        mu, cov, rf, restr, "jerárquico",
        "Hierarchical Risk Parity (clustering + bisección, sin invertir Σ).",
        avisos_hrp,
    )

    # 5. Min-CVaR sobre escenarios reales (los log-retornos de la ventana).
    resultados["Min-CVaR"] = asignacion(
        "Min-CVaR",
        min_cvar(analisis.log_retornos, restr, configuracion.nivel_confianza),
        mu, cov, rf, restr, "cvxpy/CLARABEL",
        f"Mínimo CVaR al {configuracion.nivel_confianza:.0%} (Rockafellar-Uryasev, LP).",
    )

    # 6. Black-Litterman (equilibrio + views; cae a mercado si no hay views).
    pesos_bl, diag_bl = black_litterman(mu, cov, restr, configuracion)
    resultados["Black-Litterman"] = asignacion(
        "Black-Litterman", pesos_bl, mu, cov, rf, restr, "SLSQP", diag_bl,
    )

    # 7. Máxima diversificación (premia activos anticorrelados: si algo baja y
    #    algo sube, la cartera tiende a compensarse).
    resultados["Máxima diversificación"] = asignacion(
        "Máxima diversificación",
        max_diversificacion(cov, restr),
        mu, cov, rf, restr, "SLSQP",
        "Maximiza el ratio de diversificación (Choueifaty): explota la anticorrelación.",
    )

    return resultados

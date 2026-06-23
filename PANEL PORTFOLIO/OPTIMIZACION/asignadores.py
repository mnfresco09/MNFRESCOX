"""Fachada de OPTIMIZACION: ejecuta los métodos de asignación.

Entrada: el análisis de la VENTANA ACTUAL (momentos + covarianza Ledoit-Wolf +
log-retornos como escenarios) y la configuración. Salida: un dict
{nombre_método: ResultadoAsignacion}, cada uno ya validado (pesos suman 1 y
respetan restricciones).

Los métodos se agrupan en dos familias (ver `grupos_metodos`):

  NÚCLEO ROBUSTO — NO estiman el retorno esperado μ (el insumo más ruidoso), por
  eso aguantan bien fuera de muestra:
    1. Equiponderada (1/N)   2. Mínima varianza   3. Risk parity
    4. HRP                   5. Máxima diversificación   6. Min-CVaR

  DIAGNÓSTICO — persiguen retorno y son frágiles fuera de muestra; se muestran
  como referencia honesta (NO como recomendación):
    · Markowitz (máx Sharpe)
    · Black-Litterman, SOLO si hay views en la configuración. Sin views,
      Black-Litterman es idéntico por construcción a la equiponderada (1/N), así
      que NO se calcula para no duplicar la misma cartera con dos nombres.
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

NOMBRE_1N = "Equiponderada (1/N)"

# Familias estables (orden de presentación).
METODOS_NUCLEO = (
    NOMBRE_1N,
    "Mínima varianza",
    "Risk parity",
    "HRP",
    "Máxima diversificación",
    "Min-CVaR",
)
METODOS_DIAGNOSTICO_BASE = ("Markowitz (máx Sharpe)",)


def grupos_metodos(configuracion: Configuracion) -> dict[str, tuple[str, ...]]:
    """Devuelve {'nucleo': (...), 'diagnostico': (...)} según haya views o no."""
    diagnostico = list(METODOS_DIAGNOSTICO_BASE)
    if configuracion.views_black_litterman:
        diagnostico.append("Black-Litterman")
    return {"nucleo": METODOS_NUCLEO, "diagnostico": tuple(diagnostico)}


def metodos(configuracion: Configuracion) -> tuple[str, ...]:
    """Lista ordenada de métodos activos (núcleo + diagnóstico)."""
    g = grupos_metodos(configuracion)
    return g["nucleo"] + g["diagnostico"]


def equiponderada(activos) -> pd.Series:
    """Cartera 1/N: reparto igual entre todos los activos."""
    n = len(activos)
    return pd.Series(1.0 / n, index=list(activos))


def calcular_pesos(
    log_retornos: pd.DataFrame,
    configuracion: Configuracion,
) -> dict[str, pd.Series]:
    """Pesos de los métodos activos para una ventana (versión ligera del walk-forward)."""
    restr = configuracion.restricciones
    rf = configuracion.tasa_libre_riesgo_anual
    mu = retornos_esperados(log_retornos, configuracion.dias_anio)
    cov, _ = covarianza_ledoit_wolf(log_retornos, configuracion.dias_anio)
    pesos_hrp, _ = aplicar_tope_y_renormalizar(hrp(cov), restr, "HRP")
    pesos: dict[str, pd.Series] = {
        NOMBRE_1N: equiponderada(cov.index),
        "Mínima varianza": markowitz.minima_varianza(cov, restr),
        "Risk parity": rp_mod.risk_parity(cov, restr),
        "HRP": pesos_hrp,
        "Máxima diversificación": max_diversificacion(cov, restr),
        "Min-CVaR": min_cvar(log_retornos, restr, configuracion.nivel_confianza),
        "Markowitz (máx Sharpe)": markowitz.maximo_sharpe(mu, cov, rf, restr),
    }
    if configuracion.views_black_litterman:
        pesos_bl, _ = black_litterman(mu, cov, restr, configuracion)
        pesos["Black-Litterman"] = pesos_bl
    return pesos


def asignar_todos(
    analisis: ResultadoAnalisis,
    configuracion: Configuracion,
) -> dict[str, ResultadoAsignacion]:
    mu = analisis.retornos_esperados
    cov = analisis.covarianza
    restr = configuracion.restricciones
    rf = configuracion.tasa_libre_riesgo_anual
    resultados: dict[str, ResultadoAsignacion] = {}

    # --- NÚCLEO ROBUSTO -----------------------------------------------------
    resultados[NOMBRE_1N] = asignacion(
        NOMBRE_1N, equiponderada(cov.index), mu, cov, rf, restr, "cerrado",
        "Reparto igual entre todos los activos (1/N): benchmark robusto que no estima retornos.",
    )
    resultados["Mínima varianza"] = asignacion(
        "Mínima varianza", markowitz.minima_varianza(cov, restr),
        mu, cov, rf, restr, "SLSQP", "Cartera de mínima varianza global.",
    )
    resultados["Risk parity"] = asignacion(
        "Risk parity", rp_mod.risk_parity(cov, restr),
        mu, cov, rf, restr, "SLSQP", "Igual contribución al riesgo por activo (vía optimización).",
    )
    pesos_hrp, avisos_hrp = aplicar_tope_y_renormalizar(hrp(cov), restr, "HRP")
    resultados["HRP"] = asignacion(
        "HRP", pesos_hrp, mu, cov, rf, restr, "jerárquico",
        "Hierarchical Risk Parity (clustering + bisección, sin invertir Σ).", avisos_hrp,
    )
    resultados["Máxima diversificación"] = asignacion(
        "Máxima diversificación", max_diversificacion(cov, restr),
        mu, cov, rf, restr, "SLSQP",
        "Maximiza el ratio de diversificación (Choueifaty): explota la anticorrelación.",
    )
    resultados["Min-CVaR"] = asignacion(
        "Min-CVaR", min_cvar(analisis.log_retornos, restr, configuracion.nivel_confianza),
        mu, cov, rf, restr, "cvxpy/CLARABEL",
        f"Mínimo CVaR al {configuracion.nivel_confianza:.0%} (Rockafellar-Uryasev, LP).",
    )

    # --- DIAGNÓSTICO (referencia honesta, no recomendación) -----------------
    resultados["Markowitz (máx Sharpe)"] = asignacion(
        "Markowitz (máx Sharpe)", markowitz.maximo_sharpe(mu, cov, rf, restr),
        mu, cov, rf, restr, "SLSQP",
        "Cartera tangente de máximo Sharpe (persigue retorno; frágil fuera de muestra).",
    )
    if configuracion.views_black_litterman:
        pesos_bl, diag_bl = black_litterman(mu, cov, restr, configuracion)
        resultados["Black-Litterman"] = asignacion(
            "Black-Litterman", pesos_bl, mu, cov, rf, restr, "SLSQP", diag_bl,
        )

    return resultados

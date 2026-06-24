"""Momentos anualizados y DOBLE LENTE de covarianza.

Pregunta 2 del panel: ¿cómo se relacionan los activos? Se estiman DOS matrices:

  • ESTRUCTURAL (Ledoit-Wolf): encoge la covarianza muestral hacia un objetivo
    estructurado → bien condicionada y definida positiva. Es la que se INVIERTE
    al optimizar (Markowitz, risk parity), por eso debe ser estable.

  • TÁCTICA (EWMA, RiskMetrics λ=0.94): pondera más los retornos recientes →
    captura el riesgo de MAÑANA (T+1). Es la que alimenta el forecast de VaR.

Convenciones: log-retornos, anualización ×DIAS_ANIO, todo en términos anuales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from CONTRATOS.errores import ErrorAnalisis


# --- Lente estructural: Ledoit-Wolf ----------------------------------------
def covarianza_ledoit_wolf(log_retornos: pd.DataFrame, dias_anio: int) -> tuple[pd.DataFrame, float]:
    """Devuelve (covarianza anual estructural, coeficiente de encogimiento)."""
    if log_retornos.shape[0] < 2:
        raise ErrorAnalisis("Se necesitan al menos 2 retornos para la covarianza.")
    estimador = LedoitWolf().fit(log_retornos.to_numpy())
    cov_anual = np.asarray(estimador.covariance_, dtype=float) * float(dias_anio)
    activos = list(log_retornos.columns)
    cov = pd.DataFrame(cov_anual, index=activos, columns=activos)
    sim = (cov.to_numpy() + cov.to_numpy().T) / 2.0
    if np.linalg.eigvalsh(sim).min() <= -1e-10:
        raise ErrorAnalisis("La covarianza Ledoit-Wolf no es definida positiva.")
    return cov, float(estimador.shrinkage_)


def covarianza_muestral(log_retornos: pd.DataFrame, dias_anio: int) -> pd.DataFrame:
    """Covarianza muestral cruda anualizada (solo para diagnóstico/fallback)."""
    activos = list(log_retornos.columns)
    cov = np.cov(log_retornos.to_numpy(), rowvar=False) * float(dias_anio)
    return pd.DataFrame(np.atleast_2d(cov), index=activos, columns=activos)


# --- Lente táctica: EWMA (RiskMetrics) -------------------------------------
def covarianza_ewma(
    log_retornos: pd.DataFrame,
    dias_anio: int,
    lam: float = 0.94,
) -> pd.DataFrame:
    """Covarianza EWMA anualizada (riesgo táctico T+1).

    Σ_t = (1-λ)·rᵀr ponderado exponencialmente. Pesos w_i ∝ λ^i sobre los
    retornos centrados, del más reciente (mayor peso) al más antiguo.
    """
    r = log_retornos.to_numpy(dtype=float)
    n = r.shape[0]
    if n < 2:
        raise ErrorAnalisis("EWMA requiere al menos 2 retornos.")
    r = r - r.mean(axis=0, keepdims=True)
    # Pesos: el último (más reciente) recibe el mayor peso.
    edades = np.arange(n)[::-1]                      # n-1 ... 0
    pesos = (1.0 - lam) * lam ** edades
    pesos /= pesos.sum()
    cov_diaria = (r * pesos[:, None]).T @ r
    cov_diaria = (cov_diaria + cov_diaria.T) / 2.0
    activos = list(log_retornos.columns)
    return pd.DataFrame(cov_diaria * float(dias_anio), index=activos, columns=activos)


def volatilidades(covarianza_anual: pd.DataFrame) -> pd.Series:
    """Volatilidad anual por activo = raíz de la diagonal de la covarianza."""
    return pd.Series(np.sqrt(np.diag(covarianza_anual.to_numpy())), index=covarianza_anual.index)


# --- Retornos esperados ----------------------------------------------------
def retornos_medios(log_retornos: pd.DataFrame, dias_anio: int) -> pd.Series:
    """Retorno esperado anual histórico crudo (proxy, NO predicción)."""
    return log_retornos.mean() * float(dias_anio)


def retornos_shrinkage(
    log_retornos: pd.DataFrame,
    dias_anio: int,
    intensidad: float,
) -> pd.Series:
    """Estimador conservador de retorno: encoge la media histórica hacia la gran
    media transversal (James-Stein-like).

        μ_shrunk = (1-δ)·μ_i + δ·μ̄

    Reduce el insumo más ruidoso (μ) sin inventar señal. δ=intensidad ∈ [0, 1].
    """
    mu = retornos_medios(log_retornos, dias_anio)
    gran_media = float(mu.mean())
    delta = float(min(max(intensidad, 0.0), 1.0))
    return (1.0 - delta) * mu + delta * gran_media


# --- Estadística individual por activo -------------------------------------
def estadistica_por_activo(
    log_retornos: pd.DataFrame,
    mu_medio: pd.Series,
    mu_ajustado: pd.Series,
    vols: pd.Series,
    vols_tacticas: pd.Series,
) -> dict[str, dict[str, float]]:
    """Asimetría y curtosis (exceso) por activo, más los momentos ya calculados."""
    salida: dict[str, dict[str, float]] = {}
    for activo in log_retornos.columns:
        serie = log_retornos[activo].to_numpy(dtype=float)
        s = serie - serie.mean()
        sd = s.std(ddof=0)
        asimetria = float(np.mean(s ** 3) / sd ** 3) if sd > 0 else 0.0
        curtosis = float(np.mean(s ** 4) / sd ** 4 - 3.0) if sd > 0 else 0.0
        salida[activo] = dict(
            retorno_medio=float(mu_medio[activo]),
            retorno_ajustado=float(mu_ajustado[activo]),
            volatilidad=float(vols[activo]),
            volatilidad_tactica=float(vols_tacticas[activo]),
            asimetria=asimetria,
            curtosis=curtosis,
        )
    return salida

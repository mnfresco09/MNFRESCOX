"""Momentos anualizados y covarianza Ledoit-Wolf.

Responsabilidad: a partir de los log-retornos diarios alineados, estimar el
retorno esperado, la volatilidad y la matriz de covarianza ANUALIZADOS.

Regla innegociable del panel: la covarianza se estima con el encogimiento de
Ledoit-Wolf (sklearn), NUNCA con la covarianza muestral cruda. La muestral es
ruidosa y casi singular cuando hay pocos datos por activo, y al invertirla
(Markowitz, risk parity) amplifica ese ruido en pesos extremos e inestables.
Ledoit-Wolf encoge la muestral hacia un objetivo estructurado y produce una
matriz bien condicionada y definida positiva.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from CONTRATOS.errores import ErrorAnalisis


def covarianza_ledoit_wolf(log_retornos: pd.DataFrame, dias_anio: int) -> tuple[pd.DataFrame, float]:
    """Devuelve (covarianza anualizada, coeficiente de encogimiento)."""
    if log_retornos.shape[0] < 2:
        raise ErrorAnalisis("Se necesitan al menos 2 retornos para la covarianza.")
    estimador = LedoitWolf().fit(log_retornos.to_numpy())
    cov_diaria = np.asarray(estimador.covariance_, dtype=float)
    cov_anual = cov_diaria * float(dias_anio)
    activos = list(log_retornos.columns)
    cov = pd.DataFrame(cov_anual, index=activos, columns=activos)
    # Comprobación de salud: simétrica y definida positiva.
    autovalores = np.linalg.eigvalsh((cov.to_numpy() + cov.to_numpy().T) / 2.0)
    if autovalores.min() <= -1e-10:
        raise ErrorAnalisis("La covarianza Ledoit-Wolf no es definida positiva.")
    return cov, float(estimador.shrinkage_)


def retornos_esperados(log_retornos: pd.DataFrame, dias_anio: int) -> pd.Series:
    """Retorno esperado anualizado = media de log-retornos diarios × días/año.

    Es la estimación histórica simple. Se documenta como tal: es un proxy del
    retorno futuro, no una predicción. Black-Litterman, más abajo, ofrece una
    alternativa basada en el equilibrio de mercado.
    """
    return log_retornos.mean() * float(dias_anio)


def volatilidades(covarianza_anual: pd.DataFrame) -> pd.Series:
    """Volatilidad anualizada por activo = raíz de la diagonal de la covarianza."""
    return pd.Series(np.sqrt(np.diag(covarianza_anual.to_numpy())), index=covarianza_anual.index)

"""Diversificación real de una cartera: ratio de diversificación y número
efectivo de apuestas (Meucci).

- RATIO DE DIVERSIFICACIÓN (Choueifaty): suma ponderada de volatilidades
  individuales dividida por la volatilidad de la cartera. Vale 1 si todos los
  activos están perfectamente correlacionados; crece cuanto más independientes.
- NÚMERO EFECTIVO DE APUESTAS (Meucci): diagonaliza la covarianza en carteras
  principales no correlacionadas, mide cómo se reparte la varianza entre ellas
  y calcula el "número efectivo" como la exponencial de la entropía de ese
  reparto. Responde: ¿cuántas apuestas GENUINAS e independientes tengo de hecho?
  (entre 1 y nº de activos).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorAnalisis


def _alinear(pesos: pd.Series, covarianza: pd.DataFrame) -> np.ndarray:
    w = pesos.reindex(covarianza.index)
    if w.isna().any():
        raise ErrorAnalisis("Los pesos no cubren todos los activos de la covarianza.")
    return w.to_numpy(dtype=float)


def ratio_diversificacion(pesos: pd.Series, covarianza: pd.DataFrame) -> float:
    w = _alinear(pesos, covarianza)
    cov = covarianza.to_numpy(dtype=float)
    vol_individual = np.sqrt(np.diag(cov))
    vol_cartera = float(np.sqrt(w @ cov @ w))
    if vol_cartera <= 0:
        raise ErrorAnalisis("Volatilidad de cartera no positiva en el ratio de diversificación.")
    return float((w @ vol_individual) / vol_cartera)


def numero_efectivo_apuestas(pesos: pd.Series, covarianza: pd.DataFrame) -> float:
    w = _alinear(pesos, covarianza)
    cov = covarianza.to_numpy(dtype=float)
    autovalores, autovectores = np.linalg.eigh((cov + cov.T) / 2.0)
    autovalores = np.clip(autovalores, 0.0, None)
    exposiciones = autovectores.T @ w                  # pesos sobre carteras principales
    varianzas = (exposiciones ** 2) * autovalores      # varianza aportada por cada una
    total = varianzas.sum()
    if total <= 0:
        raise ErrorAnalisis("Varianza total no positiva en el número efectivo de apuestas.")
    proporciones = varianzas / total
    proporciones = proporciones[proporciones > 1e-15]
    entropia = float(-(proporciones * np.log(proporciones)).sum())
    return float(np.exp(entropia))


def diversificacion(pesos: pd.Series, covarianza: pd.DataFrame) -> dict[str, float]:
    """Atajo: devuelve ambas medidas en un dict."""
    return {
        "ratio_diversificacion": ratio_diversificacion(pesos, covarianza),
        "numero_efectivo_apuestas": numero_efectivo_apuestas(pesos, covarianza),
    }

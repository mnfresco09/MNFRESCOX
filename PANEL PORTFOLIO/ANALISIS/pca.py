"""PCA sobre los retornos para diagnosticar diversificación de factores.

Si pocas componentes explican casi toda la varianza, los activos no son apuestas
genuinamente distintas sino el mismo factor repetido. Se hace sobre la matriz de
CORRELACIÓN (retornos estandarizados) por descomposición propia: transparente y
sin dependencias opacas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorAnalisis
from CONTRATOS.modelos import ResultadoPCA


def calcular_pca(log_retornos: pd.DataFrame) -> ResultadoPCA:
    activos = list(log_retornos.columns)
    correlacion = log_retornos.corr().to_numpy()
    if not np.isfinite(correlacion).all():
        raise ErrorAnalisis("La correlación contiene valores no finitos; no se puede hacer PCA.")

    # eigh: matriz simétrica → autovalores reales, orden ascendente. Invertimos.
    autovalores, autovectores = np.linalg.eigh(correlacion)
    orden = np.argsort(autovalores)[::-1]
    autovalores = np.clip(autovalores[orden], 0.0, None)
    autovectores = autovectores[:, orden]

    total = autovalores.sum()
    if total <= 0:
        raise ErrorAnalisis("Varianza total nula en PCA.")
    varianza_explicada = autovalores / total
    nombres = [f"PC{i + 1}" for i in range(len(activos))]
    return ResultadoPCA(
        varianza_explicada=pd.Series(varianza_explicada, index=nombres),
        varianza_acumulada=pd.Series(np.cumsum(varianza_explicada), index=nombres),
        cargas=pd.DataFrame(autovectores, index=activos, columns=nombres),
    )

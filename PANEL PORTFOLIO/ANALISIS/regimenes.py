"""Etiquetado de regímenes de mercado por REGLAS transparentes.

Nada de modelos ocultos (ni Markov ni clustering opaco): cada día se clasifica
con reglas claras y auditables sobre el activo de referencia (p. ej. ^GSPC):

  - drawdown desde máximos,
  - posición frente a su media móvil larga y la pendiente de esa media,
  - volatilidad reciente frente a su propio percentil alto.

Prioridad (de más grave a más benigno): CRISIS → BAJISTA → ALCISTA → LATERAL.
Es análisis DESCRIPTIVO del pasado, no una predicción.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorAnalisis
from CONTRATOS.modelos import ParametrosRegimen

ALCISTA = "ALCISTA"
LATERAL = "LATERAL"
BAJISTA = "BAJISTA"
CRISIS = "CRISIS"


def etiquetar_regimenes(precios_referencia: pd.Series, parametros: ParametrosRegimen) -> pd.Series:
    """Devuelve una Serie de etiquetas de régimen indexada como los precios."""
    if not isinstance(precios_referencia.index, pd.DatetimeIndex):
        raise ErrorAnalisis("La serie de referencia debe estar indexada por fechas.")
    if len(precios_referencia) < parametros.ventana_volatilidad + 2:
        raise ErrorAnalisis("Serie de referencia demasiado corta para etiquetar regímenes.")

    precio = precios_referencia.astype(float)
    maximo = precio.cummax()
    drawdown = precio / maximo - 1.0

    media_larga = precio.rolling(parametros.ventana_media_larga, min_periods=1).mean()
    pendiente = media_larga.diff(parametros.ventana_pendiente)

    retornos = np.log(precio / precio.shift(1))
    volatilidad = retornos.rolling(parametros.ventana_volatilidad).std()
    umbral_vol = volatilidad.quantile(parametros.percentil_volatilidad_crisis)

    # Condiciones booleanas (NaN se comporta como False en las comparaciones).
    es_crisis = (drawdown <= parametros.drawdown_crisis) | (volatilidad >= umbral_vol)
    es_bajista = (drawdown <= parametros.drawdown_bajista) | (
        (precio < media_larga) & (pendiente < 0)
    )
    es_alcista = (precio > media_larga) & (pendiente > 0)

    etiquetas = np.select(
        [es_crisis.to_numpy(), es_bajista.to_numpy(), es_alcista.to_numpy()],
        [CRISIS, BAJISTA, ALCISTA],
        default=LATERAL,
    )
    return pd.Series(etiquetas, index=precio.index, name="regimen")

"""Análisis de convexidad / anti-caída: ¿qué hace cada cartera en cada escenario?

Responde a la pregunta del objetivo "si todo baja me quedo igual; si algo baja y
algo sube quedo casi igual; si todo sube, subo mucho". Clasifica cada día OOS por
lo que hicieron los ACTIVOS de la cesta:

  · TODO BAJA  : todos los activos cayeron ese día.
  · MIXTO      : unos subieron y otros bajaron (aquí se ve la compensación).
  · TODO SUBE  : todos subieron.

Para cada método calcula el retorno medio diario de la cartera en cada escenario
y, además, la captura alcista y bajista frente al mercado de referencia:

  · captura_alcista = media de la cartera en días de subida del referente /
                      media del referente en esos días.
  · captura_bajista = ídem en días de bajada (cuanto MENOR, mejor protección).
  · asimetria        = captura_alcista − captura_bajista (positiva = convexa).

Es DESCRIPTIVO del pasado OOS: que un método se comportara así no garantiza que
lo repita. Y con pesos fijos no se puede garantizar "plano al bajar y mucho al
subir"; eso requeriría opciones o cobertura dinámica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _captura(cartera: pd.Series, referente: pd.Series, signo: str) -> float:
    mask = referente > 0 if signo == "alza" else referente < 0
    if mask.sum() == 0:
        return float("nan")
    ref_media = referente[mask].mean()
    if ref_media == 0:
        return float("nan")
    return float(cartera[mask].mean() / ref_media)


def analizar_convexidad(
    retornos_cartera: pd.DataFrame,
    retornos_activos: pd.DataFrame,
    activo_referencia: str,
) -> pd.DataFrame:
    """Tabla por método con retorno medio en cada escenario y capturas alza/baja.

    `retornos_cartera`: retornos OOS diarios por método (walk-forward).
    `retornos_activos`: retornos diarios de los activos (log-retornos alineados).
    """
    activos = retornos_activos.reindex(retornos_cartera.index).dropna()
    cartera = retornos_cartera.reindex(activos.index)

    n = activos.shape[1]
    n_pos = (activos > 0).sum(axis=1)
    escenario = pd.Series(
        np.where(n_pos == 0, "todo_baja", np.where(n_pos == n, "todo_sube", "mixto")),
        index=activos.index,
    )

    referente = activos[activo_referencia] if activo_referencia in activos.columns else activos.mean(axis=1)

    filas: dict[str, dict] = {}
    for metodo in cartera.columns:
        serie = cartera[metodo]
        fila = {}
        for etiqueta in ("todo_baja", "mixto", "todo_sube"):
            sub = serie[escenario == etiqueta]
            fila[f"ret_medio_{etiqueta}"] = float(sub.mean()) if len(sub) else float("nan")
            fila[f"dias_{etiqueta}"] = int(len(sub))
        fila["captura_alcista"] = _captura(serie, referente, "alza")
        fila["captura_bajista"] = _captura(serie, referente, "baja")
        fila["asimetria"] = fila["captura_alcista"] - fila["captura_bajista"]
        filas[metodo] = fila
    return pd.DataFrame.from_dict(filas, orient="index")

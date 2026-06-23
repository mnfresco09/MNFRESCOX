"""Resuelve el objetivo elegido al inicio en una recomendación HONESTA.

El objetivo define con qué criterio se juzga y qué método se destaca. La clave:
el método se elige por su comportamiento OUT-OF-SAMPLE (lo realizado), no por la
promesa in-sample. Así se evita coronar al que "ganaba" solo en el papel.
"""

from __future__ import annotations

from dataclasses import dataclass

from CONTRATOS.modelos import PaqueteReporte

from .formato import OBJETIVOS, nombre_visible, num, pct


@dataclass(frozen=True)
class Recomendacion:
    objetivo: str            # clave: sharpe / riesgo / objetivo / convexidad / comparar
    etiqueta_objetivo: str   # texto legible
    metodo: str              # clave interna del método recomendado
    criterio: str            # por qué se eligió
    detalle: str             # frase con los números


def recomendar(paquete: PaqueteReporte) -> Recomendacion:
    objetivo = (paquete.objetivo or "comparar").lower()
    metricas = paquete.riesgo.metricas
    etiqueta = OBJETIVOS.get(objetivo, OBJETIVOS["comparar"])

    if objetivo == "riesgo":
        metodo = min(metricas, key=lambda m: metricas[m].volatilidad_anual)
        v = metricas[metodo]
        criterio = "menor volatilidad realizada (OOS)"
        detalle = (f"«{nombre_visible(metodo, paquete)}» fue la más tranquila: volatilidad "
                   f"{pct(v.volatilidad_anual)} y caída máxima {pct(v.max_drawdown)} fuera de muestra.")
    elif objetivo == "objetivo":
        objetivo_ret = paquete.configuracion.retorno_objetivo_anual
        cumplen = {m: v for m, v in metricas.items() if v.retorno_anual >= objetivo_ret}
        if cumplen:
            metodo = max(cumplen, key=lambda m: cumplen[m].sharpe)
            criterio = f"alcanzó el retorno objetivo ({pct(objetivo_ret)}) con mejor Sharpe OOS"
        else:
            metodo = min(metricas, key=lambda m: abs(metricas[m].retorno_anual - objetivo_ret))
            criterio = f"el más cercano al retorno objetivo ({pct(objetivo_ret)}); ninguno lo alcanzó OOS"
        v = metricas[metodo]
        detalle = (f"«{nombre_visible(metodo, paquete)}»: retorno OOS {pct(v.retorno_anual)}, "
                   f"Sharpe {num(v.sharpe)}, caída máxima {pct(v.max_drawdown)}.")
    elif objetivo == "convexidad":
        conv = paquete.riesgo.convexidad
        metodo = conv["asimetria"].idxmax()
        fila = conv.loc[metodo]
        criterio = "mejor asimetría OOS (sube más de lo que baja)"
        detalle = (f"«{nombre_visible(metodo, paquete)}»: captura alcista {num(fila['captura_alcista'])} "
                   f"vs bajista {num(fila['captura_bajista'])} (asimetría {num(fila['asimetria'])}). "
                   "Con pesos fijos no hay convexidad garantizada, pero es el perfil más cercano a "
                   "«bajo poco al caer, subo al subir».")
    else:  # sharpe o comparar -> mejor Sharpe OOS
        metodo = max(metricas, key=lambda m: metricas[m].sharpe)
        v = metricas[metodo]
        criterio = "mejor ratio de Sharpe realizado (OOS)"
        detalle = (f"«{nombre_visible(metodo, paquete)}»: Sharpe OOS {num(v.sharpe)}, retorno "
                   f"{pct(v.retorno_anual)}, volatilidad {pct(v.volatilidad_anual)}, "
                   f"caída máxima {pct(v.max_drawdown)}.")

    return Recomendacion(objetivo=objetivo, etiqueta_objetivo=etiqueta,
                         metodo=metodo, criterio=criterio, detalle=detalle)

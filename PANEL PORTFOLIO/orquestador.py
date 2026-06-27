"""Orquestación del pipeline (compatibilidad).

El motor real vive en `motor.py`. Este módulo expone `PASOS` (incluyendo la
generación de informes) para el entry-point con barra de progreso, y delega todo
el cálculo en `motor.construir_payload` / `motor.generar_informes`.
"""

from __future__ import annotations

from CONTRATOS.modelos import Configuracion
from OPTIMIZACION.estrategias import ejecutar_optimizacion
from OPTIMIZACION.estrategias.consolidacion import (
    consolidar_por_motor,
    deduplicar_candidatos,
)
import motor


# (etiqueta, función, peso ≈ segundos) — la función recibe (ctx, log).
def _envolver(funcion):
    def paso(ctx, log):
        funcion(ctx, log)
    return paso


def _paso_optimizacion_strategy(ctx, log) -> None:
    cfg = ctx["cfg"]
    resultado = ejecutar_optimizacion(ctx["entrada"], ctx["momentos"], cfg)
    candidatos = deduplicar_candidatos(resultado.candidatos)
    ctx["frontera"] = resultado.frontera
    ctx["candidatos"] = candidatos
    ctx["candidatos_por_motor"] = consolidar_por_motor(candidatos)
    ctx["curva_top_sharpe"] = resultado.curva_top_sharpe
    ctx["motores_ejecutados"] = resultado.motores_ejecutados
    motores = ", ".join(resultado.motores_ejecutados)
    log(f"Optimización Strategy ({cfg.optimization_engine}): {motores}.")


def _inyectar_paso_strategy(etiqueta: str, funcion):
    if etiqueta == "Optimización Strategy":
        return _paso_optimizacion_strategy
    return _envolver(funcion)


PASOS = tuple(
    (etiqueta, _inyectar_paso_strategy(etiqueta, funcion), 1.0)
    for etiqueta, funcion in motor.PASOS
) + (
    ("Dashboard ejecutivo (HTML + PDF)", lambda ctx, log: ctx["rutas"].update(
        motor.generar_informes(motor.ensamblar_payload(ctx, log), log)), 8.0),
)


def construir_paquete(configuracion: Configuracion):
    """Versión no interactiva: devuelve el ReportPayload sin generar informes."""
    return motor.construir_payload(configuracion)

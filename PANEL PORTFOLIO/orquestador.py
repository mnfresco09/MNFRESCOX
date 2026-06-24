"""Orquestación del pipeline (compatibilidad).

El motor real vive en `motor.py`. Este módulo expone `PASOS` (incluyendo la
generación de informes) para el entry-point con barra de progreso, y delega todo
el cálculo en `motor.construir_payload` / `motor.generar_informes`.
"""

from __future__ import annotations

from CONTRATOS.modelos import Configuracion
import motor


# (etiqueta, función, peso ≈ segundos) — la función recibe (ctx, log).
def _envolver(funcion):
    def paso(ctx, log):
        funcion(ctx, log)
    return paso


PASOS = tuple(
    (etiqueta, _envolver(funcion), 1.0) for etiqueta, funcion in motor.PASOS
) + (
    ("Dashboard ejecutivo (HTML + PDF)", lambda ctx, log: ctx["rutas"].update(
        motor.generar_informes(motor.ensamblar_payload(ctx, log), log)), 8.0),
)


def construir_paquete(configuracion: Configuracion):
    """Versión no interactiva: devuelve el ReportPayload sin generar informes."""
    return motor.construir_payload(configuracion)

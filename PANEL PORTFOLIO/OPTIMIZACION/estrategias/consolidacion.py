"""Consolidación de candidatos generados por estrategias."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from CONTRATOS.modelos import PortfolioCandidate


def consolidar_por_motor(candidatos: tuple[PortfolioCandidate, ...]) -> dict[str, tuple[PortfolioCandidate, ...]]:
    consolidado: dict[str, list[PortfolioCandidate]] = {}
    for candidato in candidatos:
        consolidado.setdefault(candidato.motor_optimizacion, []).append(candidato)
    return {motor: tuple(items) for motor, items in consolidado.items()}


def deduplicar_candidatos(
    candidatos: tuple[PortfolioCandidate, ...],
    tolerancia: float = 1e-3,
) -> tuple[PortfolioCandidate, ...]:
    """Elimina clones de pesos dentro de cada motor y fusiona sus etiquetas."""
    salida: list[PortfolioCandidate] = []
    for candidato in candidatos:
        pesos = candidato.pesos.to_numpy(dtype=float)
        fusionado = False
        for i, existente in enumerate(salida):
            if existente.motor_optimizacion != candidato.motor_optimizacion:
                continue
            pesos_existente = existente.pesos.reindex(candidato.pesos.index).to_numpy(dtype=float)
            if np.allclose(pesos, pesos_existente, atol=tolerancia, rtol=0.0):
                salida[i] = replace(existente, nivel=f"{existente.nivel} / {candidato.nivel}")
                fusionado = True
                break
        if not fusionado:
            salida.append(candidato)
    return tuple(salida)

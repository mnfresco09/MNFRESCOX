"""Registro y ejecución de estrategias de optimización."""

from __future__ import annotations

from CONTRATOS.modelos import (
    Configuracion,
    MomentsResult,
    PortfolioInput,
    ResultadoOptimizacion,
)
from OPTIMIZACION.frontera import construir_frontera
from OPTIMIZACION.perfiles import curva_top_sharpe

from .base import OptimizadorBase
from .cvar import OptimizadorCVaR
from .media_varianza import OptimizadorMediaVarianza
from .nco import OptimizadorNCO

ORDEN_MOTORES = ("MARKOWITZ", "CVAR", "NCO")
REGISTRO: dict[str, OptimizadorBase] = {
    "MARKOWITZ": OptimizadorMediaVarianza(),
    "CVAR": OptimizadorCVaR(),
    "NCO": OptimizadorNCO(),
}


def motores_activos(cfg: Configuracion) -> tuple[str, ...]:
    if cfg.optimization_engine == "ALL":
        return ORDEN_MOTORES
    return (cfg.optimization_engine,)


def ejecutar_optimizacion(
    entrada: PortfolioInput,
    momentos: MomentsResult,
    cfg: Configuracion,
) -> ResultadoOptimizacion:
    """Ejecuta los motores configurados y devuelve una salida única."""
    frontera = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    activos = motores_activos(cfg)
    candidatos = []
    for motor in activos:
        candidatos.extend(REGISTRO[motor].optimizar(entrada, momentos, cfg, frontera=frontera))
    return ResultadoOptimizacion(
        frontera=frontera,
        candidatos=tuple(candidatos),
        curva_top_sharpe=curva_top_sharpe(frontera),
        motores_ejecutados=activos,
    )

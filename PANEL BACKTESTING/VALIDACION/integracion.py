"""Composición de la validación completa de una estrategia (Fases 2-7).

Une todas las piezas en un único flujo y produce (a) un veredicto 🟢/🟡/🔴 con
umbrales fijados a priori y (b) el dict de datos listo para el informe
institucional. Los puntos de contacto con el motor (optimizar/evaluar) y los
datos del mejor trial llegan INYECTADOS, de modo que esta capa es verificable de
forma aislada con callbacks de prueba y, en producción, recibe los callbacks
reales que construye el runner.

Orden del flujo:
  1. CPCV  → distribución de Sharpe OOS + ratio OOS/IS.
  2. WFA   → efficiency (degradación temporal).
  3. DSR   → Sharpe deflactado contra el N REAL de la investigación.
  4. MinBTL→ chequeo de cordura sobre la longitud del backtest.
  5. PBO   → (opcional) si se aporta la matriz rendimiento trial×tiempo.
  6. Bootstrap → distribución de equity final / max DD / Sharpe.
  7. Veredicto → combina todo contra los umbrales a priori.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from COMUN import estadistica as est
from COMUN import sobreajuste
from ROBUSTEZ import bootstrap as _bootstrap
from VALIDACION import orquestador, veredicto as _veredicto


@dataclass(frozen=True)
class ResultadoValidacion:
    veredicto: _veredicto.Veredicto
    cpcv: orquestador.ResultadoCPCV
    wfa: orquestador.ResultadoWFA | None
    dsr: float
    pbo: float | None
    minbtl_anios: float
    bootstrap: _bootstrap.ResultadoBootstrap | None
    datos_informe: dict = field(default_factory=dict)


def ejecutar_validacion_completa(
    *,
    n_obs: int,
    optimizar: Callable[[np.ndarray], Any],
    evaluar: Callable[[Any, np.ndarray], dict],
    metrica: str = "sharpe_ratio",
    # CPCV
    cpcv_grupos: int = 6,
    cpcv_k: int = 2,
    embargo: float = 0.01,
    duracion_trade: int = 1,
    # WFA
    wfa_activa: bool = True,
    wfa_ventanas: int = 5,
    wfa_fraccion: float = 0.15,
    wfa_anchored: bool = False,
    # DSR / MinBTL
    sharpe_hat: float,
    n_trades: int,
    n_configuraciones: int,
    varianza_sharpe_trials: float,
    asimetria: float = 0.0,
    curtosis: float = 3.0,
    sharpe_anual_objetivo: float = 1.0,
    # PBO (opcional)
    matriz_pbo: np.ndarray | None = None,
    pbo_s: int = 16,
    # Bootstrap
    retornos_mejor: np.ndarray | None = None,
    capital_inicial: float = 10_000.0,
    bootstrap_iter: int = 10_000,
    bootstrap_bloque: int = 1,
    bootstrap_compuesto: bool = False,
    bootstrap_seed: int | None = None,
    # Informe / holdout
    cabecera: dict | None = None,
    preregistro: dict | None = None,
    is_extra: dict | None = None,
    equity_valores: list | None = None,
    indice_holdout: int | None = None,
    holdout_metrica: float | None = None,
    regimen: dict | None = None,
    nula_resumen: str | None = None,
    umbrales: _veredicto.Umbrales | None = None,
) -> ResultadoValidacion:
    """Ejecuta la validación OOS completa y devuelve veredicto + datos del informe."""
    # 1) CPCV ---------------------------------------------------------------
    cpcv = orquestador.ejecutar_cpcv(
        n_obs,
        optimizar=optimizar,
        evaluar=evaluar,
        n_grupos=cpcv_grupos,
        k=cpcv_k,
        embargo=embargo,
        duracion_trade=duracion_trade,
        metrica=metrica,
    )
    dist = cpcv.distribucion_oos

    # 2) WFA ----------------------------------------------------------------
    wfa = None
    if wfa_activa:
        wfa = orquestador.ejecutar_wfa(
            n_obs,
            optimizar=optimizar,
            evaluar=evaluar,
            n_ventanas=wfa_ventanas,
            fraccion_test=wfa_fraccion,
            anchored=wfa_anchored,
            metrica=metrica,
        )

    # 3) DSR + 4) MinBTL ----------------------------------------------------
    dsr = est.deflated_sharpe_ratio(
        sharpe_hat,
        n_trades,
        n_configuraciones=n_configuraciones,
        varianza_sharpe_trials=varianza_sharpe_trials,
        asimetria=asimetria,
        curtosis=curtosis,
    )
    minbtl = est.minimum_backtest_length(n_configuraciones, sharpe_anual_objetivo)

    # 5) PBO (opcional) -----------------------------------------------------
    pbo = None
    if matriz_pbo is not None:
        pbo = sobreajuste.pbo_cscv(matriz_pbo, s=pbo_s).pbo

    # 6) Bootstrap ----------------------------------------------------------
    boot = None
    boot_p5 = None
    if retornos_mejor is not None and np.asarray(retornos_mejor).size > 0:
        boot = _bootstrap.bootstrap_trades(
            retornos_mejor,
            n_iter=bootstrap_iter,
            tam_bloque=bootstrap_bloque,
            saldo_inicial=capital_inicial,
            compuesto=bootstrap_compuesto,
            seed=bootstrap_seed,
        )
        boot_p5 = boot.p5_equity_final

    # 7) Veredicto ----------------------------------------------------------
    ver = _veredicto.evaluar_veredicto(
        dsr=dsr,
        pbo=pbo,
        ratio_oos_is=cpcv.ratio_oos_is,
        p25_sharpe_oos=dist.p25,
        mediana_sharpe_oos=dist.mediana,
        n_trades=n_trades,
        wfa_efficiency=wfa.efficiency if wfa else None,
        bootstrap_p5_equity=boot_p5,
        capital_inicial=capital_inicial if boot_p5 is not None else None,
        holdout_metrica=holdout_metrica,
        holdout_referencia=dist.mediana,
        umbrales=umbrales,
    )

    datos = _datos_informe(
        cabecera=cabecera,
        preregistro=preregistro,
        cpcv=cpcv,
        wfa=wfa,
        ver=ver,
        dsr=dsr,
        pbo=pbo,
        minbtl=minbtl,
        boot=boot,
        is_extra=is_extra,
        equity_valores=equity_valores,
        indice_holdout=indice_holdout,
        regimen=regimen,
        nula_resumen=nula_resumen,
    )

    return ResultadoValidacion(
        veredicto=ver,
        cpcv=cpcv,
        wfa=wfa,
        dsr=dsr,
        pbo=pbo,
        minbtl_anios=minbtl,
        bootstrap=boot,
        datos_informe=datos,
    )


# ---------------------------------------------------------------------------
# Ensamblado del dict del informe
# ---------------------------------------------------------------------------

def _datos_informe(
    *, cabecera, preregistro, cpcv, wfa, ver, dsr, pbo, minbtl, boot, is_extra,
    equity_valores, indice_holdout, regimen=None, nula_resumen=None,
) -> dict:
    datos: dict = {"cabecera": cabecera or {}}
    if preregistro:
        datos["preregistro"] = preregistro

    datos["oos"] = {
        "distribucion": cpcv.distribucion_oos.como_dict(),
        "valores": [float(v) for v in cpcv.valores_oos],
        "ratio_oos_is": cpcv.ratio_oos_is,
        "metrica": cpcv.metrica,
        "wfa_efficiency": wfa.efficiency if wfa else None,
        "wfa_anchored": wfa.anchored if wfa else None,
        "wfa_valores_oos": [float(v) for v in wfa.valores_oos] if wfa else None,
        "wfa_valores_is": [float(v) for v in wfa.valores_is] if wfa else None,
    }
    datos["veredicto"] = {
        "color": ver.color,
        "dsr": dsr,
        "pbo": pbo,
        "minbtl": minbtl,
        "criterios": [
            {"nombre": c.nombre, "valor": c.valor, "color": c.color, "detalle": c.detalle}
            for c in ver.criterios
        ],
    }
    robustez: dict = {}
    if boot is not None:
        robustez["bootstrap"] = {
            "iteraciones": int(boot.n_iter),
            "p5_equity_final": boot.p5_equity_final,
            "p25_equity_final": float(np.percentile(boot.equity_final, 25)),
            "mediana_equity_final": float(np.percentile(boot.equity_final, 50)),
            "p95_equity_final": float(np.percentile(boot.equity_final, 95)),
            "mediana_max_drawdown": float(np.percentile(boot.max_drawdown, 50)),
            "p95_max_drawdown": float(np.percentile(boot.max_drawdown, 95)),
            "mediana_sharpe": float(np.percentile(boot.sharpe, 50)),
        }
    if regimen:
        robustez["regimen"] = regimen
    if nula_resumen:
        robustez["nula"] = nula_resumen
    if robustez:
        datos["robustez"] = robustez
    if is_extra:
        datos["is"] = is_extra
    if equity_valores is not None:
        datos["equity"] = {"valores": list(equity_valores), "indice_holdout": indice_holdout}
    return datos

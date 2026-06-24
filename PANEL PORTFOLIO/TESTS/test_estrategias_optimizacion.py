"""Pruebas del patrón Strategy para motores de optimización institucionales."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CONTRATOS.errores import ErrorConfiguracion
from CONTRATOS.modelos import (
    PortfolioCandidate,
    PortfolioInput,
    RiskForecast,
    SimulationSummary,
)
from CONTRATOS.validacion import construir_configuracion
from RIESGO.mcr import descomponer_riesgo

ACTIVOS = ("A", "B", "C", "D")


def _entrada() -> PortfolioInput:
    rng = np.random.default_rng(11)
    fechas = pd.bdate_range("2021-01-01", periods=520)
    mercado = rng.normal(0.0002, 0.008, size=(520, 1))
    cola = rng.choice([0.0, -0.045], size=(520, 1), p=[0.96, 0.04])
    datos = {
        "A": (0.75 * mercado + rng.normal(0.00025, 0.006, size=(520, 1))).ravel(),
        "B": (0.65 * mercado + rng.normal(0.00020, 0.007, size=(520, 1))).ravel(),
        "C": (0.20 * mercado + rng.normal(0.00010, 0.004, size=(520, 1))).ravel(),
        "D": (0.90 * mercado + cola + rng.normal(0.00045, 0.010, size=(520, 1))).ravel(),
    }
    log_retornos = pd.DataFrame(datos, index=fechas)
    cierres = (1.0 + np.expm1(log_retornos)).cumprod() * 100.0
    return PortfolioInput(ACTIVOS, log_retornos, cierres, 1_000_000.0, 21)


def _cfg(engine: str = "ALL"):
    return construir_configuracion(
        tickers=ACTIVOS,
        activo_referencia="A",
        fecha_inicio="2021-01-01",
        fecha_fin="2024-12-31",
        peso_maximo=0.55,
        n_puntos_frontera=24,
        n_carteras_factibles=600,
        n_trayectorias_mc=1000,
        optimization_engine=engine,
    )


def _momentos(entrada: PortfolioInput, cfg):
    from ANALISIS.analisis import calcular_momentos

    return calcular_momentos(entrada, cfg)


def test_configuracion_valida_motor_de_optimizacion():
    cfg = _cfg("cvar")

    assert cfg.optimization_engine == "CVAR"

    with pytest.raises(ErrorConfiguracion, match="OPTIMIZATION_ENGINE"):
        _cfg("HRP")


def test_cvar_devuelve_candidate_con_contrato_comun_y_restricciones():
    from OPTIMIZACION.estrategias import OptimizadorCVaR

    entrada = _entrada()
    cfg = _cfg("CVAR")
    momentos = _momentos(entrada, cfg)

    candidatos = OptimizadorCVaR().optimizar(entrada, momentos, cfg)

    assert len(candidatos) == 1
    c = candidatos[0]
    pesos = c.pesos.to_numpy(dtype=float)
    assert c.motor_optimizacion == "CVAR"
    assert c.nivel == "cvar"
    assert abs(float(c.pesos.sum()) - 1.0) < 1e-6
    assert (pesos >= -1e-8).all()
    assert (pesos <= cfg.restricciones.peso_maximo + 1e-8).all()
    assert np.isfinite(c.retorno_esperado)
    assert c.retorno_geometrico == pytest.approx(c.retorno_esperado)
    assert np.isfinite(c.volatilidad_estructural)
    assert c.r2_curva_capital is not None
    assert 0.0 <= c.r2_curva_capital <= 1.0
    assert c.k_ratio is not None


def test_selector_all_ejecuta_los_tres_motores():
    from OPTIMIZACION.estrategias import ejecutar_optimizacion

    entrada = _entrada()
    cfg = _cfg("ALL")
    momentos = _momentos(entrada, cfg)

    resultado = ejecutar_optimizacion(entrada, momentos, cfg)

    assert set(resultado.motores_ejecutados) == {"MARKOWITZ", "CVAR", "NCO"}
    assert {c.motor_optimizacion for c in resultado.candidatos} == {"MARKOWITZ", "CVAR", "NCO"}
    assert not resultado.frontera.puntos.empty
    assert not resultado.curva_top_sharpe.empty


def _forecast(var99: float, cvar99: float) -> RiskForecast:
    return RiskForecast(
        horizonte_dias=21,
        volatilidad_tactica_diaria=0.01,
        var_hist_95=var99 * 0.7,
        var_hist_99=var99,
        cvar_hist_95=cvar99 * 0.7,
        cvar_hist_99=cvar99,
        var_param_95=var99 * 0.7,
        var_param_99=var99,
        cvar_param_95=cvar99 * 0.7,
        cvar_param_99=cvar99,
        var_fhs_95=var99 * 0.7,
        var_fhs_99=var99,
        cvar_fhs_95=cvar99 * 0.7,
        cvar_fhs_99=cvar99,
        fuente_fhs="python_fallback",
    )


def _simulacion(cdar: float) -> SimulationSummary:
    sendas = pd.DataFrame({"p5": [0.98], "p25": [0.99], "p50": [1.0], "p75": [1.01], "p95": [1.02]})
    return SimulationSummary(
        horizonte_dias=21,
        percentiles=(5, 25, 50, 75, 95),
        sendas_percentil=sendas,
        prob_perdida=0.45,
        cdar_30d=cdar,
        retorno_mediano=0.0,
        perdida_p5=-0.02,
        fuente="python_fallback",
    )


def _candidate(nombre: str, sharpe: float, cvar99: float, cdar: float) -> PortfolioCandidate:
    activos = ("A", "B")
    pesos = pd.Series([0.5, 0.5], index=activos)
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.05]], index=activos, columns=activos)
    return PortfolioCandidate(
        nivel=nombre.lower(),
        motor_optimizacion=nombre,
        pesos=pesos,
        retorno_esperado=0.08,
        retorno_geometrico=0.08,
        volatilidad_estructural=0.12,
        volatilidad_tactica=0.12,
        sharpe=sharpe,
        descomposicion=descomponer_riesgo(pesos, cov),
        forecast=_forecast(var99=cvar99 * 0.7, cvar99=cvar99),
        simulacion=_simulacion(cdar=cdar),
        r2_curva_capital=0.95,
        k_ratio=1.0,
    )


def test_super_score_castiga_cvar_y_cdar_por_encima_de_sharpe():
    from RIESGO.score import calcular_score_cartera

    cfg = _cfg("ALL")
    agresiva = _candidate("AGRESIVA", sharpe=2.8, cvar99=-0.20, cdar=-0.18)
    defensiva = _candidate("DEFENSIVA", sharpe=0.6, cvar99=-0.035, cdar=-0.03)

    puntuadas = calcular_score_cartera((agresiva, defensiva), cfg)
    por_motor = {c.motor_optimizacion: c for c in puntuadas}

    assert por_motor["DEFENSIVA"].score > por_motor["AGRESIVA"].score
    assert "cvar" in {nombre for nombre, _valor in por_motor["DEFENSIVA"].detalle_score}

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from CONTRATOS.modelos import (
    DescomposicionRiesgo,
    PortfolioCandidate,
    RiskForecast,
    SimulationSummary,
)
from CONTRATOS.validacion import construir_configuracion


ACTIVOS = ("A", "B", "C")


def _cfg():
    return construir_configuracion(
        tickers=ACTIVOS,
        activo_referencia="A",
        fecha_inicio="2020-01-01",
        fecha_fin="2024-12-31",
        peso_maximo=1.0,
        peso_minimo=0.0,
    )


def _forecast(cvar99: float) -> RiskForecast:
    return RiskForecast(
        horizonte_dias=21,
        volatilidad_tactica_diaria=0.01,
        var_hist_95=-0.02,
        var_hist_99=-0.03,
        cvar_hist_95=-0.025,
        cvar_hist_99=cvar99,
        var_param_95=-0.02,
        var_param_99=-0.03,
        cvar_param_95=-0.025,
        cvar_param_99=cvar99,
        var_fhs_95=-0.02,
        var_fhs_99=-0.03,
        cvar_fhs_95=-0.025,
        cvar_fhs_99=cvar99,
        fuente_fhs="test",
    )


def _simulacion(cdar: float) -> SimulationSummary:
    sendas = pd.DataFrame({"p5": [0.98], "p25": [0.99], "p50": [1.0], "p75": [1.01], "p95": [1.02]})
    return SimulationSummary(
        horizonte_dias=21,
        percentiles=(5, 25, 50, 75, 95),
        sendas_percentil=sendas,
        prob_perdida=0.4,
        cdar_30d=cdar,
        retorno_mediano=0.01,
        perdida_p5=-0.03,
        fuente="test",
    )


def _descomposicion(pesos: pd.Series, riesgo_pct: tuple[float, ...]) -> DescomposicionRiesgo:
    rc = pd.Series(riesgo_pct, index=pesos.index, dtype=float)
    return DescomposicionRiesgo(
        mcr=rc,
        contribucion=rc,
        contribucion_pct=rc,
        concentracion_hhi=float(np.square(pesos.to_numpy(dtype=float)).sum()),
    )


def _candidato(
    nombre: str,
    pesos: tuple[float, ...],
    retorno: float,
    cvar99: float,
    cdar: float,
    riesgo_pct: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3),
) -> PortfolioCandidate:
    w = pd.Series(pesos, index=ACTIVOS, dtype=float)
    return PortfolioCandidate(
        nivel=nombre,
        pesos=w,
        retorno_esperado=retorno,
        volatilidad_estructural=0.10,
        volatilidad_tactica=0.10,
        sharpe=1.0,
        descomposicion=_descomposicion(w, riesgo_pct),
        motor_optimizacion="TEST",
        forecast=_forecast(cvar99),
        simulacion=_simulacion(cdar),
    )


def _score_por_nivel(candidatos):
    return {c.nivel: c.score for c in candidatos}


def _log_retornos_score() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    simples = pd.DataFrame(
        {
            "A": [0.012, 0.010, 0.011, 0.013, 0.012, 0.010, 0.011, 0.012],
            "B": [0.008, -0.002, 0.009, -0.003, 0.008, -0.002, 0.009, -0.003],
            "C": [-0.030, 0.004, -0.035, 0.005, -0.040, 0.004, -0.030, 0.005],
        },
        index=idx,
    )
    return np.log1p(simples)


def test_zscore_robusto_devuelve_ceros_si_todos_iguales():
    from RIESGO.score import _z_robusto

    z = _z_robusto(np.array([1.0, 1.0, np.nan, np.inf]))

    assert np.allclose(z, np.zeros(4))


def test_menor_cvar_gana_con_resto_constante():
    from RIESGO.score import calcular_score_compuesto

    cfg = _cfg()
    candidatos = (
        _candidato("cola_alta", (0.34, 0.33, 0.33), 0.10, -0.12, -0.05),
        _candidato("cola_baja", (0.34, 0.33, 0.33), 0.10, -0.04, -0.05),
    )

    scores = _score_por_nivel(calcular_score_compuesto(candidatos, cfg))

    assert scores["cola_baja"] > scores["cola_alta"]


def test_mayor_sharpe_gana_con_resto_constante():
    from RIESGO.score import calcular_score_compuesto

    cfg = _cfg()
    bajo_sharpe = _candidato("bajo_sharpe", (0.34, 0.33, 0.33), 0.10, -0.05, -0.05)
    alto_sharpe = _candidato("alto_sharpe", (0.34, 0.33, 0.33), 0.10, -0.05, -0.05)
    bajo_sharpe = replace(bajo_sharpe, sharpe=0.25)
    alto_sharpe = replace(alto_sharpe, sharpe=1.25)

    puntuados = calcular_score_compuesto((bajo_sharpe, alto_sharpe), cfg)
    por_nivel = {c.nivel: c for c in puntuados}
    detalle_alto = dict(por_nivel["alto_sharpe"].detalle_score)

    assert por_nivel["alto_sharpe"].score > por_nivel["bajo_sharpe"].score
    assert detalle_alto["z_sharpe"] > 0.0
    assert detalle_alto["comp_sharpe"] > 0.0


def test_constantes_del_score_institucional_actual():
    import RIESGO.score as score

    assert score.SCORE_CONFIG == {
        "peso_sharpe": 2.50,
        "peso_sortino": 0.65,
        "peso_retorno_exceso": 0.35,
        "peso_k_ratio": 0.25,
        "peso_calmar": 0.20,
        "peso_cvar": 0.90,
        "peso_cdar": 1.10,
        "peso_max_drawdown": 0.55,
        "peso_hhi_pesos": 0.05,
        "peso_hhi_riesgo": 0.60,
        "peso_correlacion": 0.30,
        "peso_max_contrib_riesgo": 0.75,
        "umbral_max_contrib_riesgo": 0.50,
    }
    assert score.REGLAS_DURAS == {
        "max_drawdown_historico_max": 0.30,
        "cvar_99_max": 0.065,
        "max_contrib_riesgo_max": 0.525,
        "volatilidad_max": 0.40,
    }
    assert score.PESO_SHARPE == 2.50
    assert score.PESO_SORTINO == 0.65
    assert score.PESO_RETORNO_EXCESO == 0.35
    assert score.PESO_K_RATIO == 0.25
    assert score.PESO_CALMAR == 0.20
    assert score.PESO_CVAR == 0.90
    assert score.PESO_CDAR == 1.10
    assert score.PESO_MAX_DRAWDOWN == 0.55
    assert score.PESO_HHI_PESOS == 0.05
    assert score.PESO_HHI_RIESGO == 0.60
    assert score.PESO_CORRELACION == 0.30
    assert score.PESO_MAX_CONTRIB_RIESGO == 0.75


def test_score_avanzado_usa_sortino_k_ratio_calmar_y_max_drawdown():
    from RIESGO.score import calcular_score_compuesto

    cfg = _cfg()
    estable = replace(
        _candidato("estable", (0.80, 0.20, 0.00), 0.12, -0.04, -0.03),
        sharpe=1.10,
        k_ratio=1.50,
    )
    inestable = replace(
        _candidato("inestable", (0.00, 0.20, 0.80), 0.12, -0.04, -0.03),
        sharpe=1.10,
        k_ratio=0.20,
    )

    puntuados = calcular_score_compuesto(
        (estable, inestable),
        cfg,
        log_retornos=_log_retornos_score(),
    )
    por_nivel = {c.nivel: c for c in puntuados}
    detalle_estable = dict(por_nivel["estable"].detalle_score)
    detalle_inestable = dict(por_nivel["inestable"].detalle_score)

    assert por_nivel["estable"].score > por_nivel["inestable"].score
    assert detalle_estable["sortino"] > detalle_inestable["sortino"]
    assert detalle_estable["k_ratio"] > detalle_inestable["k_ratio"]
    assert detalle_estable["calmar"] > detalle_inestable["calmar"]
    assert detalle_estable["max_drawdown_abs"] < detalle_inestable["max_drawdown_abs"]
    assert {
        "z_sortino",
        "z_k_ratio",
        "z_calmar",
        "z_max_drawdown",
        "comp_sortino",
        "comp_k_ratio",
        "comp_calmar",
        "comp_max_drawdown",
    } <= set(detalle_estable)


def test_reglas_duras_excluyen_candidato_que_supera_limites_institucionales():
    from RIESGO.score import PENALIZACION_REGLA_DURA, calcular_score_compuesto

    cfg = _cfg()
    infractora = replace(
        _candidato("infractora", (0.34, 0.33, 0.33), 0.40, -0.08, -0.04, (0.53, 0.24, 0.23)),
        sharpe=5.0,
        volatilidad_tactica=0.41,
    )
    valida = replace(
        _candidato("valida", (0.34, 0.33, 0.33), 0.08, -0.04, -0.03, (0.34, 0.33, 0.33)),
        sharpe=0.5,
        volatilidad_tactica=0.10,
    )

    puntuados = calcular_score_compuesto((infractora, valida), cfg)
    por_nivel = {c.nivel: c for c in puntuados}
    detalle = dict(por_nivel["infractora"].detalle_score)

    assert por_nivel["valida"].score > por_nivel["infractora"].score
    assert detalle["regla_cvar_99_incumplida"] == 1.0
    assert detalle["regla_max_contrib_riesgo_incumplida"] == 1.0
    assert detalle["regla_volatilidad_incumplida"] == 1.0
    assert detalle["penalizacion_reglas_duras"] >= 3 * PENALIZACION_REGLA_DURA


def test_menor_concentracion_de_riesgo_gana_con_resto_constante():
    from RIESGO.score import calcular_score_compuesto

    cfg = _cfg()
    candidatos = (
        _candidato("riesgo_concentrado", (0.34, 0.33, 0.33), 0.10, -0.05, -0.05, (0.90, 0.05, 0.05)),
        _candidato("riesgo_equilibrado", (0.34, 0.33, 0.33), 0.10, -0.05, -0.05, (0.34, 0.33, 0.33)),
    )

    scores = _score_por_nivel(calcular_score_compuesto(candidatos, cfg))

    assert scores["riesgo_equilibrado"] > scores["riesgo_concentrado"]


def test_correlacion_ponderada_positiva_penaliza_activos_redundantes():
    from RIESGO.score import calcular_score_compuesto

    cfg = _cfg()
    correlacion = pd.DataFrame(
        [[1.0, 0.90, -0.30], [0.90, 1.0, 0.10], [-0.30, 0.10, 1.0]],
        index=ACTIVOS,
        columns=ACTIVOS,
    )
    candidatos = (
        _candidato("redundante", (0.50, 0.50, 0.00), 0.10, -0.05, -0.05, (0.50, 0.50, 0.00)),
        _candidato("diversificada", (0.50, 0.00, 0.50), 0.10, -0.05, -0.05, (0.50, 0.00, 0.50)),
    )

    scores = _score_por_nivel(calcular_score_compuesto(candidatos, cfg, correlacion=correlacion))

    assert scores["diversificada"] > scores["redundante"]


def test_mayor_retorno_con_riesgo_extremo_no_gana_automaticamente():
    from RIESGO.score import calcular_score_compuesto

    cfg = _cfg()
    candidatos = (
        _candidato("retorno_extremo", (0.34, 0.33, 0.33), 0.28, -0.45, -0.40, (0.90, 0.05, 0.05)),
        _candidato("robusta", (0.34, 0.33, 0.33), 0.10, -0.04, -0.04, (0.34, 0.33, 0.33)),
    )

    scores = _score_por_nivel(calcular_score_compuesto(candidatos, cfg))

    assert scores["robusta"] > scores["retorno_extremo"]


def test_max_contribucion_riesgo_superior_50_recibe_penalizacion_dura():
    from RIESGO.score import calcular_score_compuesto

    cfg = _cfg()
    candidatos = (
        _candidato("riesgo_51", (0.34, 0.33, 0.33), 0.10, -0.05, -0.05, (0.51, 0.245, 0.245)),
        _candidato("riesgo_50", (0.34, 0.33, 0.33), 0.10, -0.05, -0.05, (0.50, 0.25, 0.25)),
    )

    puntuados = calcular_score_compuesto(candidatos, cfg)
    por_nivel = {c.nivel: c for c in puntuados}
    detalle_51 = dict(por_nivel["riesgo_51"].detalle_score)
    detalle_50 = dict(por_nivel["riesgo_50"].detalle_score)

    assert por_nivel["riesgo_50"].score > por_nivel["riesgo_51"].score
    assert detalle_51["max_contribucion_riesgo"] > 0.50
    assert detalle_51["penalizacion_max_contribucion_riesgo"] > 0.0
    assert detalle_51["z_penalizacion_max_contrib_riesgo"] > 0.0
    assert detalle_51["comp_max_contrib_riesgo"] < 0.0
    assert detalle_50["penalizacion_max_contribucion_riesgo"] == 0.0

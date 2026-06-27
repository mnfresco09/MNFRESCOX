"""Tests cuantitativos del Motor de Riesgo Predictivo.

Cubren: log returns, doble lente (Ledoit-Wolf + EWMA), shrinkage de retorno,
frontera restringida (suma=1 y límites min/max), selección de los 4 perfiles,
MCR, score, integración Python↔Rust (binding/fallback) y validación de
fallbacks (matriz singular, Black-Litterman sin views).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from CONTRATOS.modelos import PortfolioInput, Restricciones
from CONTRATOS.validacion import construir_configuracion


# --------------------------------------------------------------------------- #
#  Fixtures: datos sintéticos reproducibles                                    #
# --------------------------------------------------------------------------- #
ACTIVOS = ["A", "B", "C", "D"]


@pytest.fixture
def log_retornos() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    fechas = pd.bdate_range("2020-01-01", periods=600)
    # Cuatro activos correlacionados con distinta vol y deriva.
    base = rng.normal(0, 0.01, size=(600, 1))
    cols = {}
    for i, a in enumerate(ACTIVOS):
        idio = rng.normal(0.0002 * (i + 1), 0.008 * (i + 1), size=(600, 1))
        cols[a] = (0.6 * base + idio).ravel()
    return pd.DataFrame(cols, index=fechas)


@pytest.fixture
def cfg():
    return construir_configuracion(
        tickers=ACTIVOS, activo_referencia="A",
        fecha_inicio="2020-01-01", fecha_fin="2024-12-31",
        peso_maximo=0.5, peso_minimo=0.0,
        n_trayectorias_mc=4000,   # simulación ligera para tests rápidos
    )


@pytest.fixture
def entrada(log_retornos):
    cierres = (1 + np.expm1(log_retornos)).cumprod() * 100
    return PortfolioInput(tuple(ACTIVOS), log_retornos, cierres, 1_000_000.0, 21)


@pytest.fixture
def momentos(entrada, cfg):
    from ANALISIS.analisis import calcular_momentos
    return calcular_momentos(entrada, cfg)


# --------------------------------------------------------------------------- #
#  Estadística y doble lente                                                   #
# --------------------------------------------------------------------------- #
def test_log_returns_signo_y_finitud(log_retornos):
    assert np.isfinite(log_retornos.to_numpy()).all()
    # log(P_t/P_{t-1}); media cercana a cero, no degenerada.
    assert (log_retornos.std() > 0).all()


def test_ledoit_wolf_psd_simetrica(momentos):
    cov = momentos.cov_estructural.to_numpy()
    assert np.allclose(cov, cov.T, atol=1e-10)
    assert np.linalg.eigvalsh(cov).min() > 0           # definida positiva
    assert 0.0 <= momentos.shrinkage_cov <= 1.0


def test_ewma_psd_y_distinta_de_estructural(momentos):
    cov_t = momentos.cov_tactica.to_numpy()
    assert np.allclose(cov_t, cov_t.T, atol=1e-10)
    assert np.linalg.eigvalsh(cov_t).min() > 0
    # La táctica no debe ser idéntica a la estructural.
    assert not np.allclose(cov_t, momentos.cov_estructural.to_numpy())


def test_shrinkage_retorno_entre_media_y_gran_media(momentos):
    mu = momentos.retornos_medios
    mu_aj = momentos.retornos_ajustados
    gran = float(mu.mean())
    # Cada μ ajustado queda entre la media cruda y la gran media (encogimiento).
    for a in mu.index:
        lo, hi = sorted((mu[a], gran))
        assert lo - 1e-9 <= mu_aj[a] <= hi + 1e-9


# --------------------------------------------------------------------------- #
#  Frontera y restricciones                                                    #
# --------------------------------------------------------------------------- #
def test_frontera_pesos_validos(momentos, cfg):
    from OPTIMIZACION.frontera import construir_frontera
    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    cols = [f"peso·{a}" for a in ACTIVOS]
    for _, fila in fr.puntos.iterrows():
        w = fila[cols].to_numpy(dtype=float)
        assert abs(w.sum() - 1.0) < 1e-4                # suma 1
        assert (w >= -1e-6).all()                       # long-only
        assert (w <= cfg.restricciones.peso_maximo + 1e-4).all()  # tope


def test_frontera_se_despliega(momentos, cfg):
    from OPTIMIZACION.frontera import construir_frontera
    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    # Con 4 activos diferenciados la frontera tiene rango de volatilidad > 0.
    assert fr.puntos["volatilidad"].max() > fr.puntos["volatilidad"].min()


def test_frontera_alcanza_maximo_retorno_factible(momentos, cfg):
    from OPTIMIZACION import optimizador as opt
    from OPTIMIZACION.frontera import construir_frontera

    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    activos = list(momentos.cov_estructural.index)
    _, retorno_max = opt.rango_retorno_factible(
        momentos.retornos_ajustados.reindex(activos),
        cfg.restricciones,
        len(activos),
    )

    assert fr.puntos["retorno"].max() == pytest.approx(retorno_max, abs=2e-4)


def test_frontera_incluye_minima_varianza_aunque_mu_domine_varianza(cfg):
    from OPTIMIZACION.frontera import construir_frontera
    from OPTIMIZACION.optimizador import minima_varianza

    activos = list(ACTIVOS)
    mu = pd.Series([0.02, 0.08, 0.18, 0.32], index=activos)
    cov = pd.DataFrame(
        np.diag([1e-8, 2e-8, 3e-8, 4e-8]),
        index=activos,
        columns=activos,
    )

    fr = construir_frontera(mu, cov, cfg)
    w_min = minima_varianza(cov, cfg.restricciones)
    vol_min = float(np.sqrt(w_min.to_numpy() @ cov.to_numpy() @ w_min.to_numpy()))

    assert fr.puntos["volatilidad"].min() == pytest.approx(vol_min, abs=1e-7)


def test_nube_factible_conserva_pesos_para_hover(momentos, cfg):
    from OPTIMIZACION.frontera import construir_frontera
    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    cols = [f"peso·{a}" for a in ACTIVOS]

    assert set(cols) <= set(fr.nube_factible.columns)
    pesos = fr.nube_factible[cols].to_numpy(dtype=float)
    assert np.allclose(pesos.sum(axis=1), 1.0)
    assert (pesos >= -1e-9).all()


def test_figuras_html_muestran_pesos_en_todos_los_puntos(momentos, entrada, cfg):
    from OPTIMIZACION.frontera import construir_frontera
    from OPTIMIZACION.perfiles import seleccionar_perfiles
    from RIESGO.exploracion_riesgo import construir_exploracion
    from REPORTES import graficos_interactivos
    from REPORTES.narrativa import texto_frontera_referencia

    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    perfiles = seleccionar_perfiles(fr, momentos, cfg)
    expl = construir_exploracion(fr, momentos, entrada, cfg, perfiles)
    payload = SimpleNamespace(
        frontera=fr,
        candidatos=perfiles,
        configuracion=cfg,
        clasificacion_nube=expl["clasificacion_nube"],
        clasificacion_frontera=expl["clasificacion_frontera"],
        anclas=expl["anclas"],
    )

    fig_frontera = graficos_interactivos.frontera(payload)
    nube_trace = fig_frontera.data[0]
    assert nube_trace.customdata is not None
    assert "Pesos" in nube_trace.hovertemplate
    assert "Frontera MV" in str(fig_frontera.data[1].name)
    assert "CVaR/NCO" in texto_frontera_referencia()

    fig_clasificada = graficos_interactivos.frontera_clasificada(payload)
    trazas_nube = [t for t in fig_clasificada.data if str(t.name).startswith("Universo")]
    assert trazas_nube
    assert all(t.customdata is not None for t in trazas_nube)
    assert all("Pesos" in t.hovertemplate for t in trazas_nube)


def test_html_acepta_figuras_png_del_pdf_sin_tuple_to_html(momentos, entrada, cfg, tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import plotly.graph_objects as go

    from OPTIMIZACION.frontera import construir_frontera
    from OPTIMIZACION.perfiles import seleccionar_perfiles
    from REPORTES.dashboard_html import generar_html
    from RIESGO.forecast import calcular_forecast, calcular_simulacion

    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    candidato = seleccionar_perfiles(fr, momentos, cfg)[0]
    candidato = replace(
        candidato,
        motor_optimizacion="MARKOWITZ",
        forecast=calcular_forecast(candidato.pesos, entrada.log_retornos, momentos, cfg),
        simulacion=calcular_simulacion(candidato.pesos, entrada.log_retornos, cfg),
        score=1.0,
    )
    payload = SimpleNamespace(
        configuracion=cfg,
        entrada=entrada,
        momentos=momentos,
        frontera=fr,
        candidatos=(candidato,),
        recomendada=candidato,
        regimen=SimpleNamespace(
            etiqueta="baja_volatilidad",
            percentil_volatilidad=0.2,
            volatilidad_actual=0.1,
            correlacion_media_actual=0.3,
            descripcion="Régimen sintético.",
        ),
        recomendacion=SimpleNamespace(detalle="Test.", criterio="Score."),
        frontera_degenerada=False,
        nota_frontera="",
        clasificacion_frontera=fr.puntos,
        clasificacion_nube=fr.nube_factible,
        anclas=(),
        leaderboard=(),
    )
    figuras_plotly = {
        clave: go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
        for clave in ("pesos", "mcr", "fan_chart", "var_forecast", "frontera", "correlacion")
    }
    monkeypatch.setattr(
        "REPORTES.dashboard_html.graficos_interactivos.generar_figuras",
        lambda _payload: figuras_plotly,
    )

    figuras_png_pdf = {
        clave: (tmp_path / f"{clave}.png", "data:image/png;base64,AA==")
        for clave in figuras_plotly
    }
    ruta = generar_html(payload, tmp_path / "informe.html", figuras=figuras_png_pdf)

    assert ruta.exists()
    assert "Informe de cartera" in ruta.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Perfiles y MCR                                                              #
# --------------------------------------------------------------------------- #
def test_seleccion_cuatro_perfiles(momentos, cfg):
    from OPTIMIZACION.frontera import construir_frontera
    from OPTIMIZACION.perfiles import seleccionar_perfiles
    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    cands = seleccionar_perfiles(fr, momentos, cfg)
    niveles = {c.nivel for c in cands}
    assert niveles == {"bajo", "medio", "alto", "max_sharpe"}
    for c in cands:
        assert abs(float(c.pesos.sum()) - 1.0) < 1e-4


def test_mcr_suma_volatilidad(momentos):
    from RIESGO.mcr import descomponer_riesgo
    pesos = pd.Series([0.25, 0.25, 0.25, 0.25], index=ACTIVOS)
    d = descomponer_riesgo(pesos, momentos.cov_tactica)
    vol = float(np.sqrt(pesos.to_numpy() @ momentos.cov_tactica.to_numpy() @ pesos.to_numpy()))
    # Σ contribuciones = volatilidad de la cartera (Euler).
    assert abs(float(d.contribucion.sum()) - vol) < 1e-8
    assert abs(float(d.contribucion_pct.sum()) - 1.0) < 1e-6
    assert 0.0 < d.concentracion_hhi <= 1.0


# --------------------------------------------------------------------------- #
#  Forecast, simulación, score (integración Python↔Rust/fallback)             #
# --------------------------------------------------------------------------- #
def test_forecast_signos_y_orden(momentos, entrada, cfg):
    from RIESGO.forecast import calcular_forecast
    pesos = pd.Series([0.4, 0.3, 0.2, 0.1], index=ACTIVOS)
    f = calcular_forecast(pesos, entrada.log_retornos, momentos, cfg)
    # VaR/CVaR negativos y CVaR no menos severo que VaR (más en la cola).
    assert f.var_fhs_99 < 0 and f.cvar_fhs_99 <= f.var_fhs_99 + 1e-9
    assert f.var_param_99 <= f.var_param_95 + 1e-9      # 99% más severo que 95%
    assert f.fuente_fhs in ("rust", "python_fallback")


def test_simulacion_fan_y_cdar(momentos, entrada, cfg):
    from RIESGO.forecast import calcular_simulacion
    pesos = pd.Series([0.4, 0.3, 0.2, 0.1], index=ACTIVOS)
    s = calcular_simulacion(pesos, entrada.log_retornos, cfg)
    assert s.sendas_percentil.shape == (cfg.horizonte_dias, len(cfg.percentiles_fan))
    assert 0.0 <= s.prob_perdida <= 1.0
    assert s.cdar_30d <= 0.0                            # drawdown negativo
    assert s.fuente in ("rust", "python_fallback")


def test_score_asignado_y_finito(momentos, entrada, cfg):
    from dataclasses import replace
    from OPTIMIZACION.frontera import construir_frontera
    from OPTIMIZACION.perfiles import seleccionar_perfiles
    from RIESGO.forecast import calcular_forecast, calcular_simulacion
    from RIESGO.score import calcular_score_cartera
    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    cands = seleccionar_perfiles(fr, momentos, cfg)
    cands = tuple(
        replace(c,
                forecast=calcular_forecast(c.pesos, entrada.log_retornos, momentos, cfg),
                simulacion=calcular_simulacion(c.pesos, entrada.log_retornos, cfg))
        for c in cands
    )
    puntuados = calcular_score_cartera(cands, cfg)
    assert all(np.isfinite(c.score) for c in puntuados)
    assert max(puntuados, key=lambda c: c.score).score is not None


def test_enriquecer_candidatos_riesgo_conserva_orden(momentos, entrada, cfg, monkeypatch):
    from OPTIMIZACION.frontera import construir_frontera
    from OPTIMIZACION.perfiles import seleccionar_perfiles
    from RIESGO import forecast

    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    candidatos = seleccionar_perfiles(fr, momentos, cfg)

    def fake_forecast(pesos, *_args):
        return f"forecast-{float(pesos.iloc[0]):.6f}"

    def fake_simulacion(pesos, *_args):
        return f"sim-{float(pesos.iloc[0]):.6f}"

    monkeypatch.setattr(forecast, "calcular_forecast", fake_forecast)
    monkeypatch.setattr(forecast, "calcular_simulacion", fake_simulacion)

    enriquecidos = forecast.enriquecer_candidatos_riesgo(
        candidatos,
        entrada.log_retornos,
        momentos,
        cfg,
        max_workers=4,
    )

    assert [c.nivel for c in enriquecidos] == [c.nivel for c in candidatos]
    assert all(str(c.forecast).startswith("forecast-") for c in enriquecidos)
    assert all(str(c.simulacion).startswith("sim-") for c in enriquecidos)


def test_binding_montecarlo_no_devuelve_trayectorias():
    from RIESGO import motor_bindings
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, 300)
    resumen, fuente = motor_bindings.montecarlo(ret, 21, 5000, (5, 25, 50, 75, 95), 42)
    assert resumen["sendas"].shape == (5, 21)          # solo percentiles, no N×H
    assert fuente in ("rust", "python_fallback")


# --------------------------------------------------------------------------- #
#  Validación de fallbacks                                                     #
# --------------------------------------------------------------------------- #
def test_ledoit_wolf_regulariza_matriz_singular():
    from ANALISIS.momentos import covarianza_ledoit_wolf
    # Dos activos perfectamente colineales → muestral singular.
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 300)
    df = pd.DataFrame({"A": x, "B": x, "C": x * 0.5 + rng.normal(0, 1e-9, 300)})
    cov, shrink = covarianza_ledoit_wolf(df, 252)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0   # Ledoit-Wolf la hace PSD
    assert shrink > 0


def test_black_litterman_sin_views_cae_a_shrinkage(momentos, entrada, cfg):
    from ANALISIS.retorno_esperado import estimar_retorno_esperado
    assert cfg.views_black_litterman == ()
    _, fuente = estimar_retorno_esperado(entrada.log_retornos, momentos.cov_estructural, cfg)
    assert fuente == "shrinkage"


def test_clasificacion_y_leaderboard(momentos, entrada, cfg):
    from OPTIMIZACION.frontera import construir_frontera
    from OPTIMIZACION.perfiles import seleccionar_perfiles
    from RIESGO.exploracion_riesgo import construir_exploracion
    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    perf = seleccionar_perfiles(fr, momentos, cfg)
    expl = construir_exploracion(fr, momentos, entrada, cfg, perf)

    # Clasificación: clases válidas y bandas ordenadas.
    clases = set(expl["clasificacion_frontera"]["clase"].unique())
    assert clases <= {"bajo", "medio", "alto"}
    b1, b2 = expl["bandas"]
    assert b1 < b2
    assert len(expl["anclas"]) == 3

    # Leaderboard: 6 criterios, cada Top-5 con riesgo PRECISO y score.
    lb = expl["leaderboard"]
    assert len(lb) == 6
    for cr in lb:
        assert 1 <= len(cr.top) <= 5
        for c in cr.top:
            assert c.forecast is not None and c.simulacion is not None
            assert c.score is not None and np.isfinite(c.score)
            assert c.diversificacion is not None and c.diversificacion >= 1.0 - 1e-6
            assert c.clase_riesgo in {"bajo", "medio", "alto"}


def test_leaderboard_criterios_ordenados(momentos, entrada, cfg):
    """Cada criterio ordena su Top por la métrica precisa correspondiente."""
    from OPTIMIZACION.frontera import construir_frontera
    from OPTIMIZACION.perfiles import seleccionar_perfiles
    from RIESGO.exploracion_riesgo import construir_exploracion
    fr = construir_frontera(momentos.retornos_ajustados, momentos.cov_estructural, cfg)
    perf = seleccionar_perfiles(fr, momentos, cfg)
    lb = {cr.clave: cr for cr in construir_exploracion(fr, momentos, entrada, cfg, perf)["leaderboard"]}
    # Sharpe descendente.
    sh = [c.sharpe for c in lb["sharpe"].top]
    assert sh == sorted(sh, reverse=True)
    # Mín VaR99: el primero es el menos negativo (mejor) de su Top.
    vars99 = [c.forecast.var_fhs_99 for c in lb["var99"].top]
    assert vars99[0] == max(vars99)


def test_optimizador_respeta_peso_minimo():
    from ANALISIS.momentos import covarianza_ledoit_wolf
    from OPTIMIZACION.optimizador import minima_varianza
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 0.01, (300, 3)), columns=["A", "B", "C"])
    cov, _ = covarianza_ledoit_wolf(df, 252)
    restr = Restricciones(solo_largos=True, peso_maximo=0.6, peso_minimo=0.1)
    w = minima_varianza(cov, restr)
    assert (w.to_numpy() >= 0.1 - 1e-4).all()
    assert (w.to_numpy() <= 0.6 + 1e-4).all()
    assert abs(float(w.sum()) - 1.0) < 1e-4

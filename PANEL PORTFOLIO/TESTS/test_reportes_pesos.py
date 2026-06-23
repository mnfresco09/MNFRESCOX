"""Pruebas de reportes centrados en pesos por perfil de riesgo."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from CONTRATOS.modelos import MetricasCartera
from REPORTES.graficos_mpl import generar_pngs
from REPORTES.graficos_plotly import todas_las_figuras
from REPORTES.html import generar_html


def _metricas(
    retorno: float = 0.08,
    volatilidad: float = 0.12,
    sharpe: float = 0.65,
) -> MetricasCartera:
    return MetricasCartera(
        retorno_anual=retorno,
        volatilidad_anual=volatilidad,
        sharpe=sharpe,
        sortino=0.80,
        calmar=0.40,
        max_drawdown=-0.14,
        duracion_drawdown_dias=35,
        fecha_recuperacion=None,
        var=-0.018,
        cvar=-0.026,
    )


def _asignacion(nombre: str, pesos: tuple[float, float, float], sharpe: float):
    activos = ("AAA", "BBB", "CCC")
    return SimpleNamespace(
        nombre=nombre,
        pesos=pd.Series(pesos, index=activos),
        metricas=SimpleNamespace(
            retorno_anual=0.08 + sharpe / 100,
            volatilidad_anual=0.10 + sharpe / 100,
            sharpe=sharpe,
        ),
        estado_solver="test",
        diagnostico="test",
        advertencias=(),
    )


def paquete_minimo():
    activos = ("AAA", "BBB", "CCC")
    fechas = pd.date_range("2024-01-01", periods=8, freq="D")
    metodos = (
        "Equiponderada (1/N)",
        "Mínima varianza",
        "Risk parity",
        "HRP",
        "Máxima diversificación",
        "Min-CVaR",
        "Markowitz (máx Sharpe)",
    )
    pesos_metodos = {
        "Equiponderada (1/N)": (1 / 3, 1 / 3, 1 / 3),
        "Mínima varianza": (0.50, 0.30, 0.20),
        "Risk parity": (0.30, 0.45, 0.25),
        "HRP": (0.20, 0.50, 0.30),
        "Máxima diversificación": (0.25, 0.25, 0.50),
        "Min-CVaR": (0.10, 0.55, 0.35),
        "Markowitz (máx Sharpe)": (0.60, 0.20, 0.20),
    }
    asignaciones = {
        metodo: _asignacion(metodo, pesos_metodos[metodo], i + 0.3)
        for i, metodo in enumerate(metodos)
    }
    metricas_oos = {
        metodo: _metricas(retorno=0.04 + i / 100, volatilidad=0.10 + i / 100, sharpe=0.4 + i / 10)
        for i, metodo in enumerate(metodos)
    }
    puntos = pd.DataFrame(
        {
            "volatilidad": [0.08, 0.12, 0.16, 0.20, 0.24],
            "retorno": [0.04, 0.07, 0.09, 0.11, 0.13],
            "sharpe": [0.50, 0.58, 0.56, 0.55, 0.54],
            "peso·AAA": [0.60, 0.45, 0.35, 0.25, 0.15],
            "peso·BBB": [0.30, 0.35, 0.40, 0.45, 0.50],
            "peso·CCC": [0.10, 0.20, 0.25, 0.30, 0.35],
        }
    )
    niveles = pd.DataFrame(
        {
            "retorno": [0.04, 0.09, 0.13],
            "AAA": [0.60, 0.35, 0.15],
            "BBB": [0.30, 0.40, 0.50],
            "CCC": [0.10, 0.25, 0.35],
        },
        index=pd.Index([0.08, 0.16, 0.24], name="volatilidad"),
    )
    carteras = (
        SimpleNamespace(
            nivel="conservador",
            volatilidad_objetivo=0.10,
            pesos=pd.Series((0.45, 0.35, 0.20), index=activos),
            retorno_esperado=0.07,
            volatilidad_esperada=0.12,
            metricas_historicas=_metricas(retorno=0.06, volatilidad=0.11, sharpe=0.55),
        ),
        SimpleNamespace(
            nivel="moderado",
            volatilidad_objetivo=0.16,
            pesos=pd.Series((0.35, 0.40, 0.25), index=activos),
            retorno_esperado=0.09,
            volatilidad_esperada=0.16,
            metricas_historicas=_metricas(retorno=0.08, volatilidad=0.15, sharpe=0.60),
        ),
        SimpleNamespace(
            nivel="agresivo",
            volatilidad_objetivo=0.22,
            pesos=pd.Series((0.20, 0.45, 0.35), index=activos),
            retorno_esperado=0.12,
            volatilidad_esperada=0.22,
            metricas_historicas=_metricas(retorno=0.10, volatilidad=0.21, sharpe=0.48),
        ),
    )
    return SimpleNamespace(
        configuracion=SimpleNamespace(
            tickers=activos,
            fecha_inicio="2024-01-01",
            fecha_fin="2024-01-08",
            frecuencia_rebalanceo="M",
            ventana_estimacion=4,
            views_black_litterman=(),
            perfil_riesgo="moderado",
            volatilidad_objetivo=None,
            idioma_reporte="es",
        ),
        datos=SimpleNamespace(
            activos=activos,
            cierres=pd.DataFrame(index=fechas, data={"AAA": range(8), "BBB": range(10, 18), "CCC": range(20, 28)}),
            log_retornos=pd.DataFrame(
                {
                    "AAA": [0.01, -0.02, 0.01, 0.00, 0.02, -0.01, 0.01, 0.00],
                    "BBB": [0.00, 0.01, -0.01, 0.02, 0.00, -0.01, 0.01, 0.02],
                    "CCC": [-0.01, 0.00, 0.01, -0.01, 0.01, 0.00, 0.02, -0.02],
                },
                index=fechas,
            ),
        ),
        analisis=SimpleNamespace(
            correlacion_media=pd.DataFrame(
                [[1.0, 0.20, -0.10], [0.20, 1.0, 0.30], [-0.10, 0.30, 1.0]],
                index=activos,
                columns=activos,
            ),
            correlacion_cola=pd.DataFrame(
                [[1.0, 0.50, 0.10], [0.50, 1.0, 0.60], [0.10, 0.60, 1.0]],
                index=activos,
                columns=activos,
            ),
            diferencia_correlacion_cola=pd.DataFrame(
                [[0.0, 0.30, 0.20], [0.30, 0.0, 0.30], [0.20, 0.30, 0.0]],
                index=activos,
                columns=activos,
            ),
            observaciones_cola=3,
            pca=SimpleNamespace(
                varianza_explicada=pd.Series([0.55, 0.30, 0.15]),
                varianza_acumulada=pd.Series([0.55, 0.85, 1.00]),
            ),
            regimenes=pd.Series(["alcista", "bajista", "alcista", "crisis"]),
        ),
        analisis_actual=SimpleNamespace(),
        asignaciones=asignaciones,
        frontera=SimpleNamespace(puntos=puntos),
        monte_carlo=SimpleNamespace(
            metricas=pd.DataFrame(
                {"volatilidad": [0.10, 0.18], "retorno": [0.05, 0.10], "sharpe": [0.50, 0.56]}
            ),
            pesos=pd.DataFrame({"AAA": [0.5, 0.2], "BBB": [0.3, 0.5], "CCC": [0.2, 0.3]}),
        ),
        riesgo=SimpleNamespace(
            walk_forward=SimpleNamespace(
                equity=pd.DataFrame(
                    {metodo: [1.0, 1.01, 0.99, 1.03] for metodo in metodos},
                    index=fechas[:4],
                ),
                rebalanceos=(object(), object()),
            ),
            metricas=metricas_oos,
            metricas_por_regimen={
                metodo: pd.DataFrame({"retorno_anual": [0.03, -0.02]}, index=["alcista", "crisis"])
                for metodo in metodos
            },
            stress={"covid": SimpleNamespace(evaluable=True, observaciones=2, metricas=metricas_oos)},
            diversificacion_crisis=pd.DataFrame(
                {"enb_global": [2.4, 2.1], "enb_crisis": [1.5, 1.3]},
                index=["Equiponderada (1/N)", "Mínima varianza"],
            ),
            convexidad=pd.DataFrame(
                {
                    "ret_medio_todo_baja": [-0.01, -0.008],
                    "ret_medio_mixto": [0.001, 0.002],
                    "ret_medio_todo_sube": [0.012, 0.010],
                },
                index=["Equiponderada (1/N)", "Mínima varianza"],
            ),
        ),
        perfil_riesgo=SimpleNamespace(
            carteras=carteras,
            recomendada=carteras[1],
            niveles_frontera=niveles,
        ),
        objetivo="comparar",
    )


def test_plotly_expone_pesos_del_perfil_y_frontera_resaltada() -> None:
    figs = todas_las_figuras(paquete_minimo())

    assert {"pesos_recomendados", "pesos_niveles", "composicion_frontera", "convexidad"} <= set(figs)
    assert any("Perfil recomendado" in str(traza.name) for traza in figs["frontera"].data)


def test_html_prioriza_pesos_y_separa_promesa_de_realidad(tmp_path) -> None:
    ruta = generar_html(paquete_minimo(), tmp_path / "informe.html")
    contenido = ruta.read_text(encoding="utf-8")

    assert contenido.index("Pesos recomendados") < contenido.index("Tabla maestra")
    assert "Promesa vs realidad" in contenido
    assert "g_convexidad" in contenido
    assert "g_composicion_frontera" in contenido


def test_html_italiano_traduce_titulos_principales(tmp_path) -> None:
    paquete = paquete_minimo()
    paquete.configuracion.idioma_reporte = "it"

    ruta = generar_html(paquete, tmp_path / "informe_it.html")
    contenido = ruta.read_text(encoding="utf-8")

    assert "Pesi consigliati" in contenido
    assert "Promessa vs realtà" in contenido
    assert "Pesi per livello di rischio" in contenido
    assert "Pesos recomendados" not in contenido
    assert "Pesos por nivel de riesgo" not in contenido
    assert "Retorno esperado" not in contenido
    assert "Volatilidad esperada" not in contenido
    assert "Tabla maestra" not in contenido


def test_pngs_pdf_incluyen_figuras_del_bloque_pesos(tmp_path) -> None:
    pngs = generar_pngs(paquete_minimo(), tmp_path)

    assert {"pesos_recomendados", "pesos_niveles", "composicion_frontera", "convexidad"} <= set(pngs)

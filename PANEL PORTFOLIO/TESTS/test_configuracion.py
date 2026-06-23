"""Pruebas de la configuración declarativa del panel."""

from dataclasses import FrozenInstanceError

import pytest

from CONFIGURACION import config
from CONTRATOS.errores import ErrorConfiguracion
from CONTRATOS.validacion import cargar_configuracion, construir_configuracion


def test_configuracion_contiene_parametros_obligatorios() -> None:
    assert len(config.TICKERS) >= 2
    assert config.ACTIVO_REFERENCIA in config.TICKERS
    assert config.NIVEL_CONFIANZA == 0.95
    assert config.COSTE_TRANSACCION_PB >= 0
    assert config.PERFIL_RIESGO in {"conservador", "moderado", "agresivo", "personalizado"}
    assert config.IDIOMA_REPORTE in {"es", "it"}
    assert config.PERFIL_REGIMEN in {"conservador", "estandar", "sensible"}
    assert not hasattr(config, "RETORNO_OBJETIVO_ANUAL")
    assert set(config.VENTANAS_STRESS) == {
        "crisis_financiera_2008",
        "covid_2020",
        "crisis_2022",
    }


def test_rechaza_activo_referencia_fuera_de_la_cesta() -> None:
    with pytest.raises(ErrorConfiguracion, match="ACTIVO_REFERENCIA"):
        construir_configuracion(
            tickers=("AAA", "BBB"),
            activo_referencia="SPY",
            peso_maximo=0.60,
        )


def test_rechaza_limite_long_only_inviable() -> None:
    with pytest.raises(ErrorConfiguracion, match="inviable"):
        construir_configuracion(
            tickers=("AAA", "BBB", "CCC"),
            activo_referencia="AAA",
            peso_maximo=0.30,
        )


def test_carga_configuracion_tipado_e_inmutable() -> None:
    resultado = cargar_configuracion()

    assert resultado.tickers == tuple(config.TICKERS)
    assert resultado.perfil_riesgo == config.PERFIL_RIESGO
    assert resultado.idioma_reporte == config.IDIOMA_REPORTE
    assert resultado.ventanas_stress[0].inicio.tzinfo is None
    with pytest.raises(FrozenInstanceError):
        resultado.dias_anio = 365  # type: ignore[misc]


def test_acepta_perfil_riesgo_e_idioma_tipados() -> None:
    resultado = construir_configuracion(
        tickers=("AAA", "BBB"),
        activo_referencia="AAA",
        peso_maximo=0.60,
        perfil_riesgo="agresivo",
        idioma_reporte="it",
    )

    assert resultado.perfil_riesgo == "agresivo"
    assert resultado.idioma_reporte == "it"


def test_rechaza_perfil_personalizado_sin_volatilidad_objetivo() -> None:
    with pytest.raises(ErrorConfiguracion, match="VOLATILIDAD_OBJETIVO_ANUAL"):
        construir_configuracion(
            tickers=("AAA", "BBB"),
            activo_referencia="AAA",
            peso_maximo=0.60,
            perfil_riesgo="personalizado",
            volatilidad_objetivo=None,
        )


def test_rechaza_idioma_reporte_no_soportado() -> None:
    with pytest.raises(ErrorConfiguracion, match="IDIOMA_REPORTE"):
        construir_configuracion(
            tickers=("AAA", "BBB"),
            activo_referencia="AAA",
            peso_maximo=0.60,
            idioma_reporte="fr",
        )


def test_rechaza_ticker_no_textual_con_error_controlado() -> None:
    with pytest.raises(ErrorConfiguracion, match="TICKERS"):
        construir_configuracion(
            tickers=("AAA", 123),  # type: ignore[arg-type]
            activo_referencia="AAA",
            peso_maximo=0.60,
        )


def test_rechaza_coeficiente_de_view_no_numerico() -> None:
    with pytest.raises(ErrorConfiguracion, match="coeficiente"):
        construir_configuracion(
            tickers=("AAA", "BBB"),
            activo_referencia="AAA",
            peso_maximo=0.60,
            views_black_litterman=(
                {
                    "activos": {"AAA": "no-numerico"},
                    "retorno_anual": 0.10,
                    "confianza": 0.80,
                },
            ),
        )


def test_rechaza_stress_con_zona_horaria() -> None:
    with pytest.raises(ErrorConfiguracion, match="zona horaria"):
        construir_configuracion(
            tickers=("AAA", "BBB"),
            activo_referencia="AAA",
            peso_maximo=0.60,
            ventanas_stress={
                "episodio": ("2024-01-01T00:00:00+00:00", "2024-02-01")
            },
        )

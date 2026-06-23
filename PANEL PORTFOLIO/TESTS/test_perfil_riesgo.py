"""Pruebas de selección de cartera recomendada por perfil de riesgo."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from CONTRATOS.validacion import construir_configuracion
from RIESGO.perfil import evaluar_perfil


def test_evaluar_perfil_expone_recomendada_segun_configuracion() -> None:
    fechas = pd.date_range("2024-01-01", periods=6, freq="D")
    datos = SimpleNamespace(
        activos=("AAA", "BBB"),
        log_retornos=pd.DataFrame(
            {"AAA": [0.01, -0.01, 0.02, 0.00, 0.01, -0.01],
             "BBB": [0.00, 0.01, -0.01, 0.02, -0.01, 0.01]},
            index=fechas,
        ),
    )
    puntos = pd.DataFrame(
        {
            "volatilidad": [0.08, 0.12, 0.16, 0.20],
            "retorno": [0.03, 0.06, 0.08, 0.10],
            "peso·AAA": [0.70, 0.50, 0.30, 0.10],
            "peso·BBB": [0.30, 0.50, 0.70, 0.90],
        }
    )
    frontera = SimpleNamespace(puntos=puntos)
    cfg = construir_configuracion(
        tickers=("AAA", "BBB"),
        activo_referencia="AAA",
        peso_maximo=1.0,
        perfil_riesgo="agresivo",
        ventana_estimacion=252,
    )

    resultado = evaluar_perfil(datos, frontera, cfg)

    assert resultado.recomendada.nivel == "agresivo"
    assert resultado.recomendada.volatilidad_esperada >= 0.16

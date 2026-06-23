"""Pruebas de construcción de frontera eficiente."""

from __future__ import annotations

import pandas as pd

from CONTRATOS.validacion import construir_configuracion
from OPTIMIZACION.frontera import construir_frontera


def test_maximo_retorno_factible_es_automatico_desde_la_frontera() -> None:
    cfg = construir_configuracion(
        tickers=("AAA", "BBB"),
        activo_referencia="AAA",
        peso_maximo=0.60,
    )
    retornos = pd.Series({"AAA": 0.10, "BBB": 0.20})
    covarianza = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=("AAA", "BBB"),
        columns=("AAA", "BBB"),
    )

    frontera = construir_frontera(retornos, covarianza, cfg)

    assert not frontera.puntos.empty
    assert frontera.maximo_retorno_factible.metricas.retorno_anual == frontera.puntos["retorno"].max()
    assert "máximo retorno factible" in frontera.maximo_retorno_factible.diagnostico

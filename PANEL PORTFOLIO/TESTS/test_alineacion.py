"""Pruebas de carga, intersección de calendarios y log-retornos."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from CONTRATOS.errores import ErrorDatos
from DATOS.alineacion import alinear_y_calcular_retornos, recortar_datos
from DATOS.cargador import cargar_cierres


def test_intersecta_calendarios_sin_forward_fill() -> None:
    cierres = {
        "BTC": pd.Series(
            [100.0, 110.0, 121.0, 133.1],
            index=pd.to_datetime(
                ["2024-01-05", "2024-01-06", "2024-01-08", "2024-01-09"]
            ),
        ),
        "SPY": pd.Series(
            [200.0, 220.0, 244.0],
            index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]),
        ),
    }

    resultado = alinear_y_calcular_retornos(cierres, min_retornos=2)

    assert resultado.cierres.index.tolist() == list(
        pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    )
    np.testing.assert_allclose(
        resultado.log_retornos.to_numpy(),
        np.array(
            [
                [np.log(1.21), np.log(1.10)],
                [np.log(1.10), np.log(244.0 / 220.0)],
            ]
        ),
    )


def test_rechaza_cobertura_comun_insuficiente() -> None:
    indice = pd.date_range("2024-01-01", periods=3)
    cierres = {
        "AAA": pd.Series([10.0, 11.0, 12.0], index=indice),
        "BBB": pd.Series([20.0, 21.0, 22.0], index=indice),
    }

    with pytest.raises(ErrorDatos, match="2 < 3"):
        alinear_y_calcular_retornos(cierres, min_retornos=3)


def test_rechaza_columna_de_retornos_constante() -> None:
    indice = pd.date_range("2024-01-01", periods=4)
    cierres = {
        "AAA": pd.Series([10.0, 10.0, 10.0, 10.0], index=indice),
        "BBB": pd.Series([20.0, 21.0, 20.5, 22.0], index=indice),
    }

    with pytest.raises(ErrorDatos, match="sin variación"):
        alinear_y_calcular_retornos(cierres, min_retornos=3)


def test_recorta_n_retornos_y_n_mas_un_cierres() -> None:
    indice = pd.date_range("2024-01-01", periods=6)
    cierres = {
        "AAA": pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], index=indice),
        "BBB": pd.Series([20.0, 19.0, 21.0, 20.0, 22.0, 23.0], index=indice),
    }
    datos = alinear_y_calcular_retornos(cierres, min_retornos=5)

    recortados = recortar_datos(datos, n_retornos=3)

    assert len(recortados.log_retornos) == 3
    assert len(recortados.cierres) == 4
    assert recortados.log_retornos.index[0] == indice[-3]
    assert recortados.cierres.index[0] == indice[-4]


def test_cargador_rechaza_fechas_duplicadas(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "cierre": [10.0, 11.0],
        }
    ).to_parquet(tmp_path / "AAA_1d.parquet", index=False)

    with pytest.raises(ErrorDatos, match="duplicadas"):
        cargar_cierres(("AAA",), tmp_path)


def test_cargador_rechaza_esquema_inesperado(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "precio": [10.0, 11.0],
        }
    ).to_parquet(tmp_path / "AAA_1d.parquet", index=False)

    with pytest.raises(ErrorDatos, match="esquema"):
        cargar_cierres(("AAA",), tmp_path)

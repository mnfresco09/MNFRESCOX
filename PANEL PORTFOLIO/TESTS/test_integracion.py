"""Pruebas del orquestador del panel."""

from __future__ import annotations

import pandas as pd

from CONTRATOS.errores import ErrorDatos
from CONTRATOS.modelos import ResumenActivo
from ejecutar import main


def test_comando_desconocido_devuelve_error(capsys) -> None:
    assert main(["desconocido"]) == 2
    assert "descargar" in capsys.readouterr().err


def test_comando_descargar_informa_resumen(
    monkeypatch,
    capsys,
) -> None:
    resumen = ResumenActivo(
        ticker="AAA",
        archivo="AAA_1d.parquet",
        filas=10,
        fecha_inicio=pd.Timestamp("2024-01-01"),
        fecha_fin=pd.Timestamp("2024-01-10"),
        huecos_sospechosos=0,
        hueco_max_dias=1,
    )
    monkeypatch.setattr(
        "ejecutar.descargar_cesta",
        lambda *args, **kwargs: (resumen,),
    )

    assert main(["descargar"]) == 0
    salida = capsys.readouterr().out
    assert "AAA" in salida
    assert "10" in salida


def test_error_controlado_devuelve_codigo_uno(monkeypatch, capsys) -> None:
    def fallar(*args, **kwargs):
        raise ErrorDatos("DESCARGADOR", "fallo deliberado")

    monkeypatch.setattr("ejecutar.descargar_cesta", fallar)

    assert main(["descargar"]) == 1
    assert "[DESCARGADOR] fallo deliberado" in capsys.readouterr().err


def test_analizar_se_detiene_hasta_instalar_la_siguiente_fase(capsys) -> None:
    assert main(["analizar"]) == 1
    assert "todavía no está instalada" in capsys.readouterr().err

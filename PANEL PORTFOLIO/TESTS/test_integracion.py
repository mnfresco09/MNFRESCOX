"""Pruebas del orquestador del panel."""

from __future__ import annotations

from CONTRATOS.errores import ErrorDatos
from CONTRATOS.validacion import construir_configuracion
from ejecutar import main


def test_comando_desconocido_devuelve_error(capsys) -> None:
    assert main(["desconocido"]) == 2
    assert "descargar" in capsys.readouterr().err


def test_comando_descargar_informa_resumen(
    monkeypatch,
    capsys,
) -> None:
    def descarga_falsa(_configuracion) -> int:
        print("AAA 10")
        return 0

    monkeypatch.setattr("ejecutar._ejecutar_descarga", descarga_falsa)

    assert main(["descargar"]) == 0
    salida = capsys.readouterr().out
    assert "AAA" in salida
    assert "10" in salida


def test_error_controlado_devuelve_codigo_uno(monkeypatch, capsys) -> None:
    def fallar(*args, **kwargs):
        raise ErrorDatos("DESCARGADOR", "fallo deliberado")

    monkeypatch.setattr("ejecutar._ejecutar_descarga", fallar)

    assert main(["descargar"]) == 1
    assert "[DESCARGADOR] fallo deliberado" in capsys.readouterr().err


def test_analizar_despacha_pipeline_instalado(monkeypatch) -> None:
    llamado = {"valor": False}

    def analizar_falso(_configuracion) -> int:
        llamado["valor"] = True
        return 0

    monkeypatch.setattr("ejecutar._ejecutar_analisis_con_progreso", analizar_falso)

    assert main(["analizar"]) == 0
    assert llamado["valor"] is True


def test_configuracion_con_idioma_devuelve_copia_tipada() -> None:
    from ejecutar import _configuracion_con_idioma

    cfg = construir_configuracion(
        tickers=("AAA", "BBB"),
        activo_referencia="AAA",
        peso_maximo=0.60,
    )

    actualizada = _configuracion_con_idioma(cfg, "it")

    assert cfg.idioma_reporte == "es"
    assert actualizada.idioma_reporte == "it"

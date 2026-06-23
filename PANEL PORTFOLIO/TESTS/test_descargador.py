"""Pruebas del descargador y de la publicación transaccional."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from CONTRATOS.errores import ErrorDatos
from DESCARGADOR.descargador import descargar_cesta, descargar_desde_yahoo


def _serie(valores: list[float]) -> pd.Series:
    return pd.Series(
        valores,
        index=pd.date_range("2024-01-02", periods=len(valores), freq="D"),
        name="cierre",
    )


def _guardar_previo(ruta: Path, valores: list[float]) -> None:
    pd.DataFrame(
        {
            "fecha": pd.date_range("2023-01-02", periods=len(valores), freq="D"),
            "cierre": valores,
        }
    ).to_parquet(ruta, index=False)


def test_descarga_y_publica_toda_la_cesta(tmp_path: Path) -> None:
    def proveedor(ticker: str, inicio: str, fin: str) -> pd.Series:
        assert inicio == "2024-01-01"
        assert fin == "2024-02-01"
        return _serie([10.0, 11.0, 12.0] if ticker == "AAA" else [20.0, 21.0])

    resumenes = descargar_cesta(
        ("AAA", "GC=F"),
        "2024-01-01",
        "2024-02-01",
        tmp_path,
        proveedor,
    )

    assert [resumen.archivo for resumen in resumenes] == [
        "AAA_1d.parquet",
        "GC_F_1d.parquet",
    ]
    assert [resumen.filas for resumen in resumenes] == [3, 2]
    assert (tmp_path / "AAA_1d.parquet").exists()
    assert (tmp_path / "GC_F_1d.parquet").exists()


def test_no_reemplaza_historicos_si_un_activo_falla(tmp_path: Path) -> None:
    previo = tmp_path / "AAA_1d.parquet"
    _guardar_previo(previo, [10.0, 10.5])

    def proveedor(ticker: str, inicio: str, fin: str) -> pd.Series:
        if ticker == "BBB":
            return pd.Series(dtype=float)
        return _serie([11.0, 12.0])

    with pytest.raises(ErrorDatos, match="BBB"):
        descargar_cesta(
            ("AAA", "BBB"),
            "2024-01-01",
            "2024-02-01",
            tmp_path,
            proveedor,
        )

    conservado = pd.read_parquet(previo)
    assert conservado["cierre"].tolist() == [10.0, 10.5]
    assert not (tmp_path / "BBB_1d.parquet").exists()


def test_restaura_toda_la_cesta_si_falla_la_publicacion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guardar_previo(tmp_path / "AAA_1d.parquet", [10.0, 10.5])
    _guardar_previo(tmp_path / "BBB_1d.parquet", [20.0, 20.5])
    reemplazo_original = Path.replace
    publicaciones = 0

    def reemplazar_con_fallo(ruta: Path, destino: Path) -> Path:
        nonlocal publicaciones
        if Path(destino).parent == tmp_path:
            publicaciones += 1
            if publicaciones == 2:
                raise OSError("fallo simulado de publicación")
        return reemplazo_original(ruta, destino)

    monkeypatch.setattr(Path, "replace", reemplazar_con_fallo)

    with pytest.raises(ErrorDatos, match="publicar"):
        descargar_cesta(
            ("AAA", "BBB"),
            "2024-01-01",
            "2024-02-01",
            tmp_path,
            lambda ticker, inicio, fin: _serie([100.0, 101.0]),
        )

    assert pd.read_parquet(tmp_path / "AAA_1d.parquet")["cierre"].tolist() == [
        10.0,
        10.5,
    ]
    assert pd.read_parquet(tmp_path / "BBB_1d.parquet")["cierre"].tolist() == [
        20.0,
        20.5,
    ]


@pytest.mark.parametrize(
    ("valores", "mensaje"),
    [
        ([10.0, 0.0], "positivos"),
        ([10.0, float("nan")], "nulos"),
    ],
)
def test_rechaza_cierres_corruptos(
    tmp_path: Path,
    valores: list[float],
    mensaje: str,
) -> None:
    with pytest.raises(ErrorDatos, match=mensaje):
        descargar_cesta(
            ("AAA", "BBB"),
            "2024-01-01",
            "2024-02-01",
            tmp_path,
            lambda ticker, inicio, fin: _serie(valores),
        )


def test_rechaza_tickers_que_colisionan_en_el_mismo_archivo(tmp_path: Path) -> None:
    with pytest.raises(ErrorDatos, match="mismo archivo"):
        descargar_cesta(
            ("A-B", "A=B"),
            "2024-01-01",
            "2024-02-01",
            tmp_path,
            lambda ticker, inicio, fin: _serie([10.0, 11.0]),
        )


def test_yahoo_convierte_fecha_final_inclusiva_a_end_exclusivo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argumentos: dict[str, object] = {}

    def download(ticker: str, **kwargs):
        argumentos.update(kwargs)
        return pd.DataFrame(
            {"Adj Close": [10.0, 11.0]},
            index=pd.to_datetime(["2024-02-01", "2024-02-02"]),
        )

    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        SimpleNamespace(download=download),
    )

    resultado = descargar_desde_yahoo("AAA", "2024-01-01", "2024-02-01")

    assert argumentos["start"] == "2024-01-01"
    assert argumentos["end"] == "2024-02-02"
    assert resultado is not None
    assert len(resultado) == 2

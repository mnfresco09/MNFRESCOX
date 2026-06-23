"""Arranque del entorno: asegura que las dependencias estén instaladas.

Responsabilidad del punto de entrada, no de las capas de cálculo. Si falta un
paquete necesario, se instala automáticamente en vez de fallar. Si la instalación
no es posible, SE DETIENE diciendo exactamente qué falta y cómo instalarlo a mano
(coherente con el principio fail-fast: nunca seguir a medias).

Solo usa la biblioteca estándar para poder ejecutarse antes de importar nada más.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

# nombre_de_import -> nombre_en_pip
REQUISITOS: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "cvxpy": "cvxpy",
    "yfinance": "yfinance",
    "plotly": "plotly",
    "matplotlib": "matplotlib",
    "reportlab": "reportlab",
    "xlsxwriter": "XlsxWriter",
    "rich": "rich",
}


class ErrorEntorno(RuntimeError):
    """No se pudo dejar el entorno listo (y no se debe continuar a medias)."""


def _falta(nombre_import: str) -> bool:
    try:
        return importlib.util.find_spec(nombre_import) is None
    except (ImportError, ValueError):
        return True


def _instalar(paquete: str) -> None:
    """Instala un paquete con pip; reintenta en entornos 'externally managed'."""
    intentos = (
        [sys.executable, "-m", "pip", "install", paquete],
        [sys.executable, "-m", "pip", "install", "--break-system-packages", paquete],
    )
    ultimo = ""
    for comando in intentos:
        resultado = subprocess.run(comando, capture_output=True, text=True)
        if resultado.returncode == 0:
            return
        ultimo = (resultado.stderr or resultado.stdout).strip().splitlines()[-1:] or [""]
        ultimo = ultimo[0]
    raise ErrorEntorno(
        f"No se pudo instalar '{paquete}'. Instálalo a mano con:\n"
        f"    {sys.executable} -m pip install {paquete}\n"
        f"Detalle: {ultimo}"
    )


def asegurar_dependencias(anunciar=print) -> list[str]:
    """Comprueba e instala lo que falte. Devuelve la lista de lo instalado."""
    faltantes = [(imp, pkg) for imp, pkg in REQUISITOS.items() if _falta(imp)]
    instalados: list[str] = []
    for nombre_import, paquete in faltantes:
        anunciar(f"Falta «{paquete}»: instalando…")
        _instalar(paquete)
        instalados.append(paquete)
    return instalados

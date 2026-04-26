from __future__ import annotations

import urllib.request
from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"
_LIB_PATH = _ASSETS / "lightweight-charts.min.js"
_CDN = (
    "https://unpkg.com/lightweight-charts@4.2.0"
    "/dist/lightweight-charts.standalone.production.js"
)


def obtener_script_libreria() -> str:
    """
    Devuelve un bloque <script> con la librería TradingView Lightweight Charts.
    Si existe el archivo local lo embede directamente (standalone, sin internet).
    Si no existe intenta descargarlo una vez. Si falla, usa tag CDN.
    """
    if not _LIB_PATH.exists():
        _intentar_descarga()
    if _LIB_PATH.exists():
        js = _LIB_PATH.read_text(encoding="utf-8")
        return f"<script>\n{js}\n</script>"
    print("[HTML] ADVERTENCIA: sin libreria local, el HTML requiere internet.")
    return f'<script src="{_CDN}"></script>'


def _intentar_descarga() -> None:
    try:
        _ASSETS.mkdir(parents=True, exist_ok=True)
        print(f"[HTML] Descargando TradingView Lightweight Charts desde CDN...")
        with urllib.request.urlopen(_CDN, timeout=30) as resp:
            datos = resp.read()
        _LIB_PATH.write_bytes(datos)
        print(f"[HTML] Libreria guardada en {_LIB_PATH} ({len(datos):,} bytes).")
    except Exception as exc:
        print(f"[HTML] Descarga fallida: {exc}")

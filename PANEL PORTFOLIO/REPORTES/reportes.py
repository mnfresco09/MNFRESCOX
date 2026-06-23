"""Fachada de REPORTES: del PaqueteReporte a HTML + PDF + Excel + manifiesto.

Única puerta de entrada de la capa. No calcula nada: solo orquesta las salidas.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from CONTRATOS.modelos import PaqueteReporte, RutasReporte

from .excel import generar_excel
from .html import generar_html
from .pdf import generar_pdf


def generar_reportes(paquete: PaqueteReporte) -> RutasReporte:
    cfg = paquete.configuracion
    carpeta = Path(cfg.carpeta_salidas)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_html = generar_html(paquete, carpeta / "informe.html")
    ruta_pdf = generar_pdf(paquete, carpeta / "informe.pdf")
    ruta_excel = generar_excel(paquete, carpeta / "informe.xlsx")

    manifiesto = carpeta / "manifiesto_reporte.json"
    manifiesto.write_text(json.dumps({
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "tickers": list(cfg.tickers),
        "fecha_inicio": cfg.fecha_inicio,
        "fecha_fin": cfg.fecha_fin,
        "retornos_comunes": int(len(paquete.datos.log_retornos)),
        "archivos": {"html": ruta_html.name, "pdf": ruta_pdf.name, "excel": ruta_excel.name},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return RutasReporte(html=ruta_html, pdf=ruta_pdf, excel=ruta_excel, manifiesto=manifiesto)

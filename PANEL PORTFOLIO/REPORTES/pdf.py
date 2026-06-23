"""Informe PDF institucional (reportlab + gráficos matplotlib).

Documento serio, paginado, apto para imprimir y circular: portada, resumen con
indicadores, tabla maestra, gráficos con pie interpretativo, glosario y aviso de
honestidad. Misma sustancia que el HTML, en formato estático.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from CONTRATOS.modelos import PaqueteReporte

from .formato import AVISO_HONESTIDAD, GLOSARIO, num, pct
from .graficos_mpl import generar_pngs
from .narrativa import construir_secciones
from .tablas import tabla_maestra

_TINTA = colors.HexColor("#0F172A")
_ACENTO = colors.HexColor("#1D4ED8")
_SUAVE = colors.HexColor("#475569")
_LINEA = colors.HexColor("#D9DEE7")
_PANEL = colors.HexColor("#F6F8FB")

_PCT_COLS = {
    "Retorno esperado (in-sample)", "Volatilidad esperada (in-sample)",
    "Retorno anual (OOS)", "Volatilidad (OOS)", "Max drawdown (OOS)",
    "VaR 95% (OOS)", "CVaR 95% (OOS)",
}


def _estilos():
    s = getSampleStyleSheet()
    base = "Times-Roman"
    s.add(ParagraphStyle("Eyebrow", fontName="Helvetica-Bold", fontSize=9, textColor=_ACENTO,
                         spaceAfter=6, leading=12))
    s.add(ParagraphStyle("Titulo", fontName="Times-Bold", fontSize=26, textColor=_TINTA, leading=30, spaceAfter=8))
    s.add(ParagraphStyle("H2", fontName="Times-Bold", fontSize=16, textColor=_TINTA, spaceBefore=16,
                         spaceAfter=8, leading=19))
    s.add(ParagraphStyle("Cuerpo", fontName=base, fontSize=10.5, textColor=_TINTA, leading=15,
                         alignment=TA_JUSTIFY, spaceAfter=7))
    s.add(ParagraphStyle("Meta", fontName="Helvetica", fontSize=9.5, textColor=_SUAVE, spaceAfter=4))
    s.add(ParagraphStyle("Pie", fontName="Helvetica-Oblique", fontSize=8.5, textColor=_SUAVE, spaceAfter=12))
    s.add(ParagraphStyle("Aviso", fontName=base, fontSize=9.5, textColor=colors.HexColor("#7C2D12"),
                         leading=14, alignment=TA_JUSTIFY))
    s.add(ParagraphStyle("GlosarioT", fontName="Times-Bold", fontSize=10, textColor=_ACENTO, leading=13))
    s.add(ParagraphStyle("GlosarioD", fontName=base, fontSize=10, textColor=_SUAVE, leading=13))
    return s


def _kpi_table(paquete: PaqueteReporte, st) -> Table:
    metr = paquete.riesgo.metricas
    mejor = max(metr.items(), key=lambda kv: kv[1].sharpe)
    menor_dd = max(metr.items(), key=lambda kv: kv[1].max_drawdown)
    wf = paquete.riesgo.walk_forward
    datos = [
        ["Mejor Sharpe OOS", "Menor caída OOS", "Retornos comunes", "Rebalanceos"],
        [num(mejor[1].sharpe), pct(menor_dd[1].max_drawdown), f"{len(paquete.datos.log_retornos):,}", str(len(wf.rebalanceos))],
        [mejor[0], menor_dd[0], "tras alinear", f"{wf.equity.index[0].date()}→{wf.equity.index[-1].date()}"],
    ]
    t = Table(datos, colWidths=[4.1 * cm] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PANEL),
        ("BOX", (0, 0), (-1, -1), 0.5, _LINEA),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), _SUAVE),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 16),
        ("TEXTCOLOR", (0, 1), (-1, 1), _TINTA),
        ("FONTSIZE", (0, 2), (-1, 2), 7.5),
        ("TEXTCOLOR", (0, 2), (-1, 2), _SUAVE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _tabla_maestra_pdf(paquete: PaqueteReporte) -> Table:
    df = tabla_maestra(paquete)
    mejor = max(paquete.riesgo.metricas.items(), key=lambda kv: kv[1].sharpe)[0]
    # Para que quepa, abreviamos cabeceras.
    abrev = {
        "Retorno esperado (in-sample)": "Ret esp", "Volatilidad esperada (in-sample)": "Vol esp",
        "Retorno anual (OOS)": "Ret OOS", "Volatilidad (OOS)": "Vol OOS", "Sharpe (OOS)": "Sharpe",
        "Sortino (OOS)": "Sortino", "Calmar (OOS)": "Calmar", "Max drawdown (OOS)": "MaxDD",
        "VaR 95% (OOS)": "VaR", "CVaR 95% (OOS)": "CVaR",
    }
    cols = list(df.columns)
    cab = ["Método"] + [abrev.get(c, c.replace("peso · ", "")) for c in cols]
    filas = [cab]
    for metodo, fila in df.iterrows():
        celdas = [metodo]
        for c in cols:
            v = fila[c]
            celdas.append(pct(v, 1) if (c.startswith("peso ·") or c in _PCT_COLS) else num(v))
        filas.append(celdas)
    n = len(cols)
    t = Table(filas, colWidths=[3.0 * cm] + [(14.5 / n) * cm] * n, repeatRows=1)
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), _TINTA),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.6),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _LINEA),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    idx = list(df.index).index(mejor) + 1
    estilo.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#EFF6FF")))
    t.setStyle(TableStyle(estilo))
    return t


def generar_pdf(paquete: PaqueteReporte, ruta: Path) -> Path:
    st = _estilos()
    cfg = paquete.configuracion
    secciones = {s.titulo: s.parrafos for s in construir_secciones(paquete)}
    carpeta_png = Path(tempfile.mkdtemp(prefix="panel_pdf_"))
    pngs = generar_pngs(paquete, carpeta_png)

    doc = SimpleDocTemplate(str(ruta), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="PANEL PORTFOLIO — Informe de optimización")
    story = []
    P = lambda txt, estilo="Cuerpo": Paragraph(txt, st[estilo])

    def parrafos(titulo):
        for p in secciones.get(titulo, ()):
            story.append(P(p))

    def img(nombre, ancho=16.5, pie=""):
        story.append(Image(str(pngs[nombre]), width=ancho * cm, height=ancho * cm * 0.62))
        if pie:
            story.append(P(pie, "Pie"))

    # Portada / resumen
    story.append(P("ANÁLISIS CUANTITATIVO · DESCRIPTIVO · OUT-OF-SAMPLE", "Eyebrow"))
    story.append(P("Optimización y riesgo de una cartera multiactivo", "Titulo"))
    story.append(P(f"Cesta de {len(cfg.tickers)} activos: {', '.join(cfg.tickers)}", "Meta"))
    story.append(P(f"Periodo {cfg.fecha_inicio} a {cfg.fecha_fin} · rebalanceo {cfg.frecuencia_rebalanceo} · ventana {cfg.ventana_estimacion} días", "Meta"))
    story.append(Spacer(1, 10))
    story.append(P("Resumen ejecutivo", "H2"))
    for p in secciones["Resumen ejecutivo"][:-1]:
        story.append(P(p))
    story.append(Spacer(1, 6))
    story.append(_kpi_table(paquete, st))
    story.append(Spacer(1, 10))
    story.append(P(AVISO_HONESTIDAD, "Aviso"))

    story.append(PageBreak())
    story.append(P("Tabla maestra: 6 métodos comparados", "H2"))
    story.append(P("La fila resaltada es el método con mejor Sharpe out-of-sample. Las columnas "
                   "in-sample son lo esperado; las OOS, lo realmente realizado en el walk-forward."))
    story.append(_tabla_maestra_pdf(paquete))
    story.append(Spacer(1, 8))
    story.append(P("Composición de cada cartera", "H2"))
    img("pesos", pie="Reparto de cada método entre los activos de la cesta.")

    story.append(PageBreak())
    story.append(P("Plano riesgo-retorno", "H2"))
    parrafos("Los 6 métodos de asignación")
    img("frontera", pie="Frontera eficiente, nube de carteras aleatorias y posición in-sample de cada método.")

    story.append(PageBreak())
    story.append(P("Backtest walk-forward (out-of-sample)", "H2"))
    parrafos("Backtest walk-forward (out-of-sample)")
    img("equity", pie="Capital acumulado fuera de muestra; todas las curvas parten de 1.0.")
    img("drawdown", pie="Caída desde máximos: cuánto y cuánto tiempo estuvo cada método bajo el agua.")

    story.append(PageBreak())
    story.append(P("Análisis, correlación de cola y PCA", "H2"))
    parrafos("Análisis y diversificación")
    img("correlacion_media", ancho=11, pie="Correlación media en todo el periodo.")
    img("correlacion_cola", ancho=11, pie="Exceso de correlación en las colas: dónde se evapora la diversificación.")
    img("pca", pie="Número de factores independientes que mueven la cesta.")

    story.append(PageBreak())
    story.append(P("Regímenes, stress y diversificación en crisis", "H2"))
    parrafos("Regímenes y stress testing")
    parrafos("Diversificación en crisis")
    img("diversificacion", pie="Apuestas independientes en calma frente a crisis.")

    story.append(PageBreak())
    story.append(P("Glosario", "H2"))
    glos = [[Paragraph(k, st["GlosarioT"]), Paragraph(v, st["GlosarioD"])] for k, v in GLOSARIO.items()]
    tg = Table(glos, colWidths=[4.5 * cm, 12 * cm])
    tg.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _LINEA),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tg)
    story.append(Spacer(1, 14))
    story.append(P(AVISO_HONESTIDAD, "Aviso"))

    doc.build(story)
    return ruta

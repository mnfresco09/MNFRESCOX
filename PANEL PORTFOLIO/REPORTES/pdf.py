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

from .formato import num, pct
from .graficos_mpl import generar_pngs
from .i18n import columna_visible, glosario, metodo_visible, t
from .narrativa import construir_secciones
from .tablas import tabla_espejismo, tabla_maestra, tabla_pesos_niveles

_TINTA = colors.HexColor("#0F172A")
_ACENTO = colors.HexColor("#1D4ED8")
_SUAVE = colors.HexColor("#475569")
_LINEA = colors.HexColor("#D9DEE7")
_PANEL = colors.HexColor("#F6F8FB")

_PCT_COLS = {
    "Retorno esperado (in-sample)", "Volatilidad esperada (in-sample)",
    "Retorno anual (OOS)", "Volatilidad (OOS)", "Max drawdown (OOS)",
    "VaR 95% (OOS)", "CVaR 95% (OOS)", "Retorno esperado",
    "Volatilidad esperada", "VaR histórico", "CVaR histórico",
    "Max drawdown histórico", "Retorno realizado (OOS)",
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
    cartera = paquete.perfil_riesgo.recomendada
    mh = cartera.metricas_historicas
    datos = [
        [t(paquete, "retorno_esperado"), t(paquete, "volatilidad_esperada"), t(paquete, "var_historico"), t(paquete, "maxdd_historico")],
        [pct(cartera.retorno_esperado), pct(cartera.volatilidad_esperada), pct(mh.var), pct(mh.max_drawdown)],
        [
            t(paquete, "frontera_eficiente"),
            t(paquete, "perfil").format(perfil=cartera.nivel),
            t(paquete, "cola_diaria"),
            t(paquete, "pesos_fijos"),
        ],
    ]
    tabla = Table(datos, colWidths=[4.1 * cm] * 4)
    tabla.setStyle(TableStyle([
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
    return tabla


def _perfil_resumen(paquete: PaqueteReporte) -> str:
    cartera = paquete.perfil_riesgo.recomendada
    mh = cartera.metricas_historicas
    pesos = ", ".join(
        f"{activo} {pct(float(peso), 1)}"
        for activo, peso in cartera.pesos.sort_values(ascending=False).items()
    )
    return t(paquete, "resumen_perfil").format(
        perfil=cartera.nivel,
        pesos=pesos,
        retorno=pct(cartera.retorno_esperado),
        volatilidad=pct(cartera.volatilidad_esperada),
        var=pct(mh.var),
        cvar=pct(mh.cvar),
        maxdd=pct(mh.max_drawdown),
    )


def _tabla_df_pdf(
    df,
    indice: str,
    *,
    pct_cols: set[str] | None = None,
    abrev: dict[str, str] | None = None,
    fuente: float = 6.6,
    paquete: PaqueteReporte | None = None,
) -> Table:
    pct_cols = pct_cols or set()
    abrev = abrev or {}
    cols = list(df.columns)
    cab = [
        columna_visible(paquete, indice) if paquete is not None else indice,
        *[abrev.get(c, columna_visible(paquete, c) if paquete is not None else c.replace("peso · ", "")) for c in cols],
    ]
    filas = [cab]
    for etiqueta, fila in df.iterrows():
        etiqueta_txt = str(etiqueta)
        celdas = [metodo_visible(paquete, etiqueta_txt) if paquete is not None and indice == "Método" else etiqueta_txt]
        for c in cols:
            v = fila[c]
            celdas.append(pct(v, 1) if (c.startswith("peso ·") or c in pct_cols) else num(v))
        filas.append(celdas)
    n = max(1, len(cols))
    t = Table(filas, colWidths=[3.0 * cm] + [(14.5 / n) * cm] * n, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _TINTA),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), fuente),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _LINEA),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _tabla_maestra_pdf(paquete: PaqueteReporte) -> Table:
    df = tabla_maestra(paquete)
    mejor = max(paquete.riesgo.metricas.items(), key=lambda kv: kv[1].sharpe)[0]
    # Para que quepa, abreviamos cabeceras.
    if getattr(paquete.configuracion, "idioma_reporte", "es") == "it":
        abrev = {
            "Retorno esperado (in-sample)": "Rend att", "Volatilidad esperada (in-sample)": "Vol att",
            "Retorno anual (OOS)": "Rend OOS", "Volatilidad (OOS)": "Vol OOS", "Sharpe (OOS)": "Sharpe",
            "Sortino (OOS)": "Sortino", "Calmar (OOS)": "Calmar", "Max drawdown (OOS)": "MaxDD",
            "VaR 95% (OOS)": "VaR", "CVaR 95% (OOS)": "CVaR",
        }
    else:
        abrev = {
            "Retorno esperado (in-sample)": "Ret esp", "Volatilidad esperada (in-sample)": "Vol esp",
            "Retorno anual (OOS)": "Ret OOS", "Volatilidad (OOS)": "Vol OOS", "Sharpe (OOS)": "Sharpe",
            "Sortino (OOS)": "Sortino", "Calmar (OOS)": "Calmar", "Max drawdown (OOS)": "MaxDD",
            "VaR 95% (OOS)": "VaR", "CVaR 95% (OOS)": "CVaR",
        }
    cols = list(df.columns)
    cab = [columna_visible(paquete, "Método")] + [abrev.get(c, columna_visible(paquete, c).replace("peso · ", "")) for c in cols]
    filas = [cab]
    for metodo, fila in df.iterrows():
        celdas = [metodo_visible(paquete, metodo)]
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


def _tabla_niveles_pdf(paquete: PaqueteReporte) -> Table:
    if getattr(paquete.configuracion, "idioma_reporte", "es") == "it":
        abrev = {
            "Retorno esperado": "Rend att",
            "Volatilidad esperada": "Vol att",
            "VaR histórico": "VaR hist",
            "CVaR histórico": "CVaR hist",
            "Max drawdown histórico": "MaxDD stor",
        }
    else:
        abrev = {
            "Retorno esperado": "Ret esp",
            "Volatilidad esperada": "Vol esp",
            "VaR histórico": "VaR hist",
            "CVaR histórico": "CVaR hist",
            "Max drawdown histórico": "MaxDD hist",
        }
    df = tabla_pesos_niveles(paquete)
    return _tabla_df_pdf(df, "Nivel", pct_cols=_PCT_COLS, abrev=abrev, fuente=6.4, paquete=paquete)


def _tabla_espejismo_pdf(paquete: PaqueteReporte) -> Table:
    if getattr(paquete.configuracion, "idioma_reporte", "es") == "it":
        abrev = {
            "Sharpe esperado (in-sample)": "Sharpe att",
            "Sharpe realizado (OOS)": "Sharpe OOS",
            "Degradación de Sharpe": "Degrado",
            "Retorno esperado (in-sample)": "Rend att",
            "Retorno realizado (OOS)": "Rend OOS",
        }
    else:
        abrev = {
            "Sharpe esperado (in-sample)": "Sharpe esp",
            "Sharpe realizado (OOS)": "Sharpe OOS",
            "Degradación de Sharpe": "Degrad.",
            "Retorno esperado (in-sample)": "Ret esp",
            "Retorno realizado (OOS)": "Ret OOS",
        }
    df = tabla_espejismo(paquete)
    return _tabla_df_pdf(df, "Método", pct_cols=_PCT_COLS, abrev=abrev, fuente=6.8, paquete=paquete)


def generar_pdf(paquete: PaqueteReporte, ruta: Path) -> Path:
    st = _estilos()
    cfg = paquete.configuracion
    secciones = {s.titulo: s.parrafos for s in construir_secciones(paquete)}
    carpeta_png = Path(tempfile.mkdtemp(prefix="panel_pdf_"))
    pngs = generar_pngs(paquete, carpeta_png)

    doc = SimpleDocTemplate(str(ruta), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"PANEL PORTFOLIO — {t(paquete, 'titulo')}")
    story = []
    P = lambda txt, estilo="Cuerpo": Paragraph(txt, st[estilo])

    def parrafos(titulo):
        for p in secciones.get(titulo, ()):
            story.append(P(p))

    def img(nombre, ancho=16.5, pie=""):
        story.append(Image(str(pngs[nombre]), width=ancho * cm, height=ancho * cm * 0.62))
        if pie:
            story.append(P(pie, "Pie"))

    # Portada / recomendación
    story.append(P(t(paquete, "eyebrow").upper(), "Eyebrow"))
    story.append(P(t(paquete, "titulo"), "Titulo"))
    story.append(P(t(paquete, "meta_analisis").format(
        n=len(cfg.tickers), inicio=cfg.fecha_inicio, fin=cfg.fecha_fin,
        rebalanceo=cfg.frecuencia_rebalanceo, ventana=cfg.ventana_estimacion,
    ), "Meta"))
    story.append(Spacer(1, 10))
    story.append(P(t(paquete, "recomendacion"), "H2"))
    story.append(P(_perfil_resumen(paquete)))
    story.append(Spacer(1, 6))
    story.append(_kpi_table(paquete, st))
    story.append(Spacer(1, 10))
    img("pesos_recomendados", pie=t(paquete, "pdf_pesos_pie"))
    story.append(P(t(paquete, "honestidad"), "Aviso"))

    story.append(PageBreak())
    story.append(P(t(paquete, "niveles"), "H2"))
    story.append(P(t(paquete, "niveles_intro")))
    story.append(_tabla_niveles_pdf(paquete))
    story.append(Spacer(1, 8))
    img("pesos_niveles", pie=t(paquete, "pdf_niveles_pie"))
    img("composicion_frontera", pie=t(paquete, "pdf_area_pie"))

    story.append(PageBreak())
    story.append(P(t(paquete, "nav_niveles"), "H2"))
    img("frontera", pie=t(paquete, "pdf_frontera_pie"))

    story.append(PageBreak())
    story.append(P(t(paquete, "covarianza"), "H2"))
    parrafos("Análisis y diversificación")
    img("correlacion_media", ancho=11, pie=t(paquete, "corr_media_titulo"))
    img("correlacion_cola", ancho=11, pie=t(paquete, "corr_cola_titulo"))
    img("pca", pie=t(paquete, "pca_titulo"))

    story.append(PageBreak())
    story.append(P(t(paquete, "validacion"), "H2"))
    parrafos("Backtest walk-forward (out-of-sample)")
    story.append(P(t(paquete, "tabla_oos"), "H2"))
    story.append(_tabla_maestra_pdf(paquete))
    story.append(Spacer(1, 8))
    story.append(P(t(paquete, "promesa_realidad"), "H2"))
    story.append(_tabla_espejismo_pdf(paquete))
    img("equity", pie=t(paquete, "equity_titulo"))
    img("drawdown", pie=t(paquete, "drawdown_titulo"))
    img("convexidad", pie=t(paquete, "pdf_convexidad_pie"))

    story.append(PageBreak())
    story.append(P(t(paquete, "regimenes"), "H2"))
    parrafos("Regímenes y stress testing")
    parrafos("Diversificación en crisis")
    img("diversificacion", pie=t(paquete, "diversificacion_titulo"))
    img("pesos", pie=t(paquete, "pdf_metodos_pie"))

    story.append(PageBreak())
    story.append(P(t(paquete, "nav_glosario"), "H2"))
    glos = [[Paragraph(k, st["GlosarioT"]), Paragraph(v, st["GlosarioD"])] for k, v in glosario(paquete).items()]
    tg = Table(glos, colWidths=[4.5 * cm, 12 * cm])
    tg.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _LINEA),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tg)
    story.append(Spacer(1, 14))
    story.append(P(t(paquete, "honestidad"), "Aviso"))

    doc.build(story)
    return ruta

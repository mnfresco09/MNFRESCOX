"""Dashboard ejecutivo PDF (reportlab).

Informe minimalista de 6 secciones + glosario.  Consume el mismo
`ReportPayload` y los mismos PNG que el HTML.  No recalcula nada.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

from CONTRATOS.modelos import ReportPayload

from . import estilo, graficos, narrativa

# ── colour aliases ──────────────────────────────────────────────────────────
_TINTA = colors.HexColor(estilo.TINTA)
_SUAVE = colors.HexColor(estilo.SUAVE)
_MUTED = colors.HexColor(estilo.MUTED)
_LINEA = colors.HexColor(estilo.LINEA)
_LINEA_FINA = colors.HexColor(estilo.LINEA_FINA)
_ACENTO = colors.HexColor(estilo.ACENTO)
_PANEL = colors.HexColor(estilo.PANEL)
_NEG = colors.HexColor(estilo.NEG)
_VERDE = colors.HexColor(estilo.VERDE)
_AMBAR = colors.HexColor(estilo.AMBAR)
_HEADER_BG = colors.HexColor(estilo.HEADER_BG)
_HEADER_FG = colors.HexColor(estilo.HEADER_FG)
_WIN = colors.HexColor("#ECFDF3")

_PAGE_W, _PAGE_H = A4
_USABLE = _PAGE_W - 3.0 * cm  # left + right margins


# ── safe formatters (toleran None / NaN / inf) ──────────────────────────────
def _pct_seg(x, dec: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return estilo.pct(x, dec)


def _num_seg(x, dec: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{dec}f}"


# ── paragraph styles ────────────────────────────────────────────────────────
def _estilos() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}

    # Cover page
    s["titulo"] = ParagraphStyle(
        "titulo", parent=base["Title"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=_TINTA, alignment=TA_CENTER, spaceAfter=6,
    )
    s["subtitulo"] = ParagraphStyle(
        "subtitulo", parent=base["Normal"],
        fontName="Helvetica", fontSize=11, leading=15,
        textColor=_SUAVE, alignment=TA_CENTER, spaceAfter=20,
    )
    s["cover_meta"] = ParagraphStyle(
        "cover_meta", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=9, leading=14,
        textColor=_TINTA, alignment=TA_CENTER, spaceAfter=4,
    )
    s["cover_date"] = ParagraphStyle(
        "cover_date", parent=base["Normal"],
        fontName="Helvetica", fontSize=8, leading=12,
        textColor=_MUTED, alignment=TA_CENTER, spaceBefore=30,
    )

    # Section headers
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=_TINTA, spaceBefore=14, spaceAfter=8,
    )

    # Body — bold by default
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=9, leading=13,
        textColor=_TINTA, spaceAfter=6,
    )

    # Body — normal weight (for secondary narrative)
    s["body_light"] = ParagraphStyle(
        "body_light", parent=base["Normal"],
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=_TINTA, spaceAfter=6,
    )

    # Footnotes / notes (not bold)
    s["nota"] = ParagraphStyle(
        "nota", parent=base["Normal"],
        fontName="Helvetica", fontSize=8, leading=11,
        textColor=_MUTED, spaceAfter=4,
    )

    # Glossary body
    s["gloss_body"] = ParagraphStyle(
        "gloss_body", parent=base["Normal"],
        fontName="Helvetica", fontSize=8, leading=11,
        textColor=_TINTA, spaceAfter=0,
    )
    s["gloss_term"] = ParagraphStyle(
        "gloss_term", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=8, leading=11,
        textColor=_TINTA, spaceAfter=0,
    )

    return s


# ── helper: thin horizontal separator ───────────────────────────────────────
def _linea_separadora() -> Table:
    t = Table([[""]], colWidths=[_USABLE], rowHeights=[1])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, _LINEA),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


# ── helper: data table with thin lines + gray headers ───────────────────────
def _tabla(datos: list[list], anchos: list[float],
           resaltar_fila: int | None = None) -> Table:
    t = Table(datos, colWidths=anchos, hAlign="LEFT")
    cmds = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), _HEADER_FG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        # Data rows
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), _TINTA),
        # Alignment
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        # Ultra-thin lines
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, _LINEA),
        ("LINEABOVE", (0, 0), (-1, 0), 0.3, _LINEA),
        # Padding
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if resaltar_fila is not None:
        cmds.append(("BACKGROUND", (0, resaltar_fila), (-1, resaltar_fila), _WIN))
    t.setStyle(TableStyle(cmds))
    return t


# ── helper: KPI row (white bg + thin border) ───────────────────────────────
def _kpi_row(items: list[tuple[str, str, str]]) -> Table:
    """Row of KPI cards.  Each item = (big_value, label, hex_color)."""
    _base = getSampleStyleSheet()["Normal"]
    celdas = []
    for valor, titulo, color_hex in items:
        p = Paragraph(
            f'<font size="14" color="{color_hex}"><b>{valor}</b></font>'
            f'<br/><font size="7" color="{estilo.SUAVE}">{titulo}</font>',
            _base,
        )
        celdas.append(p)
    n = len(celdas)
    col_w = min(4.3 * cm, _USABLE / n)
    t = Table([celdas], colWidths=[col_w] * n)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PANEL),
        ("BOX", (0, 0), (-1, -1), 0.3, _LINEA),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, _LINEA),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ── main entry point ────────────────────────────────────────────────────────
def generar_pdf(payload: ReportPayload, ruta: Path, figuras=None) -> Path:
    cfg = payload.configuracion
    carpeta = ruta.parent

    # Generate figures if not provided
    if figuras is None:
        figuras = graficos.generar_todos(payload, carpeta / "assets")
    rutas = {k: v[0] for k, v in figuras.items()}

    s = _estilos()
    rec = payload.recomendada
    sim = rec.simulacion
    f = rec.forecast
    cap = cfg.capital_base
    idi = cfg.idioma_reporte

    doc = SimpleDocTemplate(
        str(ruta), pagesize=A4,
        topMargin=1.6 * cm, bottomMargin=1.4 * cm,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        title="Informe de cartera",
    )

    E: list = []

    # ================================================================
    # PORTADA
    # ================================================================
    E.append(Spacer(1, 5.5 * cm))
    E.append(Paragraph(narrativa.t("titulo_informe", idi), s["titulo"]))
    E.append(Paragraph(
        narrativa.t("subtitulo_informe", idi), s["subtitulo"],
    ))
    E.append(Spacer(1, 1.5 * cm))
    E.append(_linea_separadora())
    E.append(Spacer(1, 18))

    tickers_str = ", ".join(cfg.tickers)
    meta_lines = [
        f"<b>{narrativa.t('meta_activos', idi).capitalize()}:</b>  {tickers_str}",
        f"<b>{narrativa.t('meta_periodo', idi)}:</b>  {cfg.fecha_inicio}  →  {cfg.fecha_fin}",
        f"<b>{narrativa.t('meta_capital', idi)}:</b>  {estilo.dinero(cap)}",
        f"<b>{narrativa.t('meta_horizonte', idi)}:</b>  {cfg.horizonte_dias} {narrativa.t('meta_dias', idi)}",
        f"<b>{narrativa.t('meta_motor', idi)}:</b>  {cfg.optimization_engine}",
    ]
    for line in meta_lines:
        E.append(Paragraph(line, s["cover_meta"]))
        E.append(Spacer(1, 2))

    E.append(Spacer(1, 2.5 * cm))
    E.append(_linea_separadora())
    E.append(Paragraph(f"{narrativa.t('meta_fecha', idi)}: {date.today().isoformat()}", s["cover_date"]))
    E.append(PageBreak())

    # ================================================================
    # 1 · RESUMEN EJECUTIVO
    # ================================================================
    r = payload.regimen
    E.append(Paragraph(narrativa.t("sec_resumen", idi), s["h2"]))
    E.append(Spacer(1, 6))

    E.append(_kpi_row([
        (r.etiqueta.replace("_", " ").title(), narrativa.t("kpi_regimen", idi), estilo.AMBAR),
        (estilo.pct(r.volatilidad_actual), narrativa.t("kpi_vol_tactica", idi), estilo.TINTA),
        (f"{rec.motor_optimizacion}/{estilo.nombre_nivel(rec.nivel, idi)}",
         narrativa.t("kpi_cartera_rec", idi), estilo.VERDE),
        (f"{rec.score:.2f}", narrativa.t("kpi_score", idi), estilo.ACENTO),
    ]))
    E.append(Spacer(1, 14))

    E.append(Paragraph(narrativa.texto_resumen_ejecutivo(payload), s["body"]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(narrativa.texto_por_que_gana(payload), s["body_light"]))
    E.append(Spacer(1, 12))
    E.append(_linea_separadora())

    # ================================================================
    # 2 · CARTERA RECOMENDADA
    # ================================================================
    E.append(Spacer(1, 12))
    E.append(Paragraph(narrativa.t("sec_cartera", idi), s["h2"]))
    E.append(Spacer(1, 6))

    # Side by side: pesos chart + MCR table (3 cols: Activo, Peso, Contrib.)
    img_pesos = Image(str(rutas["pesos"]), width=8.0 * cm, height=6.5 * cm)

    mcr_header = [narrativa.t("col_activo", idi), narrativa.t("col_peso", idi), narrativa.t("col_contrib_riesgo", idi)]
    mcr_data = [mcr_header]
    for a in rec.pesos.index:
        mcr_data.append([
            a,
            estilo.pct(float(rec.pesos[a])),
            estilo.pct(float(rec.descomposicion.contribucion_pct[a])),
        ])
    mcr_table = _tabla(mcr_data, [2.5 * cm, 2.0 * cm, 2.8 * cm])

    dual_pesos = Table(
        [[img_pesos, mcr_table]],
        colWidths=[8.4 * cm, 8.0 * cm],
    )
    dual_pesos.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    E.append(dual_pesos)
    E.append(Spacer(1, 14))

    # MCR chart below
    E.append(Image(str(rutas["mcr"]), width=14.0 * cm, height=7.0 * cm))
    E.append(PageBreak())

    # Equity + drawdown histórico de la cartera seleccionada
    E.append(Paragraph(narrativa.t("sub_equity_dd", idi), s["h2"]))
    E.append(Spacer(1, 6))
    E.append(Image(str(rutas["equity_drawdown"]), width=16.0 * cm, height=8.0 * cm))
    E.append(Spacer(1, 6))
    E.append(Paragraph(narrativa.t("nota_equity_dd", idi), s["nota"]))
    E.append(Spacer(1, 14))

    # Métricas históricas realizadas (in-sample): MaxDD exacto, CAGR, Sharpe, Calmar
    mh = getattr(payload, "metricas_historicas", None)
    if mh is not None:
        E.append(Paragraph(narrativa.t("sub_metricas_hist", idi), s["h2"]))
        E.append(Spacer(1, 6))
        E.append(_kpi_row([
            (_pct_seg(mh.max_drawdown), narrativa.t("kpi_maxdd", idi), estilo.NEG),
            (_pct_seg(mh.cagr), narrativa.t("kpi_cagr", idi), estilo.VERDE),
            (_num_seg(mh.sharpe_historico), narrativa.t("kpi_sharpe_hist", idi), estilo.TINTA),
            (_num_seg(mh.calmar), narrativa.t("kpi_calmar", idi), estilo.ACENTO),
        ]))
        E.append(Spacer(1, 6))
        E.append(Paragraph(narrativa.t("nota_metricas_hist", idi), s["nota"]))
        E.append(Spacer(1, 14))

    # Matriz de correlación — ¿cómo se relacionan?
    E.append(Paragraph(narrativa.t("ap_correlacion", idi), s["h2"]))
    E.append(Spacer(1, 6))
    E.append(Image(str(rutas["correlacion"]), width=11.0 * cm, height=9.8 * cm))

    # Correlación media móvil (252 días) — si hay serie disponible
    if "correlacion_rolling" in rutas:
        ventana = mh.ventana_rolling if mh is not None else 252
        E.append(Spacer(1, 12))
        E.append(Paragraph(narrativa.t("sub_corr_rolling", idi, n=ventana), s["h2"]))
        E.append(Spacer(1, 6))
        E.append(Image(str(rutas["correlacion_rolling"]), width=16.0 * cm, height=6.6 * cm))
    E.append(PageBreak())

    # ================================================================
    # 3 · RIESGO A HORIZONTE
    # ================================================================
    E.append(Paragraph(
        narrativa.t("sec_riesgo", idi, n=cfg.horizonte_dias),
        s["h2"],
    ))
    E.append(Spacer(1, 6))

    # Fan chart full width
    E.append(Image(str(rutas["fan_chart"]), width=16.0 * cm, height=8.0 * cm))
    E.append(Spacer(1, 12))

    # 4 KPIs
    E.append(_kpi_row([
        (estilo.pct(sim.retorno_mediano), narrativa.t("kpi_ret_mediano", idi), estilo.TINTA),
        (estilo.pct(sim.perdida_p5), narrativa.t("kpi_adverso_p5", idi), estilo.NEG),
        (f"{sim.prob_perdida:.0%}", narrativa.t("kpi_prob_perdida", idi), estilo.TINTA),
        (estilo.pct(sim.cdar_30d), narrativa.t("kpi_cdar", idi), estilo.NEG),
    ]))
    E.append(Spacer(1, 12))

    E.append(Paragraph(narrativa.texto_conclusion_riesgo(payload), s["body"]))
    E.append(PageBreak())

    # ================================================================
    # 4 · VaR DIARIO
    # ================================================================
    E.append(Paragraph(narrativa.t("sec_var", idi), s["h2"]))
    E.append(Spacer(1, 6))

    # VaR table (simplified: Method, VaR 95%, VaR 99%)
    var_header = [narrativa.t("col_metodo", idi), "VaR 95%", "VaR 99%"]
    var_data = [var_header]
    for nombre, v95, v99 in (
        (narrativa.t("metodo_historico", idi), f.var_hist_95, f.var_hist_99),
        (narrativa.t("metodo_parametrico", idi), f.var_param_95, f.var_param_99),
        ("FHS ●", f.var_fhs_95, f.var_fhs_99),
    ):
        var_data.append([
            nombre,
            f"{estilo.pct(v95, 2)}  ({v95 * cap:,.0f} €)",
            f"{estilo.pct(v99, 2)}  ({v99 * cap:,.0f} €)",
        ])

    img_var = Image(str(rutas["var_forecast"]), width=7.6 * cm, height=5.4 * cm)
    var_tbl = _tabla(var_data, [2.2 * cm, 3.2 * cm, 3.2 * cm], resaltar_fila=3)

    dual_var = Table(
        [[img_var, var_tbl]],
        colWidths=[8.0 * cm, 9.0 * cm],
    )
    dual_var.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    E.append(dual_var)
    E.append(Spacer(1, 12))

    E.append(Paragraph(narrativa.texto_conclusion_var(payload), s["body"]))
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        f"● FHS. {narrativa.t('nota_var_convenciones', idi)}",
        s["nota"],
    ))
    E.append(PageBreak())

    # ================================================================
    # 5 · COMPARATIVA DE CANDIDATOS
    # ================================================================
    E.append(Paragraph(narrativa.t("sec_comparativa", idi), s["h2"]))
    E.append(Spacer(1, 6))

    # Simplified master table: 8 columns
    comp_header = [narrativa.t("col_motor", idi), narrativa.t("col_perfil", idi),
                    narrativa.t("col_pesos", idi),
                    narrativa.t("col_retorno", idi), narrativa.t("col_vol", idi),
                    narrativa.t("col_var99", idi), narrativa.t("col_score", idi),
                    narrativa.t("col_decision", idi)]
    comp_data = [comp_header]
    win_idx = None
    
    def _pesos_compactos(pesos, n=4):
        top = pesos[pesos > 0.005].sort_values(ascending=False).head(n)
        return " · ".join(f"{a.split('.')[0]} {v * 100:.0f}%" for a, v in top.items())
        
    for i, c in enumerate(payload.candidatos, start=1):
        is_winner = (
            (c.motor_optimizacion, c.nivel)
            == (rec.motor_optimizacion, rec.nivel)
        )
        if is_winner:
            win_idx = i
        
        # Formatear el texto de pesos
        txt_pesos = _pesos_compactos(c.pesos, 4)
        
        comp_data.append([
            c.motor_optimizacion or "—",
            estilo.nombre_nivel(c.nivel, idi),
            Paragraph(txt_pesos, s["body_light"]),
            estilo.pct(c.retorno_esperado),
            estilo.pct(c.volatilidad_tactica),
            estilo.pct(c.forecast.var_fhs_99, 2),
            f"{c.score:.2f}",
            f"✓ {narrativa.t('decision_recomendada', idi)}" if is_winner else "—",
        ])

    E.append(_tabla(
        comp_data,
        [2.0 * cm, 2.0 * cm, 4.0 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm, 1.4 * cm, 3.0 * cm],
        resaltar_fila=win_idx,
    ))
    E.append(Spacer(1, 14))

    # Comparativa scores chart
    E.append(Image(str(rutas["comparativa"]), width=14.0 * cm, height=7.0 * cm))
    E.append(Spacer(1, 12))

    E.append(Paragraph(narrativa.texto_por_que_gana(payload), s["body"]))
    E.append(PageBreak())

    # ================================================================
    # 6 · GLOSARIO Y METODOLOGÍA
    # ================================================================
    E.append(Paragraph(narrativa.t("sec_glosario", idi), s["h2"]))
    E.append(Spacer(1, 8))

    glos = narrativa.glosario(idi)
    glos_header = [narrativa.t("gloss_termino", idi), narrativa.t("gloss_definicion", idi)]
    glos_data = [glos_header]
    for termino, definicion in glos:
        glos_data.append([
            Paragraph(termino, s["gloss_term"]),
            Paragraph(definicion, s["gloss_body"]),
        ])

    glos_table = Table(glos_data, colWidths=[4.0 * cm, _USABLE - 4.0 * cm],
                       hAlign="LEFT")
    glos_table.setStyle(TableStyle([
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), _HEADER_FG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        # Data
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), _TINTA),
        # Lines
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, _LINEA),
        ("LINEABOVE", (0, 0), (-1, 0), 0.3, _LINEA),
        # Padding
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    E.append(glos_table)
    E.append(Spacer(1, 20))

    E.append(_linea_separadora())
    E.append(Spacer(1, 6))
    E.append(Paragraph(
        narrativa.t("footer_glosario", idi),
        s["nota"],
    ))

    # Build
    doc.build(E)
    return ruta

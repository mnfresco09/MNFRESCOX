"""Dashboard ejecutivo PDF (reportlab). Mismo contenido y orden que el HTML,
consumiendo el MISMO `ReportPayload` y los MISMOS PNG. No recalcula nada.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from CONTRATOS.modelos import ReportPayload

from . import estilo, graficos

_TINTA = colors.HexColor(estilo.TINTA)
_ACENTO = colors.HexColor(estilo.ACENTO)
_PANEL = colors.HexColor(estilo.PANEL)
_NEG = colors.HexColor(estilo.NEG)
_VERDE = colors.HexColor(estilo.VERDE)
_LINEA = colors.HexColor(estilo.LINEA)
_WIN = colors.HexColor("#ECFDF3")


def _estilos():
    base = getSampleStyleSheet()
    s = {}
    s["eyebrow"] = ParagraphStyle("eyebrow", parent=base["Normal"], fontSize=8,
                                  textColor=_ACENTO, leading=11, spaceAfter=2)
    s["h1"] = ParagraphStyle("h1", parent=base["Title"], fontSize=20, textColor=_TINTA,
                             leading=23, spaceAfter=4, alignment=0)
    s["meta"] = ParagraphStyle("meta", parent=base["Normal"], fontSize=8.5,
                               textColor=colors.HexColor(estilo.SUAVE), leading=12, spaceAfter=6)
    s["foco"] = ParagraphStyle("foco", parent=base["Normal"], fontSize=9.5, textColor=_TINTA,
                               leading=13, backColor=_PANEL, borderPadding=8, spaceAfter=10,
                               leftIndent=4)
    s["h2"] = ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, textColor=_TINTA,
                             leading=16, spaceBefore=12, spaceAfter=6)
    s["h3"] = ParagraphStyle("h3", parent=base["Heading3"], fontSize=10.5, textColor=_ACENTO,
                             leading=13, spaceBefore=10, spaceAfter=4)
    s["nota"] = ParagraphStyle("nota", parent=base["Normal"], fontSize=8.5,
                               textColor=colors.HexColor(estilo.SUAVE), leading=12, spaceAfter=6)
    return s


def _tabla(datos, anchos, resaltar_fila=None) -> Table:
    t = Table(datos, colWidths=anchos, hAlign="LEFT")
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), _TINTA),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (0, -1), _PANEL),
        ("ROWBACKGROUNDS", (1, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBFD")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, _LINEA),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]
    if resaltar_fila is not None:
        cmds.append(("BACKGROUND", (0, resaltar_fila), (-1, resaltar_fila), _WIN))
    t.setStyle(TableStyle(cmds))
    return t


def _kpi_row(items) -> Table:
    """Fila de tarjetas KPI (valor grande + título)."""
    celdas = []
    for valor, titulo, color in items:
        celdas.append([Paragraph(f'<font size=14 color="{color}"><b>{valor}</b></font>'
                                 f'<br/><font size=7.5 color="#475569">{titulo}</font>', getSampleStyleSheet()["Normal"])])
    fila = [c[0] for c in celdas]
    t = Table([fila], colWidths=[4.3 * cm] * len(fila))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _PANEL),
        ("BOX", (0, 0), (-1, -1), 0.5, _LINEA),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def generar_pdf(payload: ReportPayload, ruta: Path, figuras=None) -> Path:
    cfg = payload.configuracion
    carpeta = ruta.parent
    if figuras is None:
        figuras = graficos.generar_todos(payload, carpeta / "assets")
    rutas = {k: v[0] for k, v in figuras.items()}
    s = _estilos()
    rec = payload.recomendada
    sim = rec.simulacion
    f = rec.forecast
    cap = cfg.capital_base
    idi = cfg.idioma_reporte

    doc = SimpleDocTemplate(str(ruta), pagesize=A4, topMargin=1.4 * cm, bottomMargin=1.3 * cm,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            title="PANEL PORTFOLIO — Motor de Riesgo Predictivo")
    E = []
    E.append(Paragraph("MOTOR DE RIESGO PREDICTIVO · BUY-SIDE", s["eyebrow"]))
    E.append(Paragraph("Decisión de cartera y riesgo prospectivo", s["h1"]))
    E.append(Paragraph(
        f"{len(cfg.tickers)} activos · {', '.join(cfg.tickers)} · {cfg.fecha_inicio} → {cfg.fecha_fin} "
        f"· capital base {estilo.dinero(cap)} · horizonte {cfg.horizonte_dias} días "
        f"· motor {cfg.optimization_engine}", s["meta"]))
    E.append(Paragraph(
        "Dado un conjunto de activos, ¿qué motor produce los pesos más robustos frente a cola y "
        "drawdown, y cuánto puedo perder razonablemente mañana / este mes bajo el régimen actual?", s["foco"]))

    # 1 · Contexto
    r = payload.regimen
    E.append(Paragraph("1 · Contexto de mercado", s["h2"]))
    E.append(_kpi_row([
        (r.etiqueta.replace("_", " ").title(), "Régimen", estilo.AMBAR),
        (estilo.pct(r.volatilidad_actual), "Vol táctica T+1", estilo.TINTA),
        (estilo.num(r.correlacion_media_actual), "Correlación media", estilo.TINTA),
        (f"{rec.motor_optimizacion}/{estilo.nombre_nivel(rec.nivel, idi)}", "Recomendada", estilo.VERDE),
    ]))
    E.append(Spacer(1, 4))
    E.append(Paragraph(r.descripcion, s["nota"]))

    # 2 · Tabla maestra
    E.append(Paragraph("2 · Tabla maestra de decisión — Champion vs Challenger", s["h2"]))
    if payload.frontera_degenerada:
        aviso = ParagraphStyle("aviso", parent=s["nota"], textColor=colors.HexColor("#7C2D12"),
                               backColor=colors.HexColor("#FFF7ED"), borderPadding=8, spaceAfter=8)
        E.append(Paragraph("⚠ " + payload.nota_frontera, aviso))
    cab = ["Motor", "Cartera", "Ret. geom.", "Vol T+1", "VaR99", "CVaR99", "CDaR", "R²", "K", "Score", "Decisión"]
    datos = [cab]
    win_idx = None
    for i, c in enumerate(payload.candidatos, start=1):
        if (c.motor_optimizacion, c.nivel) == (rec.motor_optimizacion, rec.nivel):
            win_idx = i
        datos.append([
            c.motor_optimizacion or "—", estilo.nombre_nivel(c.nivel, idi), estilo.pct(c.retorno_esperado),
            estilo.pct(c.volatilidad_tactica), estilo.pct(c.forecast.var_fhs_99, 2),
            estilo.pct(c.forecast.cvar_fhs_99, 2), estilo.pct(c.simulacion.cdar_30d, 1),
            estilo.num(c.r2_curva_capital, 2), estilo.num(c.k_ratio, 2),
            f"{c.score:.2f}",
            "RECOMENDADA" if (c.motor_optimizacion, c.nivel) == (rec.motor_optimizacion, rec.nivel) else "—",
        ])
    E.append(_tabla(
        datos,
        [1.9 * cm, 1.8 * cm, 1.7 * cm, 1.5 * cm, 1.5 * cm, 1.6 * cm,
         1.4 * cm, 1.0 * cm, 1.0 * cm, 1.2 * cm, 2.2 * cm],
        win_idx,
    ))
    E.append(Spacer(1, 4))
    E.append(Paragraph(
        f"{payload.recomendacion.detalle} Criterio: {payload.recomendacion.criterio}. "
        "R² es diagnóstico in-sample; Walk-Forward estricto queda en roadmap V2.",
        s["nota"],
    ))

    # 3 · Cartera recomendada + MCR
    E.append(Paragraph("3 · Cartera recomendada y descomposición del riesgo (MCR)", s["h2"]))
    img_pesos = Image(str(rutas["pesos"]), width=8.0 * cm, height=6.7 * cm)
    mcr_tab = [["Activo", "Peso", "Contrib. riesgo", "MCR"]]
    for a in rec.pesos.index:
        mcr_tab.append([a, estilo.pct(float(rec.pesos[a])),
                        estilo.pct(float(rec.descomposicion.contribucion_pct[a])),
                        estilo.num(float(rec.descomposicion.mcr[a]), 3)])
    dual = Table([[img_pesos, _tabla(mcr_tab, [2.0 * cm, 1.6 * cm, 2.6 * cm, 1.5 * cm])]],
                 colWidths=[8.4 * cm, 8.4 * cm])
    dual.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    E.append(dual)

    # 4 · Fan chart
    E.append(Paragraph(f"4 · ¿Cuánto puedo perder este mes? — simulación a {cfg.horizonte_dias} días", s["h2"]))
    E.append(Image(str(rutas["fan_chart"]), width=16.0 * cm, height=8.0 * cm))
    E.append(_kpi_row([
        (estilo.pct(sim.retorno_mediano), "Retorno mediano", estilo.TINTA),
        (estilo.pct(sim.perdida_p5), "Adverso P5", estilo.NEG),
        (f"{sim.prob_perdida:.0%}", "Prob. pérdida", estilo.TINTA),
        (estilo.pct(sim.cdar_30d), "CDaR cola", estilo.NEG),
    ]))

    # 5 · VaR mañana
    E.append(Paragraph("5 · ¿Cuánto puedo perder mañana? — VaR / CVaR T+1", s["h2"]))
    var_tab = [["Método", "VaR 95%", "€ 95%", "VaR 99%", "€ 99%"]]
    for nombre, v95, v99 in (("Histórico", f.var_hist_95, f.var_hist_99),
                             ("Paramétrico", f.var_param_95, f.var_param_99),
                             ("FHS", f.var_fhs_95, f.var_fhs_99)):
        var_tab.append([nombre, estilo.pct(v95, 2), f"{v95*cap:,.0f} €",
                        estilo.pct(v99, 2), f"{v99*cap:,.0f} €"])
    dual2 = Table([[Image(str(rutas["var_forecast"]), width=7.6 * cm, height=5.4 * cm),
                    _tabla(var_tab, [2.2 * cm, 1.8 * cm, 2.2 * cm, 1.8 * cm, 2.2 * cm])]],
                  colWidths=[8.0 * cm, 8.8 * cm])
    dual2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    E.append(dual2)
    E.append(Paragraph(
        "VaR/CVaR: estimaciones bajo los supuestos del modelo (no \"pérdida máxima\"). "
        f"Negativo = pérdida en la cola. Motor de simulación: {sim.fuente}.", s["nota"]))

    # 6 · Exploración multi-criterio
    if payload.leaderboard:
        E.append(Paragraph("6 · Exploración multi-criterio — todas las opciones, clasificadas", s["h2"]))
        E.append(Paragraph(
            "Frontera y nube clasificadas en bandas Bajo/Medio/Alto (anclas: mínima varianza, "
            "Máx Sharpe, máx retorno). Leaderboard Top-3 por criterio: cribado paramétrico, "
            "confirmado con FHS + Monte Carlo.", s["nota"]))
        if "frontera_clasificada" in rutas:
            E.append(Image(str(rutas["frontera_clasificada"]), width=15.0 * cm, height=8.7 * cm))
        for cr in payload.leaderboard:
            E.append(Paragraph(f"{cr.nombre} — {cr.descripcion}", s["h3"]))
            lt = [["#", "Clase", "Pesos", "Ret", "VolT+1", "Sharpe", "VaR99", "CDaR", "STARR", "Div", "Score"]]
            for i, c in enumerate(cr.top[:3], start=1):
                top = c.pesos[c.pesos > 0.005].sort_values(ascending=False).head(3)
                pesos_txt = " ".join(f"{a.split('.')[0]}{v*100:.0f}" for a, v in top.items())
                lt.append([str(i), estilo.nombre_nivel(c.clase_riesgo or "", idi), pesos_txt,
                           estilo.pct(c.retorno_esperado), estilo.pct(c.volatilidad_tactica),
                           estilo.num(c.sharpe), estilo.pct(c.forecast.var_fhs_99, 2),
                           estilo.pct(c.simulacion.cdar_30d, 1), estilo.num(c.starr),
                           estilo.num(c.diversificacion), estilo.num(c.score)])
            E.append(_tabla(lt, [0.7*cm, 1.5*cm, 3.0*cm, 1.5*cm, 1.5*cm, 1.4*cm, 1.5*cm, 1.5*cm, 1.3*cm, 1.1*cm, 1.3*cm]))

    # Apéndice
    E.append(Paragraph("Apéndice técnico", s["h2"]))
    E.append(Paragraph("A1 · Estadística individual — ¿qué activos tengo?", s["h3"]))
    est_tab = [["Activo", "Ret. medio", "Ret. ajust.", "Vol", "Vol T+1", "Asim.", "Curt."]]
    for e in payload.momentos.estadisticas:
        est_tab.append([e.ticker, estilo.pct(e.retorno_medio), estilo.pct(e.retorno_ajustado),
                        estilo.pct(e.volatilidad), estilo.pct(e.volatilidad_tactica),
                        estilo.num(e.asimetria), estilo.num(e.curtosis)])
    E.append(_tabla(est_tab, [2.6 * cm, 2.1 * cm, 2.1 * cm, 1.8 * cm, 1.9 * cm, 1.7 * cm, 1.7 * cm]))
    E.append(Paragraph("A2 · Frontera eficiente y candidatos", s["h3"]))
    E.append(Image(str(rutas["frontera"]), width=15.0 * cm, height=8.7 * cm))
    E.append(Paragraph("A3 · Correlación — ¿cómo se relacionan?", s["h3"]))
    E.append(Image(str(rutas["correlacion"]), width=9.5 * cm, height=8.5 * cm))

    doc.build(E)
    return ruta

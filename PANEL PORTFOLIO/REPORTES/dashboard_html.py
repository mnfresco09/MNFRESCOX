"""Dashboard ejecutivo HTML — informe institucional minimalista.

Estructura orientada a la DECISIÓN:
  1. Resumen ejecutivo     → régimen, score, cartera recomendada.
  2. Cartera recomendada   → pesos + descomposición del riesgo (MCR).
  3. Riesgo a horizonte    → fan chart + KPIs de simulación.
  4. VaR diario            → forecast + tabla de métodos.
  5. Comparativa           → tabla maestra simplificada.
  6. Detalle técnico       → frontera, estadística, correlación (collapsible).
  7. Glosario              → definiciones y metodología.

El reporting NO recalcula nada: solo consume ``ReportPayload``.
"""

from __future__ import annotations

import html as _html
import math as _math
from pathlib import Path

import plotly.offline as _pyo

from CONTRATOS.modelos import ReportPayload

from . import estilo, graficos_interactivos, narrativa

# ---------------------------------------------------------------------------
# Color por régimen
# ---------------------------------------------------------------------------
_REGIMEN_COLOR = {
    "baja_volatilidad": estilo.VERDE,
    "alta_volatilidad": estilo.AMBAR,
    "crisis": estilo.NEG,
}


# ---------------------------------------------------------------------------
# Helpers de renderizado
# ---------------------------------------------------------------------------

def _kpi(valor: str, titulo: str, sub: str = "", color: str = estilo.TINTA) -> str:
    """Tarjeta KPI: valor grande + etiqueta + sublínea."""
    return (
        f'<div class="kpi">'
        f'<div class="kpi-v" style="color:{color}">{_html.escape(valor)}</div>'
        f'<div class="kpi-t">{_html.escape(titulo)}</div>'
        f'<div class="kpi-s">{_html.escape(sub)}</div>'
        f'</div>'
    )


def _banner(payload: ReportPayload) -> str:
    """Fila de 4 KPIs para el resumen ejecutivo."""
    r = payload.regimen
    rec = payload.recomendada
    cfg = payload.configuracion
    idi = cfg.idioma_reporte
    color_reg = _REGIMEN_COLOR.get(r.etiqueta, estilo.TINTA)
    etiqueta = r.etiqueta.replace("_", " ").title()

    return (
        '<div class="kpis">'
        + _kpi(etiqueta, narrativa.t('kpi_regimen', idi),
               f"volatilidad en percentil {r.percentil_volatilidad:.0%}", color_reg)
        + _kpi(estilo.pct(r.volatilidad_actual), narrativa.t('kpi_vol_tactica', idi),
               "activo de referencia")
        + _kpi(estilo.nombre_nivel(rec.nivel, idi),
               narrativa.t('kpi_cartera_rec', idi),
               f"{rec.motor_optimizacion} · score {rec.score:.2f}", estilo.VERDE)
        + _kpi(f"{rec.score:.2f}", narrativa.t('kpi_score', idi),
               "mayor = mejor", estilo.ACENTO)
        + "</div>"
    )


def _num_seguro(x, dec: int = 2) -> str:
    """Formatea un número tolerando None / NaN / inf."""
    if x is None or (isinstance(x, float) and not _math.isfinite(x)):
        return "—"
    return f"{x:.{dec}f}"


def _pct_seguro(x, dec: int = 1) -> str:
    if x is None or (isinstance(x, float) and not _math.isfinite(x)):
        return "—"
    return estilo.pct(x, dec)


def _kpis_historicas(payload: ReportPayload) -> str:
    """Fila de KPIs realizados in-sample: MaxDD exacto, CAGR, Sharpe, Calmar."""
    mh = getattr(payload, "metricas_historicas", None)
    if mh is None:
        return ""
    idi = payload.configuracion.idioma_reporte
    sub_dd = ""
    if mh.fecha_valle_dd is not None:
        try:
            sub_dd = f"valle {mh.fecha_valle_dd.date().isoformat()}"
        except AttributeError:
            sub_dd = ""
    return (
        f"<h3>{narrativa.t('sub_metricas_hist', idi)}</h3>"
        '<div class="kpis">'
        + _kpi(_pct_seguro(mh.max_drawdown), narrativa.t('kpi_maxdd', idi), sub_dd, estilo.NEG)
        + _kpi(_pct_seguro(mh.cagr), narrativa.t('kpi_cagr', idi), "in-sample", estilo.VERDE)
        + _kpi(_num_seguro(mh.sharpe_historico), narrativa.t('kpi_sharpe_hist', idi), "exceso / vol")
        + _kpi(_num_seguro(mh.calmar), narrativa.t('kpi_calmar', idi), "retorno / caída", estilo.ACENTO)
        + "</div>"
        + f"<p class=\"footnote\">{_html.escape(narrativa.t('nota_metricas_hist', idi))}</p>"
    )


def _tabla_mcr_simple(payload: ReportPayload) -> str:
    """Tabla MCR simplificada: 3 columnas — Activo, Peso, Contrib. riesgo."""
    idi = payload.configuracion.idioma_reporte
    c = payload.recomendada
    activos = list(c.pesos.index)
    filas = []
    for a in activos:
        peso = float(c.pesos[a])
        if peso < 0.001:
            continue
        contrib = float(c.descomposicion.contribucion_pct[a])
        filas.append(
            f"<tr><td class='activo'>{_html.escape(a)}</td>"
            f"<td>{estilo.pct(peso)}</td>"
            f"<td>{estilo.pct(contrib)}</td></tr>"
        )
    return (
        '<div class="tabla-wrap"><table>'
        f'<thead><tr><th>{narrativa.t("col_activo", idi)}</th><th>{narrativa.t("col_peso", idi)}</th><th>{narrativa.t("col_contrib_riesgo", idi)}</th></tr></thead>'
        f"<tbody>{''.join(filas)}</tbody></table></div>"
    )


def _tabla_maestra_simple(payload: ReportPayload) -> str:
    """Tabla maestra simplificada: 7 columnas."""
    cfg = payload.configuracion
    idi = cfg.idioma_reporte
    ganadora = (payload.recomendada.motor_optimizacion, payload.recomendada.nivel)
    cabecera = (
        narrativa.t('col_motor', idi),
        narrativa.t('col_perfil', idi),
        narrativa.t('col_pesos', idi),
        narrativa.t('col_retorno', idi),
        narrativa.t('col_vol', idi),
        narrativa.t('col_var99', idi),
        narrativa.t('col_score', idi),
        narrativa.t('col_decision', idi),
    )
    th = "".join(f"<th>{_html.escape(c)}</th>" for c in cabecera)

    filas = []
    for c in payload.candidatos:
        es_win = (c.motor_optimizacion, c.nivel) == ganadora
        decision = f"✓ {narrativa.t('decision_recomendada', idi)}" if es_win else "—"
        cls = ' class="winner"' if es_win else ""
        filas.append(
            f"<tr{cls}>"
            f"<td class='activo'>{_html.escape(c.motor_optimizacion or '—')}</td>"
            f"<td>{_html.escape(estilo.nombre_nivel(c.nivel, idi))}</td>"
            f"<td class='activo' style='font-size:0.9em; color:{estilo.SUAVE}'>{_html.escape(_pesos_compactos(c.pesos, 4))}</td>"
            f"<td>{estilo.pct(c.retorno_esperado)}</td>"
            f"<td>{estilo.pct(c.volatilidad_tactica)}</td>"
            f"<td style='color:{estilo.NEG}'>{estilo.pct(c.forecast.var_fhs_99, 2)}</td>"
            f"<td><b>{c.score:.2f}</b></td>"
            f"<td>{decision}</td></tr>"
        )
    return (
        f'<div class="tabla-wrap"><table><thead><tr>{th}</tr></thead>'
        f"<tbody>{''.join(filas)}</tbody></table></div>"
    )


def _tabla_score_detallado(payload: ReportPayload) -> str:
    """Desglose auditable del Súper Score por candidato."""
    cfg = payload.configuracion
    idi = cfg.idioma_reporte
    cabecera = (
        "Motor", "Perfil", "Score", "Z Sharpe", "Z Sortino", "Z Ret. exceso",
        "Z K-Ratio", "Z Calmar", "Z CVaR", "Z CDaR", "Z MaxDD",
        "Z HHI riesgo", "Max RC", "Z Max RC", "Z Corr", "Reglas",
        "Pen. reglas", "Ret. exceso", "MaxDD", "HHI riesgo", "Corr ponderada",
    )
    th = "".join(f"<th>{_html.escape(c)}</th>" for c in cabecera)
    filas = []
    for c in payload.candidatos:
        d = dict(c.detalle_score)
        filas.append(
            "<tr>"
            f"<td class='activo'>{_html.escape(c.motor_optimizacion or '—')}</td>"
            f"<td>{_html.escape(estilo.nombre_nivel(c.nivel, idi))}</td>"
            f"<td><b>{estilo.num(c.score)}</b></td>"
            f"<td>{estilo.num(d.get('z_sharpe'))}</td>"
            f"<td>{estilo.num(d.get('z_sortino'))}</td>"
            f"<td>{estilo.num(d.get('z_retorno_exceso'))}</td>"
            f"<td>{estilo.num(d.get('z_k_ratio'))}</td>"
            f"<td>{estilo.num(d.get('z_calmar'))}</td>"
            f"<td>{estilo.num(d.get('z_cvar'))}</td>"
            f"<td>{estilo.num(d.get('z_cdar'))}</td>"
            f"<td>{estilo.num(d.get('z_max_drawdown'))}</td>"
            f"<td>{estilo.num(d.get('z_hhi_riesgo'))}</td>"
            f"<td>{estilo.pct(d.get('max_contribucion_riesgo'))}</td>"
            f"<td>{estilo.num(d.get('z_penalizacion_max_contrib_riesgo'))}</td>"
            f"<td>{estilo.num(d.get('z_correlacion'))}</td>"
            f"<td>{estilo.num(d.get('reglas_duras_incumplidas'), 0)}</td>"
            f"<td>{estilo.num(d.get('penalizacion_reglas_duras'), 0)}</td>"
            f"<td>{estilo.pct(d.get('retorno_exceso'))}</td>"
            f"<td>{estilo.pct(d.get('max_drawdown_abs'))}</td>"
            f"<td>{estilo.num(d.get('hhi_riesgo'))}</td>"
            f"<td>{estilo.num(d.get('correlacion_ponderada'))}</td>"
            "</tr>"
        )
    return (
        '<details class="nested-detail">'
        '<summary>Score breakdown</summary>'
        f'<div class="tabla-wrap"><table><thead><tr>{th}</tr></thead>'
        f"<tbody>{''.join(filas)}</tbody></table></div>"
        "</details>"
    )


def _tabla_var(payload: ReportPayload) -> str:
    """Tabla VaR por método: Histórico, Paramétrico, FHS."""
    idi = payload.configuracion.idioma_reporte
    f = payload.recomendada.forecast
    cap = payload.configuracion.capital_base

    def fila(nombre, v95, v99):
        return (
            f"<tr><td class='activo'>{_html.escape(nombre)}</td>"
            f"<td style='color:{estilo.NEG}'>{estilo.pct(v95, 2)}</td>"
            f"<td style='color:{estilo.NEG}'>{v95 * cap:,.0f} €</td>"
            f"<td style='color:{estilo.NEG}'>{estilo.pct(v99, 2)}</td>"
            f"<td style='color:{estilo.NEG}'>{v99 * cap:,.0f} €</td></tr>"
        )

    cuerpo = (
        fila(narrativa.t('metodo_historico', idi), f.var_hist_95, f.var_hist_99)
        + fila(narrativa.t('metodo_parametrico', idi), f.var_param_95, f.var_param_99)
        + fila("FHS (T+1)", f.var_fhs_95, f.var_fhs_99)
    )
    return (
        '<div class="tabla-wrap"><table>'
        f'<thead><tr><th>{narrativa.t("col_metodo", idi)}</th><th>VaR 95%</th><th>€ 95%</th>'
        f'<th>VaR 99%</th><th>€ 99%</th></tr></thead>'
        f"<tbody>{cuerpo}</tbody></table></div>"
    )


def _tabla_estadistica(payload: ReportPayload) -> str:
    """Tabla de estadística individual por activo (para apéndice)."""
    idi = payload.configuracion.idioma_reporte
    filas = []
    for e in payload.momentos.estadisticas:
        filas.append(
            f"<tr><td class='activo'>{_html.escape(e.ticker)}</td>"
            f"<td>{estilo.pct(e.retorno_medio)}</td>"
            f"<td>{estilo.pct(e.retorno_ajustado)}</td>"
            f"<td>{estilo.pct(e.volatilidad)}</td>"
            f"<td>{estilo.pct(e.volatilidad_tactica)}</td>"
            f"<td>{estilo.num(e.asimetria)}</td>"
            f"<td>{estilo.num(e.curtosis)}</td></tr>"
        )
    return (
        '<div class="tabla-wrap"><table>'
        f'<thead><tr><th>{narrativa.t("col_activo", idi)}</th><th>{narrativa.t("col_ret_medio", idi)}</th><th>{narrativa.t("col_ret_ajustado", idi)}</th>'
        f'<th>{narrativa.t("col_vol_label", idi)}</th><th>{narrativa.t("col_vol_tactica", idi)}</th><th>{narrativa.t("col_asimetria", idi)}</th><th>{narrativa.t("col_curtosis", idi)}</th></tr></thead>'
        f"<tbody>{''.join(filas)}</tbody></table></div>"
    )


def _pesos_compactos(pesos, n: int = 4) -> str:
    """Resumen compacto de los pesos más relevantes."""
    top = pesos[pesos > 0.005].sort_values(ascending=False).head(n)
    return " · ".join(f"{a.split('.')[0]} {v * 100:.0f}%" for a, v in top.items())


def _leaderboard_html(payload: ReportPayload) -> str:
    """Tablas de leaderboard por criterio (para apéndice técnico)."""
    cfg = payload.configuracion
    bloques = []
    for cr in payload.leaderboard:
        flecha = "▲ mayor mejor" if cr.sentido == "max" else "▼ menor mejor"
        filas = []
        for i, c in enumerate(cr.top, start=1):
            filas.append(
                f"<tr><td>{i}</td>"
                f"<td>{_html.escape(estilo.nombre_nivel(c.clase_riesgo or '', cfg.idioma_reporte))}</td>"
                f"<td class='activo'>{_html.escape(_pesos_compactos(c.pesos))}</td>"
                f"<td>{estilo.pct(c.retorno_esperado)}</td>"
                f"<td>{estilo.pct(c.volatilidad_tactica)}</td>"
                f"<td>{estilo.num(c.sharpe)}</td>"
                f"<td style='color:{estilo.NEG}'>{estilo.pct(c.forecast.var_fhs_99, 2)}</td>"
                f"<td style='color:{estilo.NEG}'>{estilo.pct(c.simulacion.cdar_30d, 1)}</td>"
                f"<td>{estilo.num(c.starr)}</td>"
                f"<td>{estilo.num(c.diversificacion)}</td>"
                f"<td><b>{estilo.num(c.score)}</b></td></tr>"
            )
        bloques.append(
            f"<h4>{_html.escape(cr.nombre)} "
            f"<span class='detail-note'>{_html.escape(cr.descripcion)} · {flecha}</span></h4>"
            '<div class="tabla-wrap"><table>'
            '<thead><tr><th>#</th><th>Clase</th><th>Pesos</th><th>Ret</th>'
            '<th>Vol T+1</th><th>Sharpe</th><th>VaR99 FHS</th><th>CDaR</th>'
            '<th>STARR</th><th>Div</th><th>Score</th></tr></thead>'
            f"<tbody>{''.join(filas)}</tbody></table></div>"
        )
    return "".join(bloques)


def _div(fig, div_id: str) -> str:
    """Embebe una figura Plotly interactiva (sin reincluir plotly.js)."""
    cuerpo = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={
            "displayModeBar": True,
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )
    return f'<div class="chart">{cuerpo}</div>'


def _figuras_interactivas(payload: ReportPayload, figuras=None) -> dict:
    """Devuelve figuras Plotly para HTML.

    `motor.generar_informes` comparte con el PDF un diccionario de PNG estáticos
    `(ruta, base64)`. El HTML necesita objetos Plotly con `to_html`; si recibe
    PNG, regenera únicamente las figuras interactivas sin tocar el cálculo.
    """
    if figuras and all(hasattr(fig, "to_html") for fig in figuras.values()):
        return figuras
    return graficos_interactivos.generar_figuras(payload)


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------

def generar_html(payload: ReportPayload, ruta: Path, figuras=None) -> Path:
    """Genera el informe HTML ejecutivo y lo escribe en *ruta*."""
    cfg = payload.configuracion
    idi = cfg.idioma_reporte
    figs = _figuras_interactivas(payload, figuras)
    plotlyjs = _pyo.get_plotlyjs()
    rec = payload.recomendada
    sim = rec.simulacion
    fuente = sim.fuente

    # -- Narrativas --
    txt_resumen = narrativa.texto_resumen_ejecutivo(payload)
    txt_por_que = narrativa.texto_por_que_gana(payload)
    txt_riesgo = narrativa.texto_conclusion_riesgo(payload)
    txt_var = narrativa.texto_conclusion_var(payload)
    txt_frontera = narrativa.texto_frontera_referencia(idi)
    txt_score = narrativa.texto_score(idi)
    glosario = narrativa.glosario(idi)

    # -- Glosario HTML --
    glosario_filas = "".join(
        f"<tr><td class='activo'>{_html.escape(t)}</td>"
        f"<td>{_html.escape(d)}</td></tr>"
        for t, d in glosario
    )
    glosario_html = (
        '<div class="tabla-wrap"><table>'
        f'<thead><tr><th>{narrativa.t("gloss_termino", idi)}</th><th>{narrativa.t("gloss_definicion", idi)}</th></tr></thead>'
        f"<tbody>{glosario_filas}</tbody></table></div>"
    )

    # -- Aviso frontera degenerada --
    aviso_frontera = ""
    if payload.frontera_degenerada:
        aviso_frontera = (
            '<div class="aviso">⚠ '
            + _html.escape(payload.nota_frontera)
            + "</div>"
        )

    # -- Frontera clasificada (opcional) --
    frontera_clas_html = ""
    if "frontera_clasificada" in figs:
        frontera_clas_html = _div(figs["frontera_clasificada"], "g_frontera_clas")

    # -- Leaderboard (opcional) --
    leaderboard_section = ""
    if payload.leaderboard:
        leaderboard_section = (
            '<details class="nested-detail">'
            f"<summary>{narrativa.t('sec_comparativa', idi)} — Leaderboard</summary>"
            f"{_leaderboard_html(payload)}"
            "</details>"
        )

    # -- Cartera recomendada: equity/drawdown + métricas + corr rolling (opcionales) --
    equity_dd_html = ""
    if "equity_drawdown" in figs:
        equity_dd_html = (
            f"<h3>{narrativa.t('sub_equity_dd', idi)}</h3>"
            f"{_div(figs['equity_drawdown'], 'g_equity_dd')}"
            f"<p class=\"footnote\">{_html.escape(narrativa.t('nota_equity_dd', idi))}</p>"
        )
    metricas_hist_html = _kpis_historicas(payload)
    corr_rolling_html = ""
    if "correlacion_rolling" in figs:
        corr_rolling_html = _div(figs["correlacion_rolling"], "g_corr_roll")

    # -- Activos listados --
    tickers_str = ", ".join(_html.escape(t) for t in cfg.tickers)

    doc = f"""<!DOCTYPE html>
<html lang="{idi}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PANEL PORTFOLIO — {narrativa.t('titulo_informe', idi)}</title>
<meta name="description" content="{narrativa.t('subtitulo_informe', idi)}">
<script>{plotlyjs}</script>
<style>{_CSS}</style>
</head>
<body>
<main>

  <!-- ===== HEADER / PORTADA ===== -->
  <header>
    <div class="eyebrow">{narrativa.t('eyebrow', idi)}</div>
    <h1>{narrativa.t('titulo_informe', idi)}</h1>
    <div class="meta">
      {len(cfg.tickers)} {narrativa.t('meta_activos', idi)} · {tickers_str}
      · {narrativa.t('meta_periodo', idi)} {cfg.fecha_inicio} → {cfg.fecha_fin}
      · {narrativa.t('meta_capital', idi)} {estilo.dinero(cfg.capital_base)}
      · {narrativa.t('meta_horizonte', idi)} {cfg.horizonte_dias} {narrativa.t('meta_dias', idi)}
      · {narrativa.t('meta_motor', idi)} {_html.escape(cfg.optimization_engine)}
    </div>
  </header>

  <!-- ===== SECCIÓN 1 — Resumen ejecutivo ===== -->
  <section>
    <h2>1 · {narrativa.t('sec_resumen', idi)}</h2>
    {_banner(payload)}
    <div class="narrative-box">
      <p class="narrative-bold">{_html.escape(txt_resumen)}</p>
    </div>
    <p class="narrative">{_html.escape(txt_por_que)}</p>
  </section>

  <!-- ===== SECCIÓN 2 — Cartera recomendada ===== -->
  <section>
    <h2>2 · {narrativa.t('sec_cartera', idi)}</h2>
    <div class="dos-col">
      <div>{_div(figs['pesos'], 'g_pesos')}</div>
      <div>{_tabla_mcr_simple(payload)}</div>
    </div>
    {_div(figs['mcr'], 'g_mcr')}
    {equity_dd_html}
    {metricas_hist_html}
    <h3>{narrativa.t('ap_correlacion', idi)}</h3>
    {_div(figs['correlacion'], 'g_corr')}
    {corr_rolling_html}
  </section>

  <!-- ===== SECCIÓN 3 — Riesgo a horizonte ===== -->
  <section>
    <h2>3 · {narrativa.t('sec_riesgo', idi, n=cfg.horizonte_dias)}</h2>
    {_div(figs['fan_chart'], 'g_fan')}
    <div class="kpis">
      {_kpi(estilo.pct(sim.retorno_mediano), narrativa.t('kpi_ret_mediano', idi), f'P50 a {cfg.horizonte_dias}d')}
      {_kpi(estilo.pct(sim.perdida_p5), narrativa.t('kpi_adverso_p5', idi), f'{sim.perdida_p5 * cfg.capital_base:,.0f} €', estilo.NEG)}
      {_kpi(f'{sim.prob_perdida:.0%}', narrativa.t('kpi_prob_perdida', idi), narrativa.t('kpi_a_horizonte', idi))}
      {_kpi(estilo.pct(sim.cdar_30d), narrativa.t('kpi_cdar', idi), 'media del peor 5%', estilo.NEG)}
    </div>
    <p class="narrative">{_html.escape(txt_riesgo)}</p>
  </section>

  <!-- ===== SECCIÓN 4 — VaR diario ===== -->
  <section>
    <h2>4 · {narrativa.t('sec_var', idi)}</h2>
    <div class="dos-col">
      <div>{_div(figs['var_forecast'], 'g_var')}</div>
      <div>{_tabla_var(payload)}</div>
    </div>
    <p class="narrative">{_html.escape(txt_var)}</p>
  </section>

  <!-- ===== SECCIÓN 5 — Comparativa de candidatos ===== -->
  <section>
    <h2>5 · {narrativa.t('sec_comparativa', idi)}</h2>
    {aviso_frontera}
    {_tabla_maestra_simple(payload)}
    {_tabla_score_detallado(payload)}
    <p class="narrative">{_html.escape(txt_score)}</p>
  </section>

  <!-- ===== SECCIÓN 6 — Detalle técnico (collapsible) ===== -->
  <section>
    <details class="tech-details">
      <summary>6 · {narrativa.t('sec_detalle', idi)}</summary>
      <div class="details-body">
        <h3>{narrativa.t('ap_frontera', idi)}</h3>
        <p class="footnote">{_html.escape(txt_frontera)}</p>
        {_div(figs['frontera'], 'g_frontera')}
        {frontera_clas_html}
        <div id="panel-pesos" class="panel-pesos">{narrativa.t("js_clic_frontera", idi)}</div>

        <h3>{narrativa.t('ap_estadistica', idi)}</h3>
        {_tabla_estadistica(payload)}

        {leaderboard_section}
      </div>
    </details>
  </section>

  <!-- ===== SECCIÓN 7 — Glosario ===== -->
  <section>
    <h2>{narrativa.t('sec_glosario', idi)}</h2>
    {glosario_html}
    <p class="footnote">{narrativa.t('footer_glosario', idi)}</p>
  </section>

  <!-- ===== FOOTER ===== -->
  <footer>
    {narrativa.t('footer', idi)}
  </footer>

</main>
<script>{_obtener_js_panel(idi)}</script>
</body>
</html>"""

    ruta.write_text(doc, encoding="utf-8")
    return ruta


# ---------------------------------------------------------------------------
# CSS — minimalista: blanco puro, semibold, bordes finos
# ---------------------------------------------------------------------------
_CSS = """
/* ── Reset & Base ──────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #111827;
  background: #FFFFFF;
  line-height: 1.55;
  font-weight: 600;
  -webkit-font-smoothing: antialiased;
}

main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 56px 64px 72px;
  background: #FFFFFF;
}

/* ── Header / Portada ──────────────────────────────────────── */
header {
  border-bottom: 1px solid #E5E7EB;
  padding-bottom: 28px;
  margin-bottom: 48px;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 3px;
  font-size: 11px;
  color: #1D4ED8;
  font-weight: 700;
  margin-bottom: 8px;
}

h1 {
  font-size: 32px;
  font-weight: 700;
  line-height: 1.1;
  color: #111827;
  margin-bottom: 12px;
}

.meta {
  color: #6B7280;
  font-size: 13px;
  font-weight: 400;
  line-height: 1.6;
}

/* ── Sections ──────────────────────────────────────────────── */
section {
  padding: 48px 0;
  border-bottom: 1px solid #E5E7EB;
}

section:last-of-type {
  border-bottom: none;
}

h2 {
  font-size: 20px;
  font-weight: 700;
  color: #111827;
  margin-bottom: 24px;
  letter-spacing: -0.01em;
}

h3 {
  font-size: 15px;
  font-weight: 700;
  color: #1D4ED8;
  margin: 32px 0 12px;
}

h4 {
  font-size: 14px;
  font-weight: 700;
  color: #111827;
  margin: 24px 0 8px;
}

/* ── KPI Cards ─────────────────────────────────────────────── */
.kpis {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin: 8px 0 24px;
}

.kpi {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 20px 22px;
}

.kpi-v {
  font-size: 26px;
  font-weight: 700;
  line-height: 1.1;
}

.kpi-t {
  font-size: 12px;
  color: #6B7280;
  font-weight: 600;
  margin-top: 6px;
  letter-spacing: 0.01em;
}

.kpi-s {
  font-size: 11px;
  color: #9CA3AF;
  font-weight: 400;
  margin-top: 2px;
}

/* ── Narrative Blocks ──────────────────────────────────────── */
.narrative-box {
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 20px 24px;
  margin: 16px 0;
  background: #FFFFFF;
}

.narrative-bold {
  font-size: 14px;
  font-weight: 700;
  color: #111827;
  line-height: 1.65;
  max-width: 90ch;
}

.narrative {
  font-size: 14px;
  font-weight: 600;
  color: #6B7280;
  line-height: 1.65;
  max-width: 90ch;
  margin-top: 12px;
}

/* ── Tables ────────────────────────────────────────────────── */
.tabla-wrap {
  overflow-x: auto;
  margin: 8px 0 16px;
}

table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
  font-weight: 600;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
}

thead th {
  background: #374151;
  color: #FFFFFF;
  font-weight: 600;
  padding: 10px 14px;
  text-align: right;
  white-space: nowrap;
  font-size: 12px;
  letter-spacing: 0.02em;
}

thead th:first-child {
  text-align: left;
}

td {
  padding: 9px 14px;
  text-align: right;
  border-bottom: 1px solid #E5E7EB;
  white-space: nowrap;
}

td.activo {
  text-align: left;
  font-weight: 700;
  color: #111827;
}

tr.winner td {
  background: #ECFDF3;
}

tr:last-child td {
  border-bottom: none;
}

/* ── Charts ────────────────────────────────────────────────── */
.chart {
  width: 100%;
  max-width: 100%;
  margin: 16px 0;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  background: #FFFFFF;
  padding: 4px;
}

/* ── Two Columns ───────────────────────────────────────────── */
.dos-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 28px;
  align-items: start;
}

/* ── Technical Details (Collapsible) ───────────────────────── */
.tech-details {
  border: none;
  margin: 0;
}

.tech-details > summary {
  font-size: 20px;
  font-weight: 700;
  color: #111827;
  cursor: pointer;
  padding: 4px 0;
  list-style: none;
  letter-spacing: -0.01em;
}

.tech-details > summary::-webkit-details-marker {
  display: none;
}

.tech-details > summary::before {
  content: "▶ ";
  font-size: 12px;
  color: #9CA3AF;
  margin-right: 6px;
  transition: transform 0.2s;
  display: inline-block;
}

.tech-details[open] > summary::before {
  content: "▼ ";
}

.details-body {
  padding-top: 16px;
}

.nested-detail {
  margin-top: 24px;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 16px 20px;
}

.nested-detail > summary {
  font-size: 14px;
  font-weight: 700;
  color: #1D4ED8;
  cursor: pointer;
}

.detail-note {
  font-size: 12px;
  color: #9CA3AF;
  font-weight: 400;
  font-style: italic;
  margin-left: 6px;
}

/* ── Panel Pesos (frontier interaction) ────────────────────── */
.panel-pesos {
  margin: 12px 0 24px;
  padding: 16px 20px;
  background: #FFFFFF;
  border: 1px dashed #E5E7EB;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 400;
  color: #9CA3AF;
  transition: all 0.2s ease;
}

.panel-pesos.activo {
  border-style: solid;
  border-color: #1D4ED8;
  color: #111827;
  font-weight: 600;
}

.panel-pesos .pp-titulo {
  font-size: 11px;
  color: #1D4ED8;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 8px;
}

.panel-pesos .pp-w {
  display: inline-block;
  margin: 3px 6px 3px 0;
  padding: 4px 12px;
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 600;
  color: #111827;
}

/* ── Warning / Aviso ───────────────────────────────────────── */
.aviso {
  background: #FFF7ED;
  border-left: 3px solid #B45309;
  padding: 12px 16px;
  font-size: 13px;
  font-weight: 600;
  color: #7C2D12;
  border-radius: 0 8px 8px 0;
  margin-bottom: 16px;
}

/* ── Footnotes ─────────────────────────────────────────────── */
.footnote {
  color: #9CA3AF;
  font-size: 12px;
  font-weight: 400;
  line-height: 1.6;
  margin-top: 12px;
  max-width: 90ch;
}

/* ── Footer ────────────────────────────────────────────────── */
footer {
  padding-top: 48px;
  font-size: 11px;
  font-weight: 400;
  color: #9CA3AF;
  letter-spacing: 0.02em;
}

/* ── Responsive ────────────────────────────────────────────── */
@media (max-width: 820px) {
  main { padding: 28px 20px; }
  .kpis { grid-template-columns: repeat(2, 1fr); }
  .dos-col { grid-template-columns: 1fr; }
}
"""


# ---------------------------------------------------------------------------
# JS — interacción con panel de pesos en la frontera
# ---------------------------------------------------------------------------
def _obtener_js_panel(idi: str) -> str:
    txt_sel = narrativa.t("js_cartera_seleccionada", idi)
    txt_ret = narrativa.t("js_retorno", idi)
    return f"""
(function(){{
  function pintar(p){{
    var panel = document.getElementById('panel-pesos');
    if (!panel || !p || p.customdata == null) return;
    var cd = p.customdata;
    var pesos = Array.isArray(cd) ? cd[0] : cd;
    if (pesos == null) return;
    var chips = String(pesos).split(' \\u00b7 ').map(function(s){{
      return '<span class="pp-w">' + s + '</span>';
    }}).join('');
    panel.innerHTML = '<div class="pp-titulo">{txt_sel} \\u00b7 vol '
      + p.x.toFixed(2) + '% \\u00b7 {txt_ret} ' + p.y.toFixed(2) + '%</div>'
      + '<div>' + chips + '</div>';
    panel.classList.add('activo');
  }}
  function bind(id){{
    var gd = document.getElementById(id);
    if (!gd || !gd.on) {{ setTimeout(function(){{ bind(id); }}, 300); return; }}
    gd.on('plotly_click', function(d){{ if(d.points && d.points.length) pintar(d.points[0]); }});
    gd.on('plotly_hover', function(d){{ if(d.points && d.points.length) pintar(d.points[0]); }});
  }}
  window.addEventListener('load', function(){{
    bind('g_frontera_clas');
    bind('g_frontera');
  }});
}})();
"""

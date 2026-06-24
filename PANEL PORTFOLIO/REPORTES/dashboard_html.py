"""Dashboard ejecutivo HTML — primera página orientada a la DECISIÓN.

Estructura (cada bloque responde una de las 4 preguntas):
  1. Banner de contexto  → régimen, volatilidad táctica, correlación.
  2. Tabla maestra        → Champion vs Challenger por motor.
  3. Cartera recomendada  → pesos + descomposición del riesgo (MCR).
  4. Fan chart            → cuánto puedo perder este mes.
  5. VaR forecast         → cuánto puedo perder mañana.
  Apéndice                → frontera, correlación, estadística individual.

El reporting NO recalcula nada: solo consume `ReportPayload`.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import plotly.offline as _pyo

from CONTRATOS.modelos import ReportPayload

from . import estilo, graficos_interactivos

_PREGUNTA_FOCO = (
    "Dado un conjunto de activos, ¿qué motor produce los pesos más robustos "
    "frente a cola y drawdown, y cuánto puedo perder razonablemente mañana / este mes "
    "bajo el régimen actual?"
)

_REGIMEN_COLOR = {
    "baja_volatilidad": estilo.VERDE,
    "alta_volatilidad": estilo.AMBAR,
    "crisis": estilo.NEG,
}


def _kpi(valor: str, titulo: str, sub: str = "", color: str = estilo.TINTA) -> str:
    return (f'<div class="kpi"><div class="kpi-v" style="color:{color}">{_html.escape(valor)}</div>'
            f'<div class="kpi-t">{_html.escape(titulo)}</div>'
            f'<div class="kpi-s">{_html.escape(sub)}</div></div>')


def _banner(payload: ReportPayload) -> str:
    r = payload.regimen
    color = _REGIMEN_COLOR.get(r.etiqueta, estilo.TINTA)
    etiqueta = r.etiqueta.replace("_", " ").title()
    return (
        '<div class="kpis">'
        + _kpi(etiqueta, "Régimen de mercado", f"vol en percentil {r.percentil_volatilidad:.0%}", color)
        + _kpi(estilo.pct(r.volatilidad_actual), "Volatilidad táctica (T+1)", "activo de referencia")
        + _kpi(estilo.num(r.correlacion_media_actual), "Correlación media reciente", "diversificación efectiva")
        + _kpi(estilo.nombre_nivel(payload.recomendada.nivel, payload.configuracion.idioma_reporte),
               "Cartera recomendada",
               f"{payload.recomendada.motor_optimizacion} · score {payload.recomendada.score:.2f}",
               estilo.VERDE)
        + "</div>"
    )


def _tabla_maestra(payload: ReportPayload) -> str:
    cfg = payload.configuracion
    filas = []
    cabecera = ("Motor", "Cartera", "Retorno geom.", "Vol T+1", "VaR 99% FHS",
                "CVaR 99% FHS", "CDaR 30d", "R²", "K", "Score", "Decisión")
    th = "".join(f"<th>{_html.escape(c)}</th>" for c in cabecera)
    ganadora = (payload.recomendada.motor_optimizacion, payload.recomendada.nivel)
    for c in payload.candidatos:
        es_win = (c.motor_optimizacion, c.nivel) == ganadora
        decision = "★ RECOMENDADA" if es_win else "—"
        cls = ' class="mejor"' if es_win else ""
        filas.append(
            f"<tr{cls}><th class='metodo'>{_html.escape(c.motor_optimizacion or '—')}</th>"
            f"<td>{_html.escape(estilo.nombre_nivel(c.nivel, cfg.idioma_reporte))}</td>"
            f"<td>{estilo.pct(c.retorno_esperado)}</td>"
            f"<td>{estilo.pct(c.volatilidad_tactica)}</td>"
            f"<td style='color:{estilo.NEG}'>{estilo.pct(c.forecast.var_fhs_99, 2)}</td>"
            f"<td style='color:{estilo.NEG}'>{estilo.pct(c.forecast.cvar_fhs_99, 2)}</td>"
            f"<td style='color:{estilo.NEG}'>{estilo.pct(c.simulacion.cdar_30d, 1)}</td>"
            f"<td>{estilo.num(c.r2_curva_capital, 2)}</td>"
            f"<td>{estilo.num(c.k_ratio, 2)}</td>"
            f"<td><b>{c.score:.2f}</b></td>"
            f"<td>{decision}</td></tr>"
        )
    return (f'<div class="tabla-scroll"><table class="maestra"><thead><tr>{th}</tr></thead>'
            f"<tbody>{''.join(filas)}</tbody></table></div>")


def _tabla_mcr(payload: ReportPayload) -> str:
    c = payload.recomendada
    activos = list(c.pesos.index)
    filas = []
    for a in activos:
        filas.append(
            f"<tr><th class='metodo'>{_html.escape(a)}</th>"
            f"<td>{estilo.pct(float(c.pesos[a]))}</td>"
            f"<td>{estilo.pct(float(c.descomposicion.contribucion_pct[a]))}</td>"
            f"<td>{estilo.num(float(c.descomposicion.mcr[a]), 3)}</td></tr>"
        )
    return ('<div class="tabla-scroll"><table class="maestra"><thead><tr>'
            "<th>Activo</th><th>Peso</th><th>Contribución al riesgo</th><th>MCR</th>"
            f"</tr></thead><tbody>{''.join(filas)}</tbody></table></div>")


def _tabla_var(payload: ReportPayload) -> str:
    f = payload.recomendada.forecast
    cap = payload.configuracion.capital_base

    def fila(nombre, v95, v99):
        return (f"<tr><th class='metodo'>{_html.escape(nombre)}</th>"
                f"<td style='color:{estilo.NEG}'>{estilo.pct(v95, 2)}</td>"
                f"<td style='color:{estilo.NEG}'>{v95*cap:,.0f} €</td>"
                f"<td style='color:{estilo.NEG}'>{estilo.pct(v99, 2)}</td>"
                f"<td style='color:{estilo.NEG}'>{v99*cap:,.0f} €</td></tr>")

    cuerpo = (fila("Histórico", f.var_hist_95, f.var_hist_99)
              + fila("Paramétrico (T+1)", f.var_param_95, f.var_param_99)
              + fila("FHS (T+1)", f.var_fhs_95, f.var_fhs_99))
    return ('<div class="tabla-scroll"><table class="maestra"><thead><tr>'
            "<th>Método</th><th>VaR 95%</th><th>€ 95%</th><th>VaR 99%</th><th>€ 99%</th>"
            f"</tr></thead><tbody>{cuerpo}</tbody></table></div>")


def _tabla_estadistica(payload: ReportPayload) -> str:
    filas = []
    for e in payload.momentos.estadisticas:
        filas.append(
            f"<tr><th class='metodo'>{_html.escape(e.ticker)}</th>"
            f"<td>{estilo.pct(e.retorno_medio)}</td><td>{estilo.pct(e.retorno_ajustado)}</td>"
            f"<td>{estilo.pct(e.volatilidad)}</td><td>{estilo.pct(e.volatilidad_tactica)}</td>"
            f"<td>{estilo.num(e.asimetria)}</td><td>{estilo.num(e.curtosis)}</td></tr>"
        )
    return ('<div class="tabla-scroll"><table class="maestra"><thead><tr>'
            "<th>Activo</th><th>Ret. medio</th><th>Ret. ajustado</th><th>Vol</th>"
            "<th>Vol T+1</th><th>Asimetría</th><th>Curtosis</th>"
            f"</tr></thead><tbody>{''.join(filas)}</tbody></table></div>")


def _pesos_compactos(pesos, n: int = 4) -> str:
    top = pesos[pesos > 0.005].sort_values(ascending=False).head(n)
    return " · ".join(f"{a.split('.')[0]} {v*100:.0f}%" for a, v in top.items())


def _leaderboard_html(payload: ReportPayload) -> str:
    bloques = []
    for cr in payload.leaderboard:
        flecha = "▲ mayor mejor" if cr.sentido == "max" else "▼ menor mejor"
        filas = []
        for i, c in enumerate(cr.top, start=1):
            filas.append(
                f"<tr><td>{i}</td>"
                f"<td><span class='chip {c.clase_riesgo}'>{_html.escape(estilo.nombre_nivel(c.clase_riesgo or '', payload.configuracion.idioma_reporte))}</span></td>"
                f"<td style='text-align:left'>{_html.escape(_pesos_compactos(c.pesos))}</td>"
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
            f"<h3>{_html.escape(cr.nombre)} <span class='q'>{_html.escape(cr.descripcion)} · {flecha}</span></h3>"
            '<div class="tabla-scroll"><table class="maestra"><thead><tr>'
            "<th>#</th><th>Clase</th><th>Pesos</th><th>Ret</th><th>Vol T+1</th><th>Sharpe</th>"
            "<th>VaR99 FHS</th><th>CDaR</th><th>STARR</th><th>Div</th><th>Score</th>"
            f"</tr></thead><tbody>{''.join(filas)}</tbody></table></div>"
        )
    return "".join(bloques)


def _div(fig, div_id: str) -> str:
    """Embebe una figura Plotly interactiva (sin reincluir plotly.js)."""
    cuerpo = fig.to_html(
        include_plotlyjs=False, full_html=False, div_id=div_id,
        config={"displayModeBar": True, "responsive": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    return f'<div class="grafico">{cuerpo}</div>'


def generar_html(payload: ReportPayload, ruta: Path, figuras=None) -> Path:
    cfg = payload.configuracion
    figs = graficos_interactivos.generar_figuras(payload)
    plotlyjs = _pyo.get_plotlyjs()
    rec = payload.recomendada
    fuente = rec.simulacion.fuente
    sim = rec.simulacion

    doc = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PANEL PORTFOLIO — Motor de Riesgo Predictivo</title>
<script>{plotlyjs}</script>
<style>{_CSS}</style></head><body>
<main>
  <header>
    <div class="eyebrow">Motor de Riesgo Predictivo · Buy-Side</div>
    <h1>Decisión de cartera y riesgo prospectivo</h1>
    <div class="meta">{len(cfg.tickers)} activos · {', '.join(_html.escape(t) for t in cfg.tickers)}
      · {cfg.fecha_inicio} → {cfg.fecha_fin} · capital base {estilo.dinero(cfg.capital_base)}
      · horizonte {cfg.horizonte_dias} días · motor {cfg.optimization_engine}</div>
    <div class="foco">{_html.escape(_PREGUNTA_FOCO)}</div>
  </header>

  <section>
    <h2>1 · Contexto de mercado</h2>
    {_banner(payload)}
    <p class="nota">{_html.escape(payload.regimen.descripcion)}</p>
  </section>

  <section>
    <h2>2 · Tabla maestra de decisión <span class="q">Champion vs Challenger</span></h2>
    {('<div class="aviso">⚠ ' + _html.escape(payload.nota_frontera) + '</div>') if payload.frontera_degenerada else ''}
    {_tabla_maestra(payload)}
    <p class="nota">{_html.escape(payload.recomendacion.detalle)} Criterio: {_html.escape(payload.recomendacion.criterio)}.
      El R² de la curva de capital es diagnóstico in-sample y no determina por sí solo la recomendación.
      Walk-Forward estricto queda en roadmap V2.</p>
  </section>

  <section>
    <h2>3 · Cartera recomendada y descomposición del riesgo</h2>
    <div class="dos-col">
      <div>{_div(figs['pesos'], 'g_pesos')}</div>
      <div>{_tabla_mcr(payload)}</div>
    </div>
    {_div(figs['mcr'], 'g_mcr')}
  </section>

  <section>
    <h2>4 · ¿Cuánto puedo perder este mes? <span class="q">simulación a {cfg.horizonte_dias} días</span></h2>
    {_div(figs['fan_chart'], 'g_fan')}
    <div class="kpis tres">
      {_kpi(estilo.pct(sim.retorno_mediano), 'Retorno mediano (horizonte)', f'P50 a {cfg.horizonte_dias}d')}
      {_kpi(estilo.pct(sim.perdida_p5), 'Escenario adverso (P5)', f'{sim.perdida_p5*cfg.capital_base:,.0f} €', estilo.NEG)}
      {_kpi(f'{sim.prob_perdida:.0%}', 'Probabilidad de pérdida', 'a horizonte')}
      {_kpi(estilo.pct(sim.cdar_30d), 'CDaR (drawdown de cola)', 'media del peor 5%', estilo.NEG)}
    </div>
  </section>

  <section>
    <h2>5 · ¿Cuánto puedo perder mañana? <span class="q">VaR / CVaR T+1</span></h2>
    <div class="dos-col">
      <div>{_div(figs['var_forecast'], 'g_var')}</div>
      <div>{_tabla_var(payload)}</div>
    </div>
    <p class="nota">Cifras de VaR/CVaR: estimaciones bajo los supuestos del modelo (no "pérdida máxima").
      Convención de signo: negativo = pérdida en la cola. Motor de simulación: <b>{_html.escape(fuente)}</b>.</p>
  </section>

  <section>
    <h2>6 · Exploración multi-criterio <span class="q">todas las opciones, clasificadas</span></h2>
    <p class="nota">Se recorre la frontera completa y la nube factible, clasificadas automáticamente
      en bandas Bajo / Medio / Alto (anclas: mínima varianza, Máx Sharpe, máx retorno). El leaderboard
      muestra las 5 mejores carteras por cada criterio: cribado paramétrico y confirmado con FHS + Monte Carlo.</p>
    {_div(figs['frontera_clasificada'], 'g_frontera_clas') if 'frontera_clasificada' in figs else ''}
    <div id="panel-pesos" class="panel-pesos">📍 Pasa el ratón por un punto de la frontera para ver sus pesos, o haz <b>clic</b> para fijarlos aquí.</div>
    {_leaderboard_html(payload)}
  </section>

  <section class="apendice">
    <h2>Apéndice técnico</h2>
    <h3>A1 · Estadística individual <span class="q">¿qué activos tengo?</span></h3>
    {_tabla_estadistica(payload)}
    <h3>A2 · Frontera eficiente y candidatos</h3>
    {_div(figs['frontera'], 'g_frontera')}
    <h3>A3 · Correlación <span class="q">¿cómo se relacionan?</span></h3>
    {_div(figs['correlacion'], 'g_corr')}
  </section>

  <footer>PANEL PORTFOLIO · Ledoit-Wolf (estructural) + EWMA (táctica) · Frontera restringida ·
    FHS &amp; Monte Carlo ({_html.escape(fuente)}) · Score multifactor. Documento informativo, no es asesoramiento de inversión.</footer>
</main>
<script>{_JS_PANEL}</script>
</body></html>"""
    ruta.write_text(doc, encoding="utf-8")
    return ruta


_CSS = """
:root{--tinta:#0F172A;--suave:#475569;--linea:#D9DEE7;--acento:#1D4ED8;--panel:#F6F8FB;--neg:#B91C1C;--verde:#15803D;}
*{box-sizing:border-box;}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--tinta);background:#EEF2F7;line-height:1.55;}
main{max-width:1080px;margin:0 auto;background:#fff;padding:48px 60px 70px;box-shadow:0 1px 40px rgba(15,23,42,.08);}
.eyebrow{text-transform:uppercase;letter-spacing:3px;font-size:11px;color:var(--acento);font-weight:700;}
h1{font-size:32px;margin:6px 0 10px;line-height:1.12;}
.meta{color:var(--suave);font-size:13px;margin-bottom:14px;}
.foco{background:var(--panel);border-left:4px solid var(--acento);padding:13px 18px;font-size:14px;color:var(--tinta);border-radius:0 8px 8px 0;font-style:italic;}
header{border-bottom:3px solid var(--tinta);padding-bottom:22px;margin-bottom:8px;}
section{padding:30px 0;border-bottom:1px solid var(--linea);}
section.apendice{background:#FAFBFD;margin:24px -60px -70px;padding:30px 60px 60px;}
h2{font-size:21px;margin:0 0 16px;}
h2 .q{font-size:13px;color:var(--suave);font-weight:400;font-style:italic;margin-left:8px;}
h3{font-size:15px;color:var(--acento);margin:26px 0 10px;}
h3 .q{font-size:12px;color:var(--suave);font-weight:400;font-style:italic;margin-left:6px;}
p{font-size:14px;max-width:88ch;}
.nota{color:var(--suave);font-size:13px;background:var(--panel);padding:11px 16px;border-radius:8px;}
.aviso{background:#FFF7ED;border-left:4px solid #B45309;padding:12px 16px;font-size:13px;color:#7C2D12;border-radius:0 8px 8px 0;margin-bottom:12px;}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 16px;}
.kpis.tres{grid-template-columns:repeat(4,1fr);margin-top:18px;}
.kpi{background:var(--panel);border:1px solid var(--linea);border-radius:12px;padding:16px 18px;}
.kpi-v{font-size:24px;font-weight:700;}
.kpi-t{font-size:12.5px;color:var(--suave);margin-top:4px;font-weight:600;}
.kpi-s{font-size:11px;color:#8A99AD;margin-top:2px;}
.tabla-scroll{overflow-x:auto;margin:6px 0;border:1px solid var(--linea);border-radius:10px;}
table.maestra{border-collapse:collapse;width:100%;font-size:13px;}
table.maestra th,table.maestra td{padding:10px 13px;text-align:right;border-bottom:1px solid var(--linea);white-space:nowrap;}
table.maestra thead th{background:var(--tinta);color:#fff;font-weight:600;}
table.maestra th.metodo{text-align:left;background:var(--panel);font-weight:700;}
table.maestra tr.mejor td,table.maestra tr.mejor th.metodo{background:#ECFDF3;}
.grafico{width:100%;max-width:100%;height:auto;margin:14px 0;border:1px solid var(--linea);border-radius:10px;background:#fff;}
.chip{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;color:#fff;}
.chip.bajo{background:#0EA5E9;}.chip.medio{background:#1D4ED8;}.chip.alto{background:#7C3AED;}
.panel-pesos{margin:6px 0 18px;padding:14px 18px;background:var(--panel);border:1px dashed var(--linea);border-radius:10px;font-size:13.5px;color:var(--suave);transition:.2s;}
.panel-pesos.activo{border-style:solid;border-color:var(--acento);background:#EFF5FF;color:var(--tinta);}
.panel-pesos .pp-titulo{font-size:12px;color:var(--acento);font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:7px;}
.panel-pesos .pp-w{display:inline-block;margin:3px 8px 3px 0;padding:4px 12px;background:#fff;border:1px solid var(--acento);border-radius:20px;font-size:13.5px;font-weight:600;color:var(--tinta);}
.dos-col{display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:center;}
footer{padding-top:26px;font-size:11.5px;color:#8A99AD;}
@media(max-width:820px){main{padding:24px;}.kpis,.kpis.tres{grid-template-columns:repeat(2,1fr);}.dos-col{grid-template-columns:1fr;}section.apendice{margin:24px -24px -70px;padding:24px;}}
"""


_JS_PANEL = """
(function(){
  function pintar(p){
    var panel=document.getElementById('panel-pesos');
    if(!panel||!p||p.customdata==null) return;
    var cd=p.customdata;
    var pesos=Array.isArray(cd)?cd[0]:cd;
    if(pesos==null) return;
    var chips=String(pesos).split(' \\u00b7 ').map(function(s){return '<span class=\"pp-w\">'+s+'</span>';}).join('');
    panel.innerHTML='<div class=\"pp-titulo\">Cartera seleccionada \\u00b7 vol '+p.x.toFixed(2)+'% \\u00b7 retorno '+p.y.toFixed(2)+'%</div><div>'+chips+'</div>';
    panel.classList.add('activo');
  }
  function bind(id){
    var gd=document.getElementById(id);
    if(!gd||!gd.on){ setTimeout(function(){bind(id);},300); return; }
    gd.on('plotly_click', function(d){ if(d.points&&d.points.length) pintar(d.points[0]); });
    gd.on('plotly_hover', function(d){ if(d.points&&d.points.length) pintar(d.points[0]); });
  }
  window.addEventListener('load', function(){ bind('g_frontera_clas'); bind('g_frontera'); });
})();
"""

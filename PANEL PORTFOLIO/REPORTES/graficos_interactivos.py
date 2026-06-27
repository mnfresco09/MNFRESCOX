"""Gráficos INTERACTIVOS para el HTML (Plotly): hover, zoom, pan, leyenda clicable.

Misma estética institucional que los PNG del PDF, pero explorables. Se embeben
con plotly.js inline (offline, sin CDN). El PDF sigue usando los PNG estáticos.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from CONTRATOS.modelos import ReportPayload

from . import estilo, narrativa

_FUENTE = "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def _layout(fig: go.Figure, titulo: str, x: str, y: str, alto: int = 430) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color=estilo.TINTA)),
        xaxis_title=x, yaxis_title=y, height=alto,
        font=dict(family=_FUENTE, size=12, color=estilo.SUAVE),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=24, t=44, b=48),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=estilo.LINEA, zeroline=False)
    fig.update_yaxes(gridcolor=estilo.LINEA, zeroline=False)
    return fig


def fan_chart(payload: ReportPayload) -> go.Figure:
    sim = payload.recomendada.simulacion
    cap = payload.configuracion.capital_base
    s = sim.sendas_percentil * cap
    dias = s.index.to_numpy()
    fig = go.Figure()
    if "p95" in s and "p5" in s:
        fig.add_trace(go.Scatter(x=dias, y=s["p95"], line=dict(width=0), showlegend=False,
                                 hoverinfo="skip", name="P95"))
        fig.add_trace(go.Scatter(x=dias, y=s["p5"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(147,180,255,0.30)", name="P5–P95 (90%)",
                                 hovertemplate="P5: €%{y:,.0f}<extra></extra>"))
    if "p75" in s and "p25" in s:
        fig.add_trace(go.Scatter(x=dias, y=s["p75"], line=dict(width=0), showlegend=False,
                                 hoverinfo="skip", name="P75"))
        fig.add_trace(go.Scatter(x=dias, y=s["p25"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(29,78,216,0.28)", name="P25–P75 (50%)",
                                 hovertemplate="P25: €%{y:,.0f}<extra></extra>"))
    if "p50" in s:
        fig.add_trace(go.Scatter(x=dias, y=s["p50"], line=dict(color=estilo.TINTA, width=2.6),
                                 name="Mediana (P50)", hovertemplate="Mediana: €%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=cap, line=dict(color=estilo.SUAVE, width=1, dash="dash"))
    titulo = (f"Proyección del capital a {sim.horizonte_dias} días — "
              f"{estilo.nombre_nivel(payload.recomendada.nivel, payload.configuracion.idioma_reporte)}")
    return _layout(fig, titulo, "Días bursátiles", "Capital estimado (€)", 460)


def equity_drawdown(payload: ReportPayload) -> go.Figure:
    """Equity y drawdown histórico (in-sample) de la cartera SELECCIONADA.

    Reconstruye la curva de capital aplicando los pesos recomendados (rebalanceo
    diario a peso fijo) a los log-retornos alineados desde la fecha de inicio, y
    su drawdown asociado. Estética de la imagen de referencia pero fondo blanco:
    equity en verde y drawdown en rojo, ambos como líneas SIN relleno, sobre el
    mismo eje temporal (drawdown en eje secundario para que sea legible).

    Nota: es desempeño in-sample con pesos fijos, no una promesa de retorno.
    """
    cfg = payload.configuracion
    cap = cfg.capital_base
    lr = payload.entrada.log_retornos
    pesos_vec = payload.recomendada.pesos.reindex(lr.columns).fillna(0.0).to_numpy()

    # Retorno simple diario por activo → cartera rebalanceada a peso fijo.
    simples = np.expm1(lr.to_numpy())
    port = simples @ pesos_vec
    equity = cap * np.cumprod(1.0 + port)

    fechas = lr.index.to_numpy()
    # Sembrar el punto inicial (capital base en la fecha de inicio).
    if len(fechas) > 1:
        f0 = fechas[0] - (fechas[1] - fechas[0])
    else:
        f0 = fechas[0]
    fechas_e = np.concatenate([[f0], fechas])
    equity = np.concatenate([[cap], equity])
    pico = np.maximum.accumulate(equity)
    drawdown = (equity / pico - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fechas_e, y=equity, mode="lines", name="Equity (€)",
        line=dict(color=estilo.VERDE, width=2.0), yaxis="y",
        hovertemplate="%{x|%d-%m-%Y}<br>Equity: €%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=fechas_e, y=drawdown, mode="lines", name="Drawdown (%)",
        line=dict(color=estilo.NEG, width=1.4), yaxis="y2",
        hovertemplate="%{x|%d-%m-%Y}<br>Drawdown: %{y:.1f}%<extra></extra>"))

    # Rangos enfrentados: el baseline de cada serie en el CENTRO vertical, para
    # que equity crezca hacia arriba (mitad superior) y drawdown hacia abajo
    # (mitad inferior) sin pisarse. Equity: su mínimo cae en el centro; drawdown:
    # el 0% cae en el centro.
    e_min, e_max = float(equity.min()), float(equity.max())
    pad_e = (e_max - e_min) * 0.05 or max(abs(e_max) * 0.05, 1.0)
    e_top = e_max + pad_e
    e_lo = 2.0 * e_min - e_top                       # e_min queda en el 50%
    d_min = float(drawdown.min())
    pad_d = abs(d_min) * 0.05 or 1.0
    d_bot = d_min - pad_d                             # 0% queda en el 50%

    nivel = estilo.nombre_nivel(payload.recomendada.nivel, cfg.idioma_reporte)
    titulo = f"Equity y drawdown histórico — cartera {nivel} (in-sample, peso fijo)"
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, color=estilo.TINTA)),
        height=470,
        font=dict(family=_FUENTE, size=12, color=estilo.SUAVE),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=66, r=66, t=44, b=48),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Fecha", gridcolor=estilo.LINEA, zeroline=False),
        yaxis=dict(title="Equity (€)", gridcolor=estilo.LINEA, zeroline=False,
                   tickformat=",.0f", side="left", range=[e_lo, e_top],
                   title_font=dict(color=estilo.VERDE)),
        yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right",
                    showgrid=False, zeroline=True, zerolinecolor=estilo.LINEA,
                    range=[d_bot, -d_bot], title_font=dict(color=estilo.NEG)),
    )
    return fig


def pesos(payload: ReportPayload) -> go.Figure:
    p = payload.recomendada.pesos
    activos = list(p.index)
    vals = p.to_numpy() * 100
    colores = [estilo.SERIE[i % len(estilo.SERIE)] for i in range(len(activos))]
    fig = go.Figure(go.Bar(
        x=vals, y=activos, orientation="h", marker_color=colores,
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _layout(fig, "Pesos de la cartera recomendada", "Peso (%)", "", 360)


def mcr(payload: ReportPayload) -> go.Figure:
    c = payload.recomendada
    activos = list(c.pesos.index)
    peso = c.pesos.reindex(activos).to_numpy() * 100
    ctr = c.descomposicion.contribucion_pct.reindex(activos).to_numpy() * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=activos, y=peso, name="Peso (%)", marker_color=estilo.ACENTO_CLARO,
                         hovertemplate="%{x} peso: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=activos, y=ctr, name="Contribución al riesgo (%)", marker_color=estilo.NEG,
                         hovertemplate="%{x} riesgo: %{y:.1f}%<extra></extra>"))
    fig.update_layout(barmode="group")
    f = _layout(fig, "Descomposición del riesgo (MCR) vs peso", "", "% del total", 400)
    f.update_layout(hovermode="closest")
    return f


def var_forecast(payload: ReportPayload) -> go.Figure:
    f = payload.recomendada.forecast
    cap = payload.configuracion.capital_base
    metodos = ["Histórico", "Paramétrico", "FHS"]
    v99 = np.array([f.var_hist_99, f.var_param_99, f.var_fhs_99]) * 100
    fig = go.Figure(go.Bar(
        x=metodos, y=v99, marker_color=[estilo.SUAVE, estilo.AMBAR, estilo.NEG],
        text=[f"{v:.2f}%<br>{v/100*cap:,.0f}€" for v in v99], textposition="inside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
    fig2 = _layout(fig, "VaR 99% diario estimado (recomendada)", "", "Retorno en la cola (%)", 380)
    fig2.update_layout(hovermode="closest")
    return fig2


def frontera(payload: ReportPayload) -> go.Figure:
    fr = payload.frontera
    nube = fr.nube_factible
    puntos = fr.puntos
    fig = go.Figure()
    activos = [c[len("peso·"):] for c in puntos.columns if c.startswith("peso·")]
    pesos_nube = [_pesos_texto(f, activos) for _, f in nube.iterrows()]
    fig.add_trace(go.Scattergl(
        x=nube["volatilidad"] * 100, y=nube["retorno"] * 100, mode="markers",
        marker=dict(size=4, color=nube["sharpe"], colorscale="Blues", opacity=0.45,
                    colorbar=dict(title="Sharpe")),
        name="Universo factible",
        customdata=list(zip(pesos_nube, nube["sharpe"].astype(float))),
        hovertemplate=("vol %{x:.2f}% · ret %{y:.2f}% · Sharpe %{customdata[1]:.2f}<br>"
                       "<b>Pesos:</b> %{customdata[0]}<extra></extra>")))
    pesos_str = [_pesos_texto(f, activos) for _, f in puntos.iterrows()]
    fig.add_trace(go.Scatter(
        x=puntos["volatilidad"] * 100, y=puntos["retorno"] * 100, mode="lines+markers",
        line=dict(color=estilo.TINTA, width=2.0), marker=dict(size=6, color=estilo.TINTA),
        name="Frontera MV de referencia", customdata=list(zip(pesos_str, puntos["sharpe"].astype(float))),
        hovertemplate=("vol %{x:.2f}% · ret %{y:.2f}% · Sharpe %{customdata[1]:.2f}<br>"
                       "<b>Pesos:</b> %{customdata[0]}<extra></extra>")))
    for c in payload.candidatos:
        nombre = f"{c.motor_optimizacion}/{estilo.nombre_nivel(c.nivel, payload.configuracion.idioma_reporte)}"
        motor = c.motor_optimizacion or ""
        pesos_candidato = _pesos_serie_texto(c.pesos)
        fig.add_trace(go.Scatter(
            x=[c.volatilidad_estructural * 100], y=[c.retorno_esperado * 100], mode="markers",
            marker=dict(
                size=14,
                color=estilo.MOTOR_COLOR.get(motor, estilo.ACENTO),
                symbol=_simbolo_motor(motor),
                line=dict(color="white", width=1.5),
            ),
            name=nombre,
            customdata=[(pesos_candidato, c.sharpe)],
            hovertemplate=f"{nombre}<br>"
                          "vol %{x:.2f}% · ret %{y:.2f}% · Sharpe %{customdata[1]:.2f}<br>"
                          "<b>Pesos:</b> %{customdata[0]}<extra></extra>"))
    f = _layout(fig, "Riesgo-retorno: frontera MV de referencia vs retadores",
                "Volatilidad anual (%)",
                "Retorno esperado anual (%)", 480)
    f.update_layout(
        hovermode="closest",
        margin=dict(l=60, r=24, t=68, b=106),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        annotations=[dict(
            text=narrativa.texto_frontera_breve(),
            xref="paper", yref="paper", x=0, y=-0.30, showarrow=False,
            xanchor="left", yanchor="top", align="left",
            font=dict(size=10, color=estilo.SUAVE),
        )],
    )
    return f


_CLASE_COLOR = {"bajo": "#0EA5E9", "medio": "#1D4ED8", "alto": "#7C3AED"}


def _activos_de(cf) -> list[str]:
    return [c[len("peso·"):] for c in cf.columns if c.startswith("peso·")]


def _pesos_texto(fila, activos: list[str]) -> str:
    """Cadena de pesos ordenada (para el tooltip y el panel de clic)."""
    pares = sorted(((a, float(fila[f"peso·{a}"])) for a in activos), key=lambda p: -p[1])
    return " · ".join(f"{a} {w*100:.1f}%" for a, w in pares if w > 0.005)


def _pesos_serie_texto(pesos) -> str:
    pares = pesos[pesos > 0.005].sort_values(ascending=False)
    return " · ".join(f"{a} {float(w)*100:.1f}%" for a, w in pares.items())


def _simbolo_motor(motor: str) -> str:
    return {"MARKOWITZ": "circle", "CVAR": "diamond", "NCO": "square"}.get(motor, "circle")


def _customdata_frontera(sub, activos: list[str]):
    """[pesos_str, sharpe, diversificacion] por punto."""
    pesos = [_pesos_texto(f, activos) for _, f in sub.iterrows()]
    return list(zip(pesos, sub["sharpe"].astype(float), sub["diversificacion"].astype(float)))


def _customdata_universo(sub, activos: list[str]):
    """[pesos_str, sharpe] por punto de la nube factible."""
    pesos = [_pesos_texto(f, activos) for _, f in sub.iterrows()]
    return list(zip(pesos, sub["sharpe"].astype(float)))


def frontera_clasificada(payload: ReportPayload) -> go.Figure:
    """Frontera + nube clasificadas por banda de riesgo, con anclas y líderes.

    Al situar el ratón sobre un punto de la frontera el tooltip muestra TODOS los
    pesos de esa cartera; al hacer clic, el panel lateral (en el HTML) los fija.
    """
    nube = payload.clasificacion_nube
    cf = payload.clasificacion_frontera
    activos = _activos_de(cf)
    fig = go.Figure()
    # Nube de fondo por clase.
    for clase, color in _CLASE_COLOR.items():
        sub = nube[nube["clase"] == clase]
        if not sub.empty:
            fig.add_trace(go.Scattergl(
                x=sub["volatilidad"] * 100, y=sub["retorno"] * 100, mode="markers",
                marker=dict(size=4, color=color, opacity=0.18), name=f"Universo factible · {clase}",
                showlegend=False,
                customdata=_customdata_universo(sub, activos),
                hovertemplate=(f"<b>Universo {clase}</b> · vol %{{x:.2f}}% · ret %{{y:.2f}}%<br>"
                               "Sharpe %{customdata[1]:.2f}<br>"
                               "<b>Pesos:</b> %{customdata[0]}<extra></extra>")))
    # Frontera por clase (con pesos en el hover y en customdata para el clic).
    for clase, color in _CLASE_COLOR.items():
        sub = cf[cf["clase"] == clase]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["volatilidad"] * 100, y=sub["retorno"] * 100, mode="markers",
                marker=dict(size=8, color=color, line=dict(color="white", width=0.6)),
                name=f"Frontera MV · {estilo.nombre_nivel(clase, payload.configuracion.idioma_reporte)}",
                hovertemplate=(f"<b>{clase.title()}</b> · vol %{{x:.2f}}% · ret %{{y:.2f}}%<br>"
                               "Sharpe %{customdata[1]:.2f} · Div %{customdata[2]:.2f}<br>"
                               "<b>Pesos:</b> %{customdata[0]}<extra></extra>"),
                customdata=_customdata_frontera(sub, activos)))
    # Anclas.
    for nombre, vol in payload.anclas:
        fila = cf.iloc[(cf["volatilidad"] - vol).abs().argmin()]
        fig.add_trace(go.Scatter(
            x=[vol * 100], y=[fila["retorno"] * 100], mode="markers+text",
            marker=dict(size=15, color=estilo.TINTA, symbol="diamond", line=dict(color="white", width=1.5)),
            text=[nombre], textposition="top center", textfont=dict(size=9, color=estilo.TINTA),
            name=nombre, showlegend=False, hovertemplate=f"{nombre}<br>vol %{{x:.2f}}%<extra></extra>"))
    f = _layout(fig, "Frontera MV clasificada por banda de riesgo",
                "Volatilidad anual (%)", "Retorno esperado anual (%)", 500)
    f.update_layout(
        hovermode="closest",
        margin=dict(l=60, r=24, t=64, b=72),
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0),
    )
    return f


def correlacion(payload: ReportPayload) -> go.Figure:
    corr = payload.momentos.correlacion
    activos = list(corr.index)
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=activos, y=activos, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.round(2).to_numpy(), texttemplate="%{text}",
        hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>"))
    return _layout(fig, "Correlación media", "", "", 440)


def correlacion_rolling(payload: ReportPayload) -> go.Figure:
    """Correlación media par-a-par en ventana móvil (cómo varía la diversificación
    a lo largo del tiempo). Consume la serie ya calculada en MetricasHistoricas."""
    mh = getattr(payload, "metricas_historicas", None)
    ventana = mh.ventana_rolling if mh is not None else 252
    fig = go.Figure()
    if mh is not None and len(mh.correlacion_rolling):
        s = mh.correlacion_rolling
        fig.add_trace(go.Scatter(
            x=s.index.to_numpy(), y=s.to_numpy(), mode="lines",
            line=dict(color=estilo.ACENTO, width=1.8),
            name=f"Correlación media (rolling {ventana}d)",
            hovertemplate="%{x|%d-%m-%Y}<br>Corr media: %{y:.2f}<extra></extra>"))
        media = float(s.mean())
        fig.add_hline(y=media, line=dict(color=estilo.SUAVE, width=1, dash="dash"),
                      annotation_text=f"media {media:.2f}", annotation_position="top left")
    return _layout(fig, f"Correlación media móvil ({ventana} días)",
                   "Fecha", "Correlación media", 380)


def generar_figuras(payload: ReportPayload) -> dict[str, go.Figure]:
    figs = {
        "fan_chart": fan_chart(payload),
        "equity_drawdown": equity_drawdown(payload),
        "pesos": pesos(payload),
        "mcr": mcr(payload),
        "var_forecast": var_forecast(payload),
        "frontera": frontera(payload),
        "correlacion": correlacion(payload),
    }
    if len(payload.clasificacion_frontera) and len(payload.clasificacion_nube):
        figs["frontera_clasificada"] = frontera_clasificada(payload)
    mh = getattr(payload, "metricas_historicas", None)
    if mh is not None and len(getattr(mh, "correlacion_rolling", [])):
        figs["correlacion_rolling"] = correlacion_rolling(payload)
    return figs

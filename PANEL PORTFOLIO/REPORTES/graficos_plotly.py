"""Gráficos interactivos (Plotly) para el informe HTML.

No son visualizaciones genéricas: cada figura responde una pregunta concreta del
análisis (dónde caen las carteras en el plano riesgo-retorno, cómo evolucionan
out-of-sample, dónde se evapora la diversificación). Se devuelven objetos Figure;
el ensamblador HTML los incrusta offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from CONTRATOS.modelos import PaqueteReporte
from OPTIMIZACION.asignadores import metodos

from .formato import ACENTO, COLOR_METODO, LINEA, NEGATIVO, POSITIVO, SUAVE, TINTA, nombre_visible
from .i18n import perfil_visible, t

_FUENTE = "Georgia, 'Times New Roman', serif"
_PALETA_ACTIVOS = ["#1D4ED8", "#0E7490", "#15803D", "#B45309", "#BE123C", "#7C3AED", "#475569"]


def _layout(fig: go.Figure, titulo: str, x: str, y: str, alto: int = 460) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=17, color=TINTA, family=_FUENTE)),
        font=dict(family=_FUENTE, size=13, color=TINTA),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=alto,
        margin=dict(l=70, r=40, t=60, b=60),
        xaxis=dict(title=x, gridcolor=LINEA, zeroline=False, linecolor=LINEA),
        yaxis=dict(title=y, gridcolor=LINEA, zeroline=False, linecolor=LINEA),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=LINEA, borderwidth=1),
        hovermode="closest",
    )
    return fig


def _custom_pesos(pesos_filas, activos) -> list[list[float]]:
    """customdata: por cada punto, los pesos en el orden de `activos`."""
    return [[round(float(fila[a]), 6) for a in activos] for _, fila in pesos_filas.iterrows()]


def _titulo_perfil(paquete: PaqueteReporte, nivel: str) -> str:
    return perfil_visible(paquete, nivel)


def _metricas_recomendacion(paquete: PaqueteReporte) -> str:
    cartera = paquete.perfil_riesgo.recomendada
    mh = cartera.metricas_historicas
    if getattr(paquete.configuracion, "idioma_reporte", "es") == "it":
        return (
            f"rend. atteso {cartera.retorno_esperado:.1%} · "
            f"vol attesa {cartera.volatilidad_esperada:.1%} · "
            f"VaR {mh.var:.1%} · CVaR {mh.cvar:.1%} · maxDD {mh.max_drawdown:.1%}"
        )
    return (
        f"ret esperado {cartera.retorno_esperado:.1%} · "
        f"vol esperada {cartera.volatilidad_esperada:.1%} · "
        f"VaR {mh.var:.1%} · CVaR {mh.cvar:.1%} · maxDD {mh.max_drawdown:.1%}"
    )


def fig_frontera(paquete: PaqueteReporte) -> go.Figure:
    activos = list(paquete.datos.activos)
    mc = paquete.monte_carlo.metricas
    mc_pesos = paquete.monte_carlo.pesos
    fr = paquete.frontera.puntos.sort_values("volatilidad")
    cols_peso = [f"peso·{a}" for a in activos]

    fig = go.Figure()
    # Nube: clic en cualquier cartera aleatoria muestra su composición.
    fig.add_trace(go.Scattergl(
        x=mc["volatilidad"], y=mc["retorno"], mode="markers",
        marker=dict(size=4, color=mc["sharpe"], colorscale="Blues", opacity=0.45,
                    colorbar=dict(title="Sharpe", thickness=12)),
        name=t(paquete, "carteras_aleatorias"),
        customdata=[[round(float(v), 6) for v in mc_pesos.iloc[i][activos].to_numpy()]
                    for i in range(len(mc_pesos))],
        hovertemplate="vol %{x:.2%}<br>ret %{y:.2%}<br><i>clic para ver pesos</i><extra></extra>",
    ))
    # Frontera eficiente: cada punto lleva sus pesos.
    fig.add_trace(go.Scatter(
        x=fr["volatilidad"], y=fr["retorno"], mode="lines+markers",
        line=dict(color=TINTA, width=2.5), marker=dict(size=6, color=TINTA),
        name=t(paquete, "frontera_eficiente_nombre"),
        customdata=_custom_pesos(fr[cols_peso].rename(columns=dict(zip(cols_peso, activos))), activos),
        hovertemplate=f"{t(paquete, 'frontera_eficiente_nombre')}<br>vol %{{x:.2%}}<br>ret %{{y:.2%}}<extra></extra>",
    ))
    for cartera in paquete.perfil_riesgo.carteras:
        fig.add_trace(go.Scatter(
            x=[cartera.volatilidad_esperada], y=[cartera.retorno_esperado], mode="markers+text",
            marker=dict(size=13, color="white", line=dict(color=ACENTO, width=2), symbol="circle"),
            text=[_titulo_perfil(paquete, cartera.nivel)], textposition="top center",
            name=f"{t(paquete, 'nivel_prefijo')} {_titulo_perfil(paquete, cartera.nivel)}",
            customdata=[[round(float(cartera.pesos[a]), 6) for a in activos]],
            hovertemplate=(
                f"{_titulo_perfil(paquete, cartera.nivel)}<br>vol %{{x:.2%}}<br>ret %{{y:.2%}}<extra></extra>"
            ),
        ))
    recomendada = paquete.perfil_riesgo.recomendada
    fig.add_trace(go.Scatter(
        x=[recomendada.volatilidad_esperada], y=[recomendada.retorno_esperado],
        mode="markers+text",
        marker=dict(size=20, color=ACENTO, line=dict(color="white", width=2), symbol="star"),
        text=[t(paquete, "perfil_recomendado")], textposition="bottom center",
        name=f"{t(paquete, 'perfil_recomendado')} · {_titulo_perfil(paquete, recomendada.nivel)}",
        customdata=[[round(float(recomendada.pesos[a]), 6) for a in activos]],
        hovertemplate=(
            f"{t(paquete, 'perfil_recomendado')} · {_titulo_perfil(paquete, recomendada.nivel)}<br>"
            "vol %{x:.2%}<br>ret %{y:.2%}<extra></extra>"
        ),
    ))
    for metodo in metodos(paquete.configuracion):
        m = paquete.asignaciones[metodo].metricas
        w = paquete.asignaciones[metodo].pesos
        etiqueta = nombre_visible(metodo, paquete)
        fig.add_trace(go.Scatter(
            x=[m.volatilidad_anual], y=[m.retorno_anual], mode="markers",
            marker=dict(size=14, color=COLOR_METODO.get(metodo, ACENTO),
                        line=dict(color="white", width=1.5), symbol="diamond"),
            name=etiqueta,
            customdata=[[round(float(w[a]), 6) for a in activos]],
            hovertemplate=f"{etiqueta}<br>vol %{{x:.2%}}<br>ret %{{y:.2%}}<br><i>clic para ver pesos</i><extra></extra>",
        ))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, t(paquete, "frontera_titulo"),
                   t(paquete, "volatilidad_anual"), t(paquete, "retorno_anual_esperado"), alto=560)


def fig_pesos_recomendados(paquete: PaqueteReporte) -> go.Figure:
    """Resultado principal: pesos de la cartera eficiente del perfil elegido."""
    cartera = paquete.perfil_riesgo.recomendada
    pesos = cartera.pesos.sort_values()
    fig = go.Figure(go.Bar(
        x=pesos.values,
        y=list(pesos.index),
        orientation="h",
        marker_color=[_PALETA_ACTIVOS[i % len(_PALETA_ACTIVOS)] for i in range(len(pesos))],
        text=[f"{v:.1%}" for v in pesos.values],
        textposition="auto",
        hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"{t(paquete, 'recomendacion')} · {t(paquete, 'perfil').format(perfil=_titulo_perfil(paquete, cartera.nivel))}<br><sup>{_metricas_recomendacion(paquete)}</sup>",
            font=dict(size=18, color=TINTA, family=_FUENTE),
        ),
        font=dict(family=_FUENTE, size=13, color=TINTA),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=430,
        margin=dict(l=90, r=40, t=80, b=50),
        xaxis=dict(title=t(paquete, "peso_recomendado"), tickformat=".0%", range=[0, max(0.01, float(pesos.max()) * 1.18)], gridcolor=LINEA),
        yaxis=dict(gridcolor=LINEA),
        showlegend=False,
    )
    return fig


def fig_pesos_niveles(paquete: PaqueteReporte) -> go.Figure:
    """Pesos de las carteras eficientes por perfil natural de riesgo."""
    activos = list(paquete.datos.activos)
    niveles = list(paquete.perfil_riesgo.carteras)
    etiquetas = [_titulo_perfil(paquete, c.nivel) for c in niveles]
    fig = go.Figure()
    for i, activo in enumerate(activos):
        fig.add_trace(go.Bar(
            x=etiquetas,
            y=[float(c.pesos[activo]) for c in niveles],
            name=activo,
            marker_color=_PALETA_ACTIVOS[i % len(_PALETA_ACTIVOS)],
            hovertemplate=f"{activo}: %{{y:.2%}}<extra></extra>",
        ))
    recomendada = paquete.perfil_riesgo.recomendada
    fig.add_vline(
        x=etiquetas.index(_titulo_perfil(paquete, recomendada.nivel)),
        line=dict(color=TINTA, dash="dot", width=1.5),
        annotation_text=t(paquete, "perfil_recomendado"),
        annotation_position="top",
    )
    fig.update_layout(
        barmode="stack",
        title=dict(text=t(paquete, "niveles"), font=dict(size=16, color=TINTA, family=_FUENTE)),
        font=dict(family=_FUENTE, size=13, color=TINTA),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=430,
        margin=dict(l=70, r=40, t=60, b=70),
        xaxis=dict(title=t(paquete, "nivel_riesgo"), gridcolor=LINEA),
        yaxis=dict(title=t(paquete, "peso"), tickformat=".0%", range=[0, 1], gridcolor=LINEA),
        legend=dict(orientation="h", y=-0.20, bgcolor="rgba(255,255,255,0.7)"),
        hovermode="x unified",
    )
    return fig


def fig_composicion_frontera(paquete: PaqueteReporte) -> go.Figure:
    """Área apilada: cómo rota la composición al recorrer la frontera."""
    niveles = paquete.perfil_riesgo.niveles_frontera.sort_index()
    activos = [a for a in paquete.datos.activos if a in niveles.columns]
    x = niveles.index.to_numpy(dtype=float)
    fig = go.Figure()
    for i, activo in enumerate(activos):
        fig.add_trace(go.Scatter(
            x=x,
            y=niveles[activo].to_numpy(dtype=float),
            mode="lines",
            stackgroup="pesos",
            name=activo,
            line=dict(width=0.5, color=_PALETA_ACTIVOS[i % len(_PALETA_ACTIVOS)]),
            hovertemplate=f"{activo}: %{{y:.2%}}<extra></extra>",
        ))
    rec = paquete.perfil_riesgo.recomendada
    fig.add_vline(
        x=rec.volatilidad_esperada,
        line=dict(color=TINTA, dash="dot", width=1.6),
        annotation_text=t(paquete, "perfil").format(perfil=_titulo_perfil(paquete, rec.nivel)),
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text=t(paquete, "composicion_frontera_titulo"), font=dict(size=16, color=TINTA, family=_FUENTE)),
        font=dict(family=_FUENTE, size=13, color=TINTA),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=440,
        margin=dict(l=70, r=40, t=60, b=70),
        xaxis=dict(title=t(paquete, "volatilidad_anual"), tickformat=".0%", gridcolor=LINEA),
        yaxis=dict(title=t(paquete, "peso"), tickformat=".0%", range=[0, 1], gridcolor=LINEA),
        legend=dict(orientation="h", y=-0.20, bgcolor="rgba(255,255,255,0.7)"),
        hovermode="x unified",
    )
    return fig


def fig_convexidad(paquete: PaqueteReporte) -> go.Figure:
    """Retorno medio diario por escenario (todo baja / mixto / todo sube)."""
    conv = paquete.riesgo.convexidad
    escenarios = [("ret_medio_todo_baja", t(paquete, "todo_baja"), NEGATIVO),
                  ("ret_medio_mixto", t(paquete, "mixto"), SUAVE),
                  ("ret_medio_todo_sube", t(paquete, "todo_sube"), POSITIVO)]
    fig = go.Figure()
    metodos_ord = [m for m in metodos(paquete.configuracion) if m in conv.index]
    etiquetas = [nombre_visible(m, paquete) for m in metodos_ord]
    for clave, nombre, color in escenarios:
        if clave not in conv.columns:
            continue
        fig.add_trace(go.Bar(
            x=etiquetas, y=conv.loc[metodos_ord, clave].to_numpy(), name=nombre, marker_color=color,
            hovertemplate=f"{nombre}: %{{y:.3%}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text=t(paquete, "convexidad_titulo"),
                   font=dict(size=16, color=TINTA, family=_FUENTE)),
        font=dict(family=_FUENTE, size=12, color=TINTA), paper_bgcolor="white", plot_bgcolor="white",
        height=440, margin=dict(l=70, r=40, t=60, b=110),
        xaxis=dict(gridcolor=LINEA, tickangle=-20),
        yaxis=dict(title=t(paquete, "retorno_medio_diario"), tickformat=".2%", gridcolor=LINEA, zeroline=True, zerolinecolor=SUAVE),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=LINEA, borderwidth=1),
    )
    return fig


def fig_equity(paquete: PaqueteReporte) -> go.Figure:
    eq = paquete.riesgo.walk_forward.equity
    fig = go.Figure()
    for metodo in eq.columns:
        etiqueta = nombre_visible(metodo, paquete)
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq[metodo], mode="lines", name=etiqueta,
            line=dict(color=COLOR_METODO.get(metodo, SUAVE), width=2),
            hovertemplate=f"{etiqueta}<br>%{{x|%Y-%m-%d}}<br>%{{y:.3f}}×<extra></extra>",
        ))
    return _layout(fig, t(paquete, "equity_titulo"),
                   t(paquete, "fecha"), t(paquete, "capital_acumulado"), alto=480)


def fig_drawdown(paquete: PaqueteReporte) -> go.Figure:
    eq = paquete.riesgo.walk_forward.equity
    fig = go.Figure()
    for metodo in eq.columns:
        etiqueta = nombre_visible(metodo, paquete)
        caida = eq[metodo] / eq[metodo].cummax() - 1.0
        fig.add_trace(go.Scatter(
            x=caida.index, y=caida, mode="lines", name=etiqueta,
            line=dict(color=COLOR_METODO.get(metodo, SUAVE), width=1.6),
            hovertemplate=f"{etiqueta}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2%}}<extra></extra>",
        ))
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, t(paquete, "drawdown_titulo"),
                   t(paquete, "fecha"), t(paquete, "drawdown"), alto=420)


def _heatmap(matriz: pd.DataFrame, titulo: str, escala, zmid=None) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matriz.values, x=list(matriz.columns), y=list(matriz.index),
        colorscale=escala, zmid=zmid,
        text=np.round(matriz.values, 2), texttemplate="%{text}",
        textfont=dict(size=12), hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=12),
    ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=16, color=TINTA, family=_FUENTE)),
        font=dict(family=_FUENTE, size=12, color=TINTA),
        paper_bgcolor="white", plot_bgcolor="white", height=440,
        margin=dict(l=110, r=40, t=60, b=80),
    )
    return fig


def fig_correlacion_media(paquete: PaqueteReporte) -> go.Figure:
    return _heatmap(paquete.analisis.correlacion_media,
                    t(paquete, "corr_media_titulo"), "RdBu", zmid=0)


def fig_correlacion_cola(paquete: PaqueteReporte) -> go.Figure:
    return _heatmap(paquete.analisis.diferencia_correlacion_cola,
                    t(paquete, "corr_cola_titulo"),
                    "Reds")


def fig_pca(paquete: PaqueteReporte) -> go.Figure:
    ve = paquete.analisis.pca.varianza_explicada
    vac = paquete.analisis.pca.varianza_acumulada
    etiquetas = [f"PC{i + 1}" for i in range(len(ve))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=etiquetas, y=ve.values, name=t(paquete, "varianza_explicada"),
                         marker_color=ACENTO, hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=etiquetas, y=vac.values, name=t(paquete, "acumulada"), yaxis="y2",
                             mode="lines+markers", line=dict(color=TINTA, width=2)))
    fig.add_hline(y=0.90, line=dict(color=SUAVE, dash="dot"), yref="y2")
    fig.update_layout(
        title=dict(text=t(paquete, "pca_titulo"), font=dict(size=16, color=TINTA, family=_FUENTE)),
        font=dict(family=_FUENTE, size=13, color=TINTA), paper_bgcolor="white", plot_bgcolor="white",
        height=440, margin=dict(l=70, r=70, t=60, b=60),
        xaxis=dict(gridcolor=LINEA),
        yaxis=dict(title=t(paquete, "varianza_explicada"), tickformat=".0%", gridcolor=LINEA),
        yaxis2=dict(title=t(paquete, "acumulada"), overlaying="y", side="right", tickformat=".0%", range=[0, 1.02]),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=LINEA, borderwidth=1),
    )
    return fig


def fig_pesos(paquete: PaqueteReporte) -> go.Figure:
    activos = list(paquete.datos.activos)
    fig = go.Figure()
    metodos_ord = list(metodos(paquete.configuracion))
    etiquetas = [nombre_visible(m, paquete) for m in metodos_ord]
    for i, activo in enumerate(activos):
        valores = [float(paquete.asignaciones[m].pesos[activo]) for m in metodos_ord]
        fig.add_trace(go.Bar(
            y=etiquetas, x=valores, orientation="h", name=activo,
            marker_color=_PALETA_ACTIVOS[i % len(_PALETA_ACTIVOS)],
            hovertemplate=f"{activo}: %{{x:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        title=dict(text=t(paquete, "pesos_metodos_titulo"), font=dict(size=16, color=TINTA, family=_FUENTE)),
        font=dict(family=_FUENTE, size=13, color=TINTA), paper_bgcolor="white", plot_bgcolor="white",
        height=420, margin=dict(l=150, r=40, t=60, b=50),
        xaxis=dict(title=t(paquete, "peso"), tickformat=".0%", gridcolor=LINEA, range=[0, 1]),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.18, bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


def fig_regimen(paquete: PaqueteReporte) -> go.Figure:
    """Retorno anualizado de cada método dentro de cada régimen (heatmap)."""
    por = paquete.riesgo.metricas_por_regimen
    regimenes = sorted({r for df in por.values() for r in df.index})
    matriz = pd.DataFrame(
        {nombre_visible(metodo, paquete): [por[metodo]["retorno_anual"].get(r, np.nan) for r in regimenes]
         for metodo in metodos(paquete.configuracion) if metodo in por},
        index=regimenes,
    ).T
    return _heatmap(matriz.round(2),
                    t(paquete, "regimen_titulo"),
                    "RdYlGn", zmid=0)


def fig_diversificacion(paquete: PaqueteReporte) -> go.Figure:
    div = paquete.riesgo.diversificacion_crisis
    etiquetas = [nombre_visible(m, paquete) for m in div.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=etiquetas, y=div["enb_global"], name=t(paquete, "enb_global"),
                         marker_color=ACENTO, hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=etiquetas, y=div["enb_crisis"], name=t(paquete, "enb_crisis"),
                         marker_color="#B45309", hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_layout(
        barmode="group",
        title=dict(text=t(paquete, "diversificacion_titulo"), font=dict(size=16, color=TINTA, family=_FUENTE)),
        font=dict(family=_FUENTE, size=13, color=TINTA), paper_bgcolor="white", plot_bgcolor="white",
        height=420, margin=dict(l=70, r=40, t=60, b=70),
        xaxis=dict(gridcolor=LINEA), yaxis=dict(title=t(paquete, "apuestas_independientes"), gridcolor=LINEA),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=LINEA, borderwidth=1),
    )
    return fig


def todas_las_figuras(paquete: PaqueteReporte) -> dict[str, go.Figure]:
    return {
        "frontera": fig_frontera(paquete),
        "pesos_recomendados": fig_pesos_recomendados(paquete),
        "pesos_niveles": fig_pesos_niveles(paquete),
        "composicion_frontera": fig_composicion_frontera(paquete),
        "equity": fig_equity(paquete),
        "drawdown": fig_drawdown(paquete),
        "correlacion_media": fig_correlacion_media(paquete),
        "correlacion_cola": fig_correlacion_cola(paquete),
        "pca": fig_pca(paquete),
        "pesos": fig_pesos(paquete),
        "regimen": fig_regimen(paquete),
        "diversificacion": fig_diversificacion(paquete),
        "convexidad": fig_convexidad(paquete),
    }

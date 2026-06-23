"""Informe HTML institucional: offline, interactivo, autoexplicado.

Técnicas: Plotly incrustado inline (sin CDN, abre sin internet), navegación
lateral fija, tabla maestra con resaltado, tarjetas-resumen, narrativa por
sección y glosario. Diseño sobrio de mesa profesional, no plantilla genérica.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

from CONTRATOS.modelos import PaqueteReporte

from .formato import (
    COLOR_METODO,
    fecha,
    num,
    pct,
)
from .graficos_plotly import todas_las_figuras
from .i18n import columna_visible, glosario, metodo_visible, perfil_visible, t
from .narrativa import construir_secciones
from .tablas import tabla_espejismo, tabla_maestra, tabla_pesos_niveles

_COLS_PCT = {
    "Retorno esperado (in-sample)", "Volatilidad esperada (in-sample)",
    "Retorno anual (OOS)", "Volatilidad (OOS)", "Max drawdown (OOS)",
    "VaR 95% (OOS)", "CVaR 95% (OOS)", "Retorno esperado",
    "Volatilidad esperada", "VaR histórico", "CVaR histórico",
    "Max drawdown histórico", "Retorno realizado (OOS)",
}


def _div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displaylogo": False, "responsive": True})


def _tabla_maestra_html(paquete: PaqueteReporte) -> str:
    df = tabla_maestra(paquete)
    mejor = max(paquete.riesgo.metricas.items(), key=lambda kv: kv[1].sharpe)[0]
    return _tabla_df_html(df, indice="Método", fila_resaltada=mejor, paquete=paquete)


def _tabla_df_html(
    df: pd.DataFrame,
    indice: str,
    fila_resaltada: str | None = None,
    paquete: PaqueteReporte | None = None,
) -> str:
    cabecera = "".join(f"<th>{_html.escape(columna_visible(paquete, c) if paquete is not None else c)}</th>" for c in df.columns)
    filas = []
    for etiqueta, fila in df.iterrows():
        etiqueta_txt = str(etiqueta)
        clase = ' class="mejor"' if etiqueta_txt == fila_resaltada else ""
        color = COLOR_METODO.get(etiqueta_txt, "#475569")
        punto = f'<span class="punto" style="background:{color}"></span>' if indice == "Método" else ""
        etiqueta_visible = metodo_visible(paquete, etiqueta_txt) if paquete is not None and indice == "Método" else etiqueta_txt
        celdas = [f'<th class="metodo">{punto}{_html.escape(etiqueta_visible)}</th>']
        for col in df.columns:
            valor = fila[col]
            if col.startswith("peso ·") or col in _COLS_PCT:
                celdas.append(f"<td>{pct(valor)}</td>")
            else:
                celdas.append(f"<td>{num(valor)}</td>")
        filas.append(f"<tr{clase}>" + "".join(celdas) + "</tr>")
    return (
        '<div class="tabla-scroll"><table class="maestra">'
        f"<thead><tr><th>{_html.escape(columna_visible(paquete, indice) if paquete is not None else indice)}</th>{cabecera}</tr></thead>"
        f"<tbody>{''.join(filas)}</tbody></table></div>"
    )


def _tabla_niveles_html(paquete: PaqueteReporte) -> str:
    df = tabla_pesos_niveles(paquete)
    return _tabla_df_html(df, indice="Nivel", paquete=paquete)


def _tabla_espejismo_html(paquete: PaqueteReporte) -> str:
    df = tabla_espejismo(paquete)
    return _tabla_df_html(df, indice="Método", paquete=paquete)


def _kpis(paquete: PaqueteReporte) -> str:
    cartera = paquete.perfil_riesgo.recomendada
    mh = cartera.metricas_historicas
    tarjetas = [
        (t(paquete, "retorno_esperado"), pct(cartera.retorno_esperado), t(paquete, "frontera_eficiente")),
        (t(paquete, "volatilidad_esperada"), pct(cartera.volatilidad_esperada), t(paquete, "perfil").format(perfil=cartera.nivel)),
        (t(paquete, "var_historico"), pct(mh.var), t(paquete, "cola_diaria")),
        (t(paquete, "maxdd_historico"), pct(mh.max_drawdown), t(paquete, "pesos_fijos")),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-t">{_html.escape(t)}</div>'
        f'<div class="kpi-s">{_html.escape(s)}</div></div>'
        for t, v, s in tarjetas
    ) + "</div>"


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


def _seccion(idx: str, titulo: str, parrafos, extra: str = "") -> str:
    cuerpo = "".join(f"<p>{_html.escape(p)}</p>" for p in parrafos)
    return f'<section id="{idx}"><h2>{_html.escape(titulo)}</h2>{cuerpo}{extra}</section>'


def _glosario_html(paquete: PaqueteReporte) -> str:
    filas = "".join(
        f"<tr><th>{_html.escape(k)}</th><td>{_html.escape(v)}</td></tr>"
        for k, v in glosario(paquete).items()
    )
    return f'<table class="glosario"><tbody>{filas}</tbody></table>'


def generar_html(paquete: PaqueteReporte, ruta: Path) -> Path:
    figs = todas_las_figuras(paquete)
    secciones = construir_secciones(paquete)
    cfg = paquete.configuracion

    nav_items = [
        ("recomendacion", t(paquete, "nav_recomendacion")),
        ("niveles", t(paquete, "nav_niveles")),
        ("covarianza", t(paquete, "nav_covarianza")),
        ("validacion", t(paquete, "nav_validacion")),
        ("regimenes", t(paquete, "nav_regimenes")),
        ("diversificacion", t(paquete, "nav_diversificacion")),
        ("glosario", t(paquete, "nav_glosario")),
    ]
    nav = "".join(f'<a href="#{i}">{_html.escape(t)}</a>' for i, t in nav_items)

    # Mapa de secciones de narrativa por título.
    por_titulo = {s.titulo: s.parrafos for s in secciones}

    bloques = []
    bloques.append(
        f'<section id="recomendacion" class="hero-pesos"><h2>{_html.escape(t(paquete, "recomendacion"))}</h2>'
        + f"<p>{_html.escape(_perfil_resumen(paquete))}</p>"
        + _kpis(paquete)
        + _div(figs["pesos_recomendados"], "g_pesos_recomendados")
        + f'<div class="aviso">{_html.escape(t(paquete, "honestidad"))}</div></section>'
    )
    bloques.append(_seccion("niveles", t(paquete, "niveles"),
                            (t(paquete, "niveles_intro"),),
                            _tabla_niveles_html(paquete)
                            + _div(figs["pesos_niveles"], "g_pesos_niveles")
                            + _div(figs["composicion_frontera"], "g_composicion_frontera")
                            + _div(figs["frontera"], "g_frontera")))
    bloques.append(_seccion("covarianza", t(paquete, "covarianza"),
                            por_titulo["Análisis y diversificación"],
                            _div(figs["correlacion_media"], "g_corr") + _div(figs["correlacion_cola"], "g_corr_cola")
                            + _div(figs["pca"], "g_pca")))
    bloques.append(_seccion("validacion", t(paquete, "validacion"),
                            por_titulo["Backtest walk-forward (out-of-sample)"],
                            f"<h3>{_html.escape(t(paquete, 'tabla_oos'))}</h3>"
                            + _tabla_maestra_html(paquete)
                            + f"<h3>{_html.escape(t(paquete, 'promesa_realidad'))}</h3>"
                            + _tabla_espejismo_html(paquete)
                            + _div(figs["equity"], "g_equity")
                            + _div(figs["drawdown"], "g_drawdown")
                            + _div(figs["convexidad"], "g_convexidad")))
    bloques.append(_seccion("regimenes", t(paquete, "regimenes"),
                            por_titulo["Regímenes y stress testing"],
                            _div(figs["regimen"], "g_regimen")))
    bloques.append(_seccion("diversificacion", t(paquete, "diversificacion"),
                            por_titulo["Diversificación en crisis"],
                            _div(figs["diversificacion"], "g_div") + _div(figs["pesos"], "g_pesos_metodos")))
    bloques.append(f'<section id="glosario"><h2>{_html.escape(t(paquete, "nav_glosario"))}</h2>{_glosario_html(paquete)}'
                   f'<div class="aviso">{_html.escape(t(paquete, "honestidad"))}</div></section>')

    plotly_js = pyo.get_plotlyjs()
    doc = f"""<!DOCTYPE html>
<html lang="{_html.escape(t(paquete, "lang"))}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PANEL PORTFOLIO — {_html.escape(t(paquete, "titulo"))}</title>
<script>{plotly_js}</script>
<style>{_CSS}</style>
</head><body>
<nav class="lateral">
  <div class="marca">PANEL<br>PORTFOLIO</div>
  <div class="sub">{_html.escape(t(paquete, "subtitulo_nav"))}</div>
  {nav}
  <div class="pie">{_html.escape(', '.join(cfg.tickers))}<br>{cfg.fecha_inicio} → {cfg.fecha_fin}</div>
</nav>
<main>
  <header>
    <div class="eyebrow">{_html.escape(t(paquete, "eyebrow"))}</div>
    <h1>{_html.escape(t(paquete, "titulo"))}</h1>
    <div class="meta">{_html.escape(t(paquete, "meta_analisis").format(n=len(cfg.tickers), inicio=cfg.fecha_inicio, fin=cfg.fecha_fin, rebalanceo=cfg.frecuencia_rebalanceo, ventana=cfg.ventana_estimacion))}</div>
  </header>
  {''.join(bloques)}
  <footer>{_html.escape(t(paquete, "footer"))}</footer>
</main>
</body></html>"""
    ruta.write_text(doc, encoding="utf-8")
    return ruta


_CSS = """
:root{--tinta:#0F172A;--suave:#475569;--linea:#D9DEE7;--acento:#1D4ED8;--panel:#F6F8FB;--neg:#B91C1C;}
*{box-sizing:border-box;}
body{margin:0;font-family:Georgia,'Times New Roman',serif;color:var(--tinta);background:#fff;line-height:1.6;}
.lateral{position:fixed;top:0;left:0;width:230px;height:100vh;background:var(--tinta);color:#E6ECF5;padding:28px 22px;display:flex;flex-direction:column;gap:6px;}
.lateral .marca{font-size:20px;font-weight:700;letter-spacing:2px;line-height:1.15;}
.lateral .sub{font-size:11px;color:#9FB3CE;margin-bottom:22px;border-bottom:1px solid #27364d;padding-bottom:14px;}
.lateral a{color:#C7D4E6;text-decoration:none;font-size:13px;padding:7px 0;border-bottom:1px solid #1c2940;transition:.15s;}
.lateral a:hover{color:#fff;padding-left:6px;}
.lateral .pie{margin-top:auto;font-size:11px;color:#7E91AC;line-height:1.5;}
main{margin-left:230px;padding:54px 64px 80px;max-width:1080px;}
.eyebrow{text-transform:uppercase;letter-spacing:3px;font-size:11px;color:var(--acento);font-family:Arial,sans-serif;}
h1{font-size:34px;margin:8px 0 10px;line-height:1.15;}
.meta{color:var(--suave);font-size:14px;margin-bottom:10px;}
header{border-bottom:3px solid var(--tinta);padding-bottom:26px;margin-bottom:10px;}
section{padding:34px 0;border-bottom:1px solid var(--linea);}
h2{font-size:23px;margin:0 0 14px;position:relative;padding-left:16px;}
h2:before{content:"";position:absolute;left:0;top:5px;bottom:5px;width:5px;background:var(--acento);}
p{margin:0 0 13px;font-size:15px;max-width:80ch;}
.aviso{background:#FFF7ED;border-left:4px solid #B45309;padding:14px 18px;font-size:13.5px;color:#7C2D12;margin-top:18px;border-radius:0 6px 6px 0;}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:22px 0 6px;}
.kpi{background:var(--panel);border:1px solid var(--linea);border-radius:10px;padding:18px;}
.kpi-v{font-size:28px;font-weight:700;color:var(--tinta);}
.kpi-t{font-size:13px;color:var(--suave);margin-top:4px;font-family:Arial,sans-serif;}
.kpi-s{font-size:11px;color:#8A99AD;margin-top:2px;}
.tabla-scroll{overflow-x:auto;margin-top:14px;border:1px solid var(--linea);border-radius:8px;}
table.maestra{border-collapse:collapse;width:100%;font-size:12.5px;font-family:Arial,sans-serif;}
table.maestra th,table.maestra td{padding:9px 11px;text-align:right;border-bottom:1px solid var(--linea);white-space:nowrap;}
table.maestra thead th{background:var(--tinta);color:#fff;position:sticky;top:0;font-weight:600;}
table.maestra th.metodo{text-align:left;background:var(--panel);font-weight:700;}
table.maestra tr.mejor td,table.maestra tr.mejor th.metodo{background:#EFF6FF;}
.punto{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;}
.glosario{border-collapse:collapse;width:100%;font-size:14px;}
.glosario th{text-align:left;width:230px;vertical-align:top;padding:9px 14px 9px 0;color:var(--acento);font-weight:700;border-bottom:1px solid var(--linea);}
.glosario td{padding:9px 0;border-bottom:1px solid var(--linea);color:var(--suave);}
footer{padding-top:30px;font-size:12px;color:#8A99AD;}
.plotly-graph-div{margin:18px 0;border:1px solid var(--linea);border-radius:8px;}
@media(max-width:820px){.lateral{display:none;}main{margin-left:0;padding:28px;}.kpis{grid-template-columns:repeat(2,1fr);}}
"""

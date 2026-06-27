"""Informe institucional unificado (Fase 7).

Un único HTML interactivo por estrategia que cuenta la historia en el ORDEN
CORRECTO: la narrativa importa porque un informe que abre con KPIs in-sample te
entrena a mirar el número equivocado. El orden es deliberado:

  1. Pre-registro          — la hipótesis económica escrita ANTES de empezar.
  2. Resultados OOS         — TITULAR: distribución de Sharpe (CPCV), WFA eff.
  3. Veredicto anti-sobreajuste — DSR (deflactado), PBO, MinBTL.
  4. Batería de robustez    — bootstrap, régimen, estrategia nula.
  5. Optimización IS        — al final, como CONTEXTO, no como conclusión.
  6. Curva de equity        — con el tramo OOS/holdout claramente marcado.

El número in-sample es contexto, no protagonista. (La "sensibilidad a costes" del
documento no se incluye: depende del modelo de costes de ejecución, fuera de
alcance.)

Generación de HTML pura (stdlib): no depende de Polars ni del motor, por lo que
es verificable comprobando el contenido y el orden de las secciones. Los gráficos
usan Plotly desde CDN (mejor medio que el PDF para una herramienta de research).
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

_COLORES = {"verde": "#1a9850", "ambar": "#f0a020", "rojo": "#d73027"}
_EMOJI = {"verde": "🟢", "ambar": "🟡", "rojo": "🔴"}

# Orden canónico de las secciones (la narrativa correcta del documento).
ORDEN_SECCIONES = (
    "pre-registro",
    "resultados-oos",
    "veredicto",
    "robustez",
    "optimizacion-is",
    "equity",
)


def generar_informe_institucional(datos: dict, *, ruta_salida=None) -> str:
    """Construye el HTML del informe y, opcionalmente, lo escribe en disco.

    `datos` es un dict con claves opcionales (cada sección se renderiza solo si
    está presente): ``cabecera``, ``preregistro``, ``oos``, ``veredicto``,
    ``robustez``, ``is``, ``equity``. El esquema está documentado en los helpers
    de cada sección. Devuelve el HTML como string.
    """
    secciones = [
        _seccion_pre_registro(datos.get("preregistro")),
        _seccion_oos(datos.get("oos")),
        _seccion_veredicto(datos.get("veredicto")),
        _seccion_robustez(datos.get("robustez")),
        _seccion_optimizacion_is(datos.get("is")),
        _seccion_equity(datos.get("equity")),
    ]
    cuerpo = "\n".join(s for s in secciones if s)
    htmls = _documento(datos.get("cabecera", {}), cuerpo)

    if ruta_salida is not None:
        ruta = Path(ruta_salida)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(htmls, encoding="utf-8")
    return htmls


# ---------------------------------------------------------------------------
# Secciones (en orden narrativo)
# ---------------------------------------------------------------------------

def _seccion_pre_registro(pre: dict | None) -> str:
    if not pre:
        return ""
    filas = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in pre.items()
    )
    return _bloque(
        "pre-registro",
        "1 · Pre-registro de la hipótesis",
        "La tesis económica escrita ANTES de ver ningún resultado.",
        f"<table class='kv'>{filas}</table>",
    )


def _seccion_oos(oos: dict | None) -> str:
    if not oos:
        return ""
    dist = oos.get("distribucion", {})
    items = [
        ("Sharpe OOS medio", _fmt(dist.get("media"))),
        ("Desviación", _fmt(dist.get("desviacion"))),
        ("p5", _fmt(dist.get("p5"))),
        ("p25", _fmt(dist.get("p25"))),
        ("Mediana", _fmt(dist.get("mediana"))),
        ("p75", _fmt(dist.get("p75"))),
        ("Fracción positiva", _fmt(dist.get("fraccion_positiva"))),
        ("Sharpe OOS / IS", _fmt(oos.get("ratio_oos_is"))),
        ("WFA efficiency", _fmt(oos.get("wfa_efficiency"))),
        ("Trayectorias CPCV", _fmt(dist.get("n"), entero=True)),
    ]
    tarjetas = "".join(
        f"<div class='card'><div class='card-v'>{v}</div><div class='card-k'>{html.escape(k)}</div></div>"
        for k, v in items
    )
    return _bloque(
        "resultados-oos",
        "2 · Resultados fuera de muestra (TITULAR)",
        "Lo primero que ves: la distribución OOS de CPCV, no un número in-sample.",
        f"<div class='cards'>{tarjetas}</div>",
    )


def _seccion_veredicto(ver: dict | None) -> str:
    if not ver:
        return ""
    color = str(ver.get("color", "ambar"))
    titular = (
        f"<div class='veredicto' style='border-color:{_COLORES.get(color, '#888')}'>"
        f"<span class='vbig'>{_EMOJI.get(color, '')} {html.escape(color.upper())}</span></div>"
    )
    titulares = [
        ("DSR (deflactado)", _fmt(ver.get("dsr"))),
        ("PBO", _fmt(ver.get("pbo"))),
        ("MinBTL (años)", _fmt(ver.get("minbtl"))),
    ]
    tarjetas = "".join(
        f"<div class='card'><div class='card-v'>{v}</div><div class='card-k'>{html.escape(k)}</div></div>"
        for k, v in titulares
    )
    criterios = ver.get("criterios", [])
    filas = "".join(
        f"<tr><td>{html.escape(str(c.get('nombre','')))}</td>"
        f"<td>{_fmt(c.get('valor'))}</td>"
        f"<td style='color:{_COLORES.get(str(c.get('color')), '#888')}'>"
        f"{_EMOJI.get(str(c.get('color')), '')} {html.escape(str(c.get('detalle','')))}</td></tr>"
        for c in criterios
    )
    tabla = (
        "<table class='grid'><thead><tr><th>Criterio</th><th>Valor</th>"
        f"<th>Evaluación</th></tr></thead><tbody>{filas}</tbody></table>"
        if filas else ""
    )
    return _bloque(
        "veredicto",
        "3 · Veredicto anti-sobreajuste",
        "La pregunta «¿esto es real?» respondida con números fijados a priori.",
        titular + f"<div class='cards'>{tarjetas}</div>" + tabla,
    )


def _seccion_robustez(rob: dict | None) -> str:
    if not rob:
        return ""
    partes = []
    boot = rob.get("bootstrap")
    if boot:
        items = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v)}</td></tr>" for k, v in boot.items()
        )
        partes.append(f"<h3>Bootstrap de trades</h3><table class='kv'>{items}</table>")
    reg = rob.get("regimen")
    if reg:
        filas = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{_fmt(v.get('sharpe') if isinstance(v, dict) else v)}</td></tr>"
            for k, v in reg.items()
        )
        partes.append(
            f"<h3>Rendimiento por régimen</h3><table class='grid'>"
            f"<thead><tr><th>Régimen</th><th>Sharpe</th></tr></thead><tbody>{filas}</tbody></table>"
        )
    nula = rob.get("nula")
    if nula:
        partes.append(f"<h3>Estrategia nula</h3><p>{html.escape(str(nula))}</p>")
    if not partes:
        return ""
    return _bloque(
        "robustez",
        "4 · Batería de robustez",
        "¿Y si tuve suerte? Intervalos de bootstrap, régimen y control vs. nula.",
        "".join(partes),
    )


def _seccion_optimizacion_is(is_: dict | None) -> str:
    if not is_:
        return ""
    items = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{_fmt(v)}</td></tr>" for k, v in is_.items()
    )
    return _bloque(
        "optimizacion-is",
        "5 · Optimización in-sample (contexto)",
        "Al final y como contexto, NO como conclusión: meseta de parámetros.",
        f"<table class='kv'>{items}</table>",
    )


def _seccion_equity(equity: dict | None) -> str:
    if not equity:
        return ""
    valores = list(equity.get("valores", []))
    idx_holdout = equity.get("indice_holdout")
    datos_js = json.dumps({"y": valores, "holdout": idx_holdout})
    grafico = (
        "<div id='equity-plot' style='height:420px'></div>"
        "<script>(function(){var d=" + datos_js + ";"
        "var x=d.y.map(function(_,i){return i;});"
        "var trazas=[{x:x,y:d.y,type:'scatter',mode:'lines',name:'Equity',line:{color:'#1a73e8'}}];"
        "var shapes=[];"
        "if(d.holdout!=null){shapes.push({type:'rect',xref:'x',yref:'paper',x0:d.holdout,"
        "x1:x.length-1,y0:0,y1:1,fillcolor:'rgba(215,48,39,0.10)',line:{width:0}});}"
        "if(window.Plotly){Plotly.newPlot('equity-plot',trazas,{margin:{t:10},"
        "shapes:shapes,annotations:d.holdout!=null?[{x:d.holdout,y:1,yref:'paper',"
        "text:'HOLDOUT BLOQUEADO',showarrow:false,font:{color:'#d73027'}}]:[]});}})();</script>"
    )
    nota = (
        "<p class='nota'>El tramo sombreado es el <strong>holdout bloqueado</strong>: "
        "datos que la optimización nunca vio.</p>"
    )
    return _bloque(
        "equity",
        "6 · Curva de equity completa",
        "Con el tramo OOS/holdout claramente marcado y separado del de optimización.",
        grafico + nota,
    )


# ---------------------------------------------------------------------------
# Plantilla y helpers
# ---------------------------------------------------------------------------

def _bloque(anchor: str, titulo: str, subtitulo: str, contenido: str) -> str:
    return (
        f"<section id='{anchor}'><h2>{html.escape(titulo)}</h2>"
        f"<p class='sub'>{html.escape(subtitulo)}</p>{contenido}</section>"
    )


def _documento(cabecera: dict, cuerpo: str) -> str:
    titulo = html.escape(str(cabecera.get("titulo", "Informe institucional de estrategia")))
    meta_items = [
        ("Estrategia", cabecera.get("estrategia")),
        ("Activo", cabecera.get("activo")),
        ("Timeframe", cabecera.get("timeframe")),
        ("Salida", cabecera.get("salida")),
        ("Modo", cabecera.get("modo")),
        ("Huella", cabecera.get("huella")),
    ]
    meta = " · ".join(
        f"{html.escape(str(k))}: <strong>{html.escape(str(v))}</strong>"
        for k, v in meta_items
        if v is not None
    )
    generado = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{{color-scheme:light dark}}
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:0 auto;padding:24px;line-height:1.5;color:#1a1a1a;background:#fafafa}}
h1{{margin:0 0 4px;font-size:22px}}
.meta{{color:#555;font-size:13px;margin-bottom:24px}}
section{{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:20px;margin-bottom:18px}}
h2{{margin:0 0 2px;font-size:17px}}
.sub{{margin:0 0 14px;color:#666;font-size:13px}}
h3{{font-size:14px;margin:16px 0 6px}}
.cards{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{flex:1 1 120px;background:#f4f7fb;border-radius:8px;padding:12px;text-align:center}}
.card-v{{font-size:20px;font-weight:700}}
.card-k{{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.04em}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}}
.kv th{{text-align:left;width:40%;color:#555;font-weight:600;padding:6px 8px;border-bottom:1px solid #eee}}
.kv td{{padding:6px 8px;border-bottom:1px solid #eee}}
.grid th,.grid td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}}
.veredicto{{border:2px solid #888;border-radius:8px;padding:14px;text-align:center;margin-bottom:12px}}
.vbig{{font-size:24px;font-weight:800}}
.nota{{font-size:12px;color:#777;margin-top:8px}}
</style></head>
<body>
<h1>{titulo}</h1>
<div class="meta">{meta}<br>Generado (UTC): {html.escape(generado)}</div>
{cuerpo}
</body></html>"""


def _fmt(valor, *, entero: bool = False) -> str:
    if valor is None:
        return "—"
    try:
        f = float(valor)
    except (TypeError, ValueError):
        return html.escape(str(valor))
    if entero:
        return f"{int(round(f)):,}"
    if f != f:  # NaN
        return "—"
    return f"{f:,.4g}"

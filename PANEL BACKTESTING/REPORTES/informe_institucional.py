"""Informe institucional unificado (Fase 7).

Un único HTML interactivo por estrategia que cuenta la historia en el ORDEN
CORRECTO: la narrativa importa porque un informe que abre con KPIs in-sample te
entrena a mirar el número equivocado. El orden es deliberado:

  0. Resumen ejecutivo      — veredicto + titulares (DSR, PBO, Sharpe OOS).
  1. Pre-registro           — la hipótesis económica escrita ANTES de empezar.
  2. Resultados OOS         — TITULAR: distribución de Sharpe (CPCV) + WFA.
  3. Veredicto anti-sobreajuste — DSR (deflactado), PBO, MinBTL + criterios.
  4. Batería de robustez    — bootstrap, régimen, estrategia nula.
  5. Optimización IS        — al final, como CONTEXTO, no como conclusión.
  6. Curva de equity        — con el tramo OOS/holdout claramente marcado.

El número in-sample es contexto, no protagonista. (La "sensibilidad a costes"
del documento no se incluye: depende del modelo de costes de ejecución, fuera de
alcance.)

Generación de HTML pura (stdlib): verificable comprobando contenido y orden de
secciones. Los gráficos usan Plotly desde CDN.
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

# Umbrales de referencia (Parte IV del protocolo) para la leyenda del informe.
_LEYENDA_UMBRALES = (
    ("DSR (deflactado)", "> 0.95", "0.90 – 0.95", "< 0.90"),
    ("PBO", "< 0.20", "0.20 – 0.50", "> 0.50"),
    ("Sharpe OOS / IS", "≥ 0.70", "0.50 – 0.70", "< 0.50"),
    ("Distribución Sharpe OOS", "p25 > 0", "mediana > 0", "mediana ≤ 0"),
    ("Nº de trades", "≥ 100", "30 – 100", "< 30"),
    ("WFA efficiency", "> 0.60", "0.50 – 0.60", "< 0.50"),
)


def generar_informe_institucional(datos: dict, *, ruta_salida=None) -> str:
    """Construye el HTML del informe y, opcionalmente, lo escribe en disco.

    `datos` es un dict con claves opcionales (cada sección se renderiza solo si
    está presente): ``cabecera``, ``preregistro``, ``oos``, ``veredicto``,
    ``robustez``, ``is``, ``equity``. Devuelve el HTML como string.
    """
    resumen = _resumen_ejecutivo(datos.get("veredicto"), datos.get("oos"), datos.get("is"))
    secciones = [
        _seccion_pre_registro(datos.get("preregistro")),
        _seccion_oos(datos.get("oos")),
        _seccion_veredicto(datos.get("veredicto")),
        _seccion_robustez(datos.get("robustez")),
        _seccion_optimizacion_is(datos.get("is")),
        _seccion_equity(datos.get("equity")),
    ]
    cuerpo = resumen + "\n" + "\n".join(s for s in secciones if s)
    htmls = _documento(datos.get("cabecera", {}), cuerpo)

    if ruta_salida is not None:
        ruta = Path(ruta_salida)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(htmls, encoding="utf-8")
    return htmls


# ---------------------------------------------------------------------------
# Resumen ejecutivo (cabecera)
# ---------------------------------------------------------------------------

def _resumen_ejecutivo(ver: dict | None, oos: dict | None, is_: dict | None) -> str:
    color = str((ver or {}).get("color", "ambar"))
    c = _COLORES.get(color, "#888")
    dist = (oos or {}).get("distribucion", {})
    titulares = [
        ("Veredicto", f"{_EMOJI.get(color, '')} {color.upper()}", c),
        ("DSR", _fmt((ver or {}).get("dsr")), None),
        ("PBO", _fmt((ver or {}).get("pbo")), None),
        ("Sharpe OOS (mediana)", _fmt(dist.get("mediana")), None),
        ("Sharpe OOS / IS", _fmt((oos or {}).get("ratio_oos_is")), None),
        ("WFA efficiency", _fmt((oos or {}).get("wfa_efficiency")), None),
    ]
    tarjetas = "".join(
        f"<div class='kpi'><div class='kpi-v' style='{('color:' + col) if col else ''}'>{v}</div>"
        f"<div class='kpi-k'>{html.escape(k)}</div></div>"
        for k, v, col in titulares
    )
    return (
        f"<section id='resumen' class='resumen' style='border-left:6px solid {c}'>"
        f"<div class='resumen-h'>Resumen ejecutivo</div>"
        f"<div class='kpis'>{tarjetas}</div></section>"
    )


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
    histograma = _histograma_oos(oos.get("valores"))
    return _bloque(
        "resultados-oos",
        "2 · Resultados fuera de muestra (TITULAR)",
        "Lo primero que ves: la distribución OOS de CPCV, no un número in-sample.",
        f"<div class='cards'>{tarjetas}</div>{histograma}",
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
        titular + f"<div class='cards'>{tarjetas}</div>" + tabla + _leyenda_umbrales(),
    )


def _seccion_robustez(rob: dict | None) -> str:
    if not rob:
        return ""
    partes = []
    boot = rob.get("bootstrap")
    if boot:
        etiquetas = {
            "iteraciones": "Iteraciones",
            "p5_equity_final": "Equity final p5",
            "p25_equity_final": "Equity final p25",
            "mediana_equity_final": "Equity final mediana",
            "p95_equity_final": "Equity final p95",
            "mediana_max_drawdown": "Max DD mediana",
            "p95_max_drawdown": "Max DD p95",
            "mediana_sharpe": "Sharpe mediana",
        }
        items = "".join(
            f"<tr><th>{html.escape(etiquetas.get(k, str(k)))}</th><td>{_fmt(v)}</td></tr>"
            for k, v in boot.items()
        )
        partes.append(
            "<h3>Bootstrap de trades</h3>"
            "<p class='nota'>Distribución del resultado remuestreando la secuencia de trades: "
            "¿cuánto fue habilidad y cuánto el orden afortunado?</p>"
            f"<table class='kv'>{items}</table>"
        )
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
    etiquetas = {
        "mejor_score": "Mejor score (IS)",
        "sharpe_is": "Sharpe IS",
        "total_trades_is": "Nº trades (IS)",
    }
    items = "".join(
        f"<tr><th>{html.escape(etiquetas.get(k, str(k)))}</th><td>{_fmt(v)}</td></tr>"
        for k, v in is_.items()
    )
    return _bloque(
        "optimizacion-is",
        "5 · Optimización in-sample (contexto)",
        "Al final y como contexto, NO como conclusión.",
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
        "text:'HOLDOUT BLOQUEADO',showarrow:false,font:{color:'#d73027'}}]:[]},"
        "{displayModeBar:false,responsive:true});}})();</script>"
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
# Gráficos y bloques auxiliares
# ---------------------------------------------------------------------------

def _histograma_oos(valores) -> str:
    if not valores:
        return ""
    datos_js = json.dumps([float(v) for v in valores])
    return (
        "<div id='oos-hist' style='height:300px'></div>"
        "<script>(function(){var v=" + datos_js + ";"
        "if(window.Plotly){Plotly.newPlot('oos-hist',"
        "[{x:v,type:'histogram',marker:{color:'#1a9850'},nbinsx:12}],"
        "{margin:{t:10},xaxis:{title:'Sharpe OOS'},yaxis:{title:'Trayectorias'},"
        "shapes:[{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,"
        "line:{color:'#d73027',width:1,dash:'dot'}}]},"
        "{displayModeBar:false,responsive:true});}})();</script>"
        "<p class='nota'>Distribución de Sharpe OOS de las trayectorias CPCV "
        "(la línea roja marca el cero).</p>"
    )


def _leyenda_umbrales() -> str:
    filas = "".join(
        f"<tr><td>{html.escape(m)}</td>"
        f"<td style='color:{_COLORES['verde']}'>{html.escape(v)}</td>"
        f"<td style='color:{_COLORES['ambar']}'>{html.escape(a)}</td>"
        f"<td style='color:{_COLORES['rojo']}'>{html.escape(r)}</td></tr>"
        for (m, v, a, r) in _LEYENDA_UMBRALES
    )
    return (
        "<details class='leyenda'><summary>Umbrales de referencia (fijados a priori)</summary>"
        "<table class='grid'><thead><tr><th>Métrica</th><th>🟢 Verde</th>"
        f"<th>🟡 Ámbar</th><th>🔴 Rojo</th></tr></thead><tbody>{filas}</tbody></table></details>"
    )


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
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1040px;margin:0 auto;padding:24px;line-height:1.5;color:#1a1a1a;background:#fafafa}}
h1{{margin:0 0 4px;font-size:22px}}
.meta{{color:#555;font-size:13px;margin-bottom:20px}}
section{{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:20px;margin-bottom:18px}}
.resumen{{background:#0f172a;color:#fff;border:none}}
.resumen-h{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#94a3b8;margin-bottom:12px}}
.kpis{{display:flex;flex-wrap:wrap;gap:14px}}
.kpi{{flex:1 1 130px;text-align:center}}
.kpi-v{{font-size:22px;font-weight:800}}
.kpi-k{{font-size:11px;color:#cbd5e1;text-transform:uppercase;letter-spacing:.04em;margin-top:2px}}
h2{{margin:0 0 2px;font-size:17px}}
.sub{{margin:0 0 14px;color:#666;font-size:13px}}
h3{{font-size:14px;margin:18px 0 6px}}
.cards{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{flex:1 1 110px;background:#f4f7fb;border-radius:8px;padding:12px;text-align:center}}
.card-v{{font-size:19px;font-weight:700}}
.card-k{{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.04em}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
.kv th{{text-align:left;width:42%;color:#555;font-weight:600;padding:6px 8px;border-bottom:1px solid #eee}}
.kv td{{padding:6px 8px;border-bottom:1px solid #eee}}
.grid th,.grid td{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}}
.grid th{{color:#555;font-weight:600}}
.veredicto{{border:2px solid #888;border-radius:8px;padding:14px;text-align:center;margin-bottom:12px}}
.vbig{{font-size:24px;font-weight:800}}
.leyenda{{margin-top:14px;font-size:12px}}
.leyenda summary{{cursor:pointer;color:#555;font-weight:600}}
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

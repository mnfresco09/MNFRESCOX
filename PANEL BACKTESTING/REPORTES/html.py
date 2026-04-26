from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path

import polars as pl

from REPORTES.persistencia import slug


def generar_htmls(
    *,
    run_dir: Path,
    df: pl.DataFrame,
    trials: list,
    max_plots: int,
    grafica_rango: str,
    grafica_desde: str,
    grafica_hasta: str,
) -> list[Path]:
    if max_plots <= 0:
        return []

    html_dir = run_dir / "html"
    html_dir.mkdir(exist_ok=True)
    mejores = sorted(trials, key=lambda t: t.score, reverse=True)[:max_plots]
    paths = []

    for trial in mejores:
        path = html_dir / f"trial_{trial.numero:04d}_{slug(trial.salida.tipo)}.html"
        payload = _crear_payload(
            df=df,
            trial=trial,
            grafica_rango=grafica_rango,
            grafica_desde=grafica_desde,
            grafica_hasta=grafica_hasta,
        )
        path.write_text(_render_html(payload), encoding="utf-8")
        paths.append(path)

    verificar_htmls(paths)
    return paths


def verificar_htmls(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise ValueError(f"[HTML] No se genero {path}.")
        contenido = path.read_text(encoding="utf-8")
        for token in ("candles", "markers", "<canvas", "tooltip"):
            if token not in contenido:
                raise ValueError(f"[HTML] {path.name} no contiene bloque requerido: {token}.")


def _crear_payload(
    *,
    df: pl.DataFrame,
    trial,
    grafica_rango: str,
    grafica_desde: str,
    grafica_hasta: str,
) -> dict:
    df_idx = df.with_row_index("idx")
    df_plot = _filtrar_rango(df_idx, grafica_rango, grafica_desde, grafica_hasta)
    if df_plot.is_empty():
        raise ValueError("[HTML] El rango de grafica no contiene velas.")

    idx_min = int(df_plot["idx"][0])
    idx_max = int(df_plot["idx"][-1])
    rsi = _rsi_series(df, trial.parametros.get("rsi_periodo"))

    candles = []
    for row in df_plot.select(["idx", "timestamp", "open", "high", "low", "close"]).iter_rows(named=True):
        idx = int(row["idx"])
        candles.append(
            {
                "idx": idx,
                "t": str(row["timestamp"]),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
                "rsi": None if rsi is None or rsi[idx] is None else float(rsi[idx]),
            }
        )

    markers = []
    for n, trade in enumerate(trial.resultado.trades, start=1):
        idx_entrada = int(trade.idx_entrada)
        idx_salida = int(trade.idx_salida)
        if idx_min <= idx_entrada <= idx_max:
            markers.append(_marker(n, "entrada", trade, idx_entrada))
        if idx_min <= idx_salida <= idx_max:
            markers.append(_marker(n, "salida", trade, idx_salida))

    return {
        "titulo": f"{trial.activo} {trial.timeframe} | {trial.estrategia_nombre} | {trial.salida.tipo}",
        "trial": int(trial.numero),
        "score": float(trial.score),
        "metricas": trial.metricas,
        "parametros": trial.parametros,
        "candles": candles,
        "markers": markers,
    }


def _marker(n: int, tipo: str, trade, idx: int) -> dict:
    direccion = int(trade.direccion)
    precio = float(trade.precio_entrada if tipo == "entrada" else trade.precio_salida)
    return {
        "trade": n,
        "tipo": tipo,
        "idx": idx,
        "direccion": "LONG" if direccion == 1 else "SHORT",
        "precio": precio,
        "pnl": float(trade.pnl),
        "roi": float(trade.roi),
        "duracion": int(trade.duracion_velas),
        "motivo": str(trade.motivo_salida),
    }


def _filtrar_rango(
    df_idx: pl.DataFrame,
    grafica_rango: str,
    grafica_desde: str,
    grafica_hasta: str,
) -> pl.DataFrame:
    rango = str(grafica_rango).lower()
    if rango == "all":
        return df_idx
    if rango == "custom":
        return _filtrar_fechas(df_idx, grafica_desde, grafica_hasta)
    if rango.endswith("m") and rango[:-1].isdigit():
        meses = int(rango[:-1])
        ultimo = df_idx["timestamp"][-1]
        corte = _restar_meses(date.fromisoformat(str(ultimo)[:10]), meses).isoformat()
        return _filtrar_fechas(df_idx, corte, str(ultimo)[:10])
    raise ValueError(f"[HTML] GRAFICA_RANGO no soportado: {grafica_rango!r}.")


def _filtrar_fechas(df_idx: pl.DataFrame, desde: str, hasta: str) -> pl.DataFrame:
    inicio = pl.lit(desde).str.to_datetime(format="%Y-%m-%d", time_unit="us").dt.replace_time_zone("UTC")
    fin = pl.lit(hasta).str.to_datetime(format="%Y-%m-%d", time_unit="us").dt.replace_time_zone("UTC")
    return df_idx.filter((pl.col("timestamp") >= inicio) & (pl.col("timestamp") <= fin))


def _restar_meses(fecha: date, meses: int) -> date:
    mes = fecha.month - meses
    anno = fecha.year
    while mes <= 0:
        mes += 12
        anno -= 1
    dia = min(fecha.day, 28)
    return date(anno, mes, dia)


def _rsi_series(df: pl.DataFrame, periodo) -> list[float | None] | None:
    if periodo is None:
        return None
    periodo = int(periodo)
    delta = df["close"].diff()
    ganancia = delta.clip(lower_bound=0)
    perdida = (-delta).clip(lower_bound=0)
    alpha = 1.0 / periodo
    media_gan = ganancia.ewm_mean(alpha=alpha, adjust=False)
    media_per = perdida.ewm_mean(alpha=alpha, adjust=False)
    rsi = 100.0 - (100.0 / (1.0 + (media_gan / media_per)))
    mascara = pl.Series([None] * periodo + [1] * (len(rsi) - periodo), dtype=pl.Float64)
    return (rsi * mascara).to_list()


def _render_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    titulo = html.escape(payload["titulo"])
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{titulo} trial {payload["trial"]}</title>
<style>
body{{margin:0;font-family:Arial,sans-serif;background:#f6f7f9;color:#172033}}
header{{padding:14px 18px;background:#172033;color:white}}
h1{{font-size:18px;margin:0 0 8px 0;font-weight:700}}
.meta{{display:flex;gap:14px;flex-wrap:wrap;font-size:13px}}
main{{padding:14px}}
.chart{{position:relative;background:white;border:1px solid #d9dde5;border-radius:6px;padding:10px}}
canvas{{display:block;width:100%;height:520px}}
#rsi{{height:180px;margin-top:8px}}
#tooltip{{position:absolute;display:none;pointer-events:none;background:#172033;color:white;padding:8px;border-radius:4px;font-size:12px;max-width:260px;z-index:3}}
</style>
</head>
<body>
<header>
<h1>{titulo}</h1>
<div class="meta" id="meta"></div>
</header>
<main>
<div class="chart">
<canvas id="price"></canvas>
<canvas id="rsi"></canvas>
<div id="tooltip"></div>
</div>
</main>
<script>
const payload={data};
const candles=payload.candles;
const markers=payload.markers;
const price=document.getElementById('price');
const rsiCanvas=document.getElementById('rsi');
const tooltip=document.getElementById('tooltip');
document.getElementById('meta').innerHTML=[
`Trial ${{payload.trial}}`,
`Score ${{payload.score.toFixed(6)}}`,
`ROI ${{(payload.metricas.roi_total*100).toFixed(2)}}%`,
`Win ${{(payload.metricas.win_rate*100).toFixed(2)}}%`,
`DD ${{(payload.metricas.max_drawdown*100).toFixed(2)}}%`,
`Trades ${{payload.metricas.total_trades}}`
].map(x=>`<span>${{x}}</span>`).join('');

let markerPixels=[];
function resizeCanvas(canvas,h) {{
  const ratio=window.devicePixelRatio||1;
  canvas.width=Math.floor(canvas.clientWidth*ratio);
  canvas.height=Math.floor(h*ratio);
  canvas.style.height=h+'px';
  const ctx=canvas.getContext('2d');
  ctx.setTransform(ratio,0,0,ratio,0,0);
  return ctx;
}}
function draw() {{
  const w=price.clientWidth;
  const priceH=520;
  const rsiH=180;
  const ctx=resizeCanvas(price,priceH);
  const rctx=resizeCanvas(rsiCanvas,rsiH);
  ctx.clearRect(0,0,w,priceH);
  rctx.clearRect(0,0,w,rsiH);
  markerPixels=[];
  if(!candles.length) return;
  const hi=Math.max(...candles.map(c=>c.h));
  const lo=Math.min(...candles.map(c=>c.l));
  const pad=18;
  const xStep=Math.max(1,w/candles.length);
  const cw=Math.max(1,Math.min(8,xStep*0.65));
  const y=v=>pad+(hi-v)/(hi-lo||1)*(priceH-pad*2);
  ctx.strokeStyle='#d5dae3';
  ctx.lineWidth=1;
  for(let i=0;i<5;i++) {{
    const yy=pad+i*(priceH-pad*2)/4;
    ctx.beginPath();ctx.moveTo(0,yy);ctx.lineTo(w,yy);ctx.stroke();
  }}
  candles.forEach((c,i)=>{{
    const x=i*xStep+xStep/2;
    const up=c.c>=c.o;
    ctx.strokeStyle=up?'#16865a':'#c2412d';
    ctx.fillStyle=ctx.strokeStyle;
    ctx.beginPath();ctx.moveTo(x,y(c.h));ctx.lineTo(x,y(c.l));ctx.stroke();
    const top=y(Math.max(c.o,c.c));
    const bot=y(Math.min(c.o,c.c));
    ctx.fillRect(x-cw/2,top,cw,Math.max(1,bot-top));
  }});
  const idxToX=new Map(candles.map((c,i)=>[c.idx,i*xStep+xStep/2]));
  markers.forEach(m=>{{
    if(!idxToX.has(m.idx)) return;
    const x=idxToX.get(m.idx);
    const yy=y(m.precio);
    const color=m.tipo==='entrada'?(m.direccion==='LONG'?'#2563eb':'#f97316'):(m.pnl>=0?'#16a34a':'#dc2626');
    ctx.fillStyle=color;ctx.strokeStyle=color;
    ctx.beginPath();
    if(m.tipo==='entrada') {{
      ctx.arc(x,yy,4,0,Math.PI*2);
      ctx.fill();
    }} else {{
      ctx.moveTo(x-5,yy-5);ctx.lineTo(x+5,yy+5);ctx.moveTo(x+5,yy-5);ctx.lineTo(x-5,yy+5);ctx.stroke();
    }}
    markerPixels.push({{x:x,y:yy+price.offsetTop,m:m}});
  }});
  drawRsi(rctx,w,rsiH,xStep);
}}
function drawRsi(ctx,w,h,xStep) {{
  ctx.strokeStyle='#d5dae3';
  [30,50,70].forEach(level=>{{
    const y=10+(100-level)/100*(h-20);
    ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();
  }});
  ctx.strokeStyle='#4f46e5';
  ctx.beginPath();
  let started=false;
  candles.forEach((c,i)=>{{
    if(c.rsi===null) return;
    const x=i*xStep+xStep/2;
    const y=10+(100-c.rsi)/100*(h-20);
    if(!started){{ctx.moveTo(x,y);started=true;}} else ctx.lineTo(x,y);
  }});
  ctx.stroke();
}}
price.addEventListener('mousemove',ev=>{{
  const rect=price.getBoundingClientRect();
  const x=ev.clientX-rect.left;
  const y=ev.clientY-rect.top;
  let best=null,dist=999;
  markerPixels.forEach(p=>{{
    const d=Math.hypot(p.x-x,p.y-y);
    if(d<dist){{dist=d;best=p;}}
  }});
  if(best&&dist<12) {{
    const m=best.m;
    tooltip.style.display='block';
    tooltip.style.left=(best.x+16)+'px';
    tooltip.style.top=(best.y+8)+'px';
    tooltip.innerHTML=`<b>${{m.tipo}} ${{m.direccion}}</b><br>Trade ${{m.trade}}<br>Precio ${{m.precio.toFixed(2)}}<br>PNL ${{m.pnl.toFixed(2)}}<br>ROI ${{(m.roi*100).toFixed(2)}}%<br>Duracion ${{m.duracion}} velas<br>Motivo ${{m.motivo}}`;
  }} else tooltip.style.display='none';
}});
window.addEventListener('resize',draw);
draw();
</script>
</body>
</html>"""

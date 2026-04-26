from __future__ import annotations

import html as _html
import json
from datetime import date
from pathlib import Path

import polars as pl

from REPORTES.persistencia import slug
from REPORTES.tv_library import obtener_script_libreria


def generar_htmls(
    *,
    run_dir: Path,
    df: pl.DataFrame,
    trials: list,
    estrategia,
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
    tv_script = obtener_script_libreria()

    df_idx = df.with_row_index("_i_")
    idx_to_time: dict[int, int] = {
        int(row["_i_"]): int(row["timestamp"].timestamp())
        for row in df_idx.select(["_i_", "timestamp"]).iter_rows(named=True)
    }

    paths = []
    for trial in mejores:
        path = html_dir / f"trial_{trial.numero:04d}_{slug(trial.salida.tipo)}.html"
        payload = _crear_payload(
            df=df,
            df_idx=df_idx,
            idx_to_time=idx_to_time,
            trial=trial,
            estrategia=estrategia,
            grafica_rango=grafica_rango,
            grafica_desde=grafica_desde,
            grafica_hasta=grafica_hasta,
        )
        path.write_text(_render_html(payload, tv_script), encoding="utf-8")
        paths.append(path)

    verificar_htmls(paths)
    return paths


def verificar_htmls(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise ValueError(f"[HTML] No se genero {path}.")
        contenido = path.read_text(encoding="utf-8")
        for token in ("candles", "markers", "createChart", "addCandlestickSeries"):
            if token not in contenido:
                raise ValueError(f"[HTML] {path.name} no contiene bloque requerido: {token}.")


def _crear_payload(
    *,
    df: pl.DataFrame,
    df_idx: pl.DataFrame,
    idx_to_time: dict[int, int],
    trial,
    estrategia,
    grafica_rango: str,
    grafica_desde: str,
    grafica_hasta: str,
) -> dict:
    df_plot = _filtrar_rango(df_idx, grafica_rango, grafica_desde, grafica_hasta)
    if df_plot.is_empty():
        raise ValueError("[HTML] El rango de grafica no contiene velas.")

    ts_min = int(df_plot["timestamp"][0].timestamp())
    ts_max = int(df_plot["timestamp"][-1].timestamp())

    candles = [
        {
            "time": int(row["timestamp"].timestamp()),
            "open":  float(row["open"]),
            "high":  float(row["high"]),
            "low":   float(row["low"]),
            "close": float(row["close"]),
        }
        for row in df_plot.select(["timestamp", "open", "high", "low", "close"]).iter_rows(named=True)
    ]

    # Indicadores: calculados sobre el df completo, filtrados al rango visible
    indicadores_raw = estrategia.indicadores_para_grafica(df, trial.parametros)
    indicadores = []
    for ind in indicadores_raw:
        data_rango = [d for d in ind["data"] if ts_min <= d["t"] <= ts_max]
        indicadores.append({**ind, "data": data_rango})

    # Trades y markers
    trades = []
    markers = []
    for n, trade in enumerate(trial.resultado.trades, start=1):
        idx_e = int(trade.idx_entrada)
        idx_s = int(trade.idx_salida)
        t_e = idx_to_time.get(idx_e)
        t_s = idx_to_time.get(idx_s)
        direccion = "LONG" if int(trade.direccion) == 1 else "SHORT"
        pnl = float(trade.pnl)
        roi = float(trade.roi)

        trades.append({
            "n":             n,
            "direccion":     direccion,
            "time_entrada":  t_e,
            "time_salida":   t_s,
            "precio_entrada": float(trade.precio_entrada),
            "precio_salida":  float(trade.precio_salida),
            "pnl":     pnl,
            "roi":     roi,
            "duracion": int(trade.duracion_velas),
            "motivo":  str(trade.motivo_salida),
        })

        if t_e and ts_min <= t_e <= ts_max:
            markers.append({
                "trade":     n,
                "tipo":      "entrada",
                "time":      t_e,
                "direccion": direccion,
                "precio":    float(trade.precio_entrada),
                "pnl":       pnl,
                "roi":       roi,
                "duracion":  int(trade.duracion_velas),
                "motivo":    str(trade.motivo_salida),
            })
        if t_s and ts_min <= t_s <= ts_max:
            markers.append({
                "trade":     n,
                "tipo":      "salida",
                "time":      t_s,
                "direccion": direccion,
                "precio":    float(trade.precio_salida),
                "pnl":       pnl,
                "roi":       roi,
                "duracion":  int(trade.duracion_velas),
                "motivo":    str(trade.motivo_salida),
            })

    return {
        "titulo":      f"{trial.activo} {trial.timeframe} | {trial.estrategia_nombre} | {trial.salida.tipo}",
        "trial":       int(trial.numero),
        "score":       float(trial.score),
        "metricas":    trial.metricas,
        "parametros":  trial.parametros,
        "candles":     candles,
        "markers":     markers,
        "trades":      trades,
        "indicadores": indicadores,
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
    fin    = pl.lit(hasta).str.to_datetime(format="%Y-%m-%d", time_unit="us").dt.replace_time_zone("UTC")
    return df_idx.filter((pl.col("timestamp") >= inicio) & (pl.col("timestamp") <= fin))


def _restar_meses(fecha: date, meses: int) -> date:
    mes = fecha.month - meses
    anno = fecha.year
    while mes <= 0:
        mes += 12
        anno -= 1
    return date(anno, mes, min(fecha.day, 28))


def _render_html(payload: dict, tv_script: str) -> str:
    titulo = _html.escape(payload["titulo"])
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{titulo} — Trial {payload['trial']}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#d1d4dc;font-size:13px}}
header{{padding:12px 18px;background:#131722;border-bottom:1px solid #2a2e39;display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
#titulo{{font-size:15px;font-weight:700;color:#fff;flex:1;min-width:200px}}
#trial-badge{{background:#1e2130;color:#818cf8;font-size:11px;font-weight:600;padding:3px 8px;border-radius:12px;white-space:nowrap}}
#metrics{{display:flex;gap:14px;flex-wrap:wrap;align-items:center}}
.metric{{display:flex;flex-direction:column;gap:1px}}
.metric-label{{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.metric-val{{font-size:13px;font-weight:600;color:#d1d4dc}}
.pos{{color:#26a69a!important}}
.neg{{color:#ef5350!important}}
.long{{color:#26a69a}}
.short{{color:#ef5350}}
#app{{display:flex;flex-direction:column;gap:0}}
#charts{{display:flex;flex-direction:column;background:#131722}}
#chart-price{{position:relative}}
.pane-wrapper{{position:relative;border-top:1px solid #1e2130}}
.pane-label{{position:absolute;top:6px;left:10px;z-index:5;font-size:11px;color:#818cf8;background:#131722cc;padding:2px 6px;border-radius:3px;pointer-events:none;font-weight:600}}
#tooltip{{position:absolute;display:none;pointer-events:none;background:#1a1d29;border:1px solid #2a2e39;border-radius:6px;padding:10px 13px;font-size:12px;line-height:1.6;max-width:250px;z-index:200;box-shadow:0 8px 24px rgba(0,0,0,.6)}}
.tt-head{{font-weight:700;font-size:13px;margin-bottom:4px}}
.tt-row{{display:flex;justify-content:space-between;gap:12px;color:#9099a8}}
.tt-row b{{color:#d1d4dc}}
.tt-sep{{border:none;border-top:1px solid #2a2e39;margin:6px 0}}
#params-bar{{padding:8px 18px;background:#0d1117;border-bottom:1px solid #1e2130;display:flex;gap:10px;flex-wrap:wrap}}
.param-chip{{background:#1a1d29;border:1px solid #2a2e39;border-radius:4px;padding:3px 8px;font-size:11px;color:#9099a8}}
.param-chip b{{color:#d1d4dc}}
#trade-section{{padding:16px 18px 24px}}
#trade-section h2{{font-size:13px;font-weight:700;color:#9099a8;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px}}
#trade-table{{width:100%;border-collapse:collapse;font-size:12px}}
#trade-table th{{background:#1a1d29;color:#64748b;font-weight:600;padding:7px 10px;text-align:left;border-bottom:1px solid #2a2e39;white-space:nowrap}}
#trade-table td{{padding:5px 10px;border-bottom:1px solid #141720;white-space:nowrap}}
#trade-table tr.win:hover td{{background:#162620}}
#trade-table tr.loss:hover td{{background:#261520}}
</style>
</head>
<body>
<header>
  <div id="titulo">{titulo}</div>
  <div id="trial-badge">Trial {payload['trial']}</div>
  <div id="metrics"></div>
</header>
<div id="params-bar"></div>
<div id="app">
  <div id="charts">
    <div id="chart-price"></div>
  </div>
  <div id="trade-section">
    <h2>Operaciones</h2>
    <table id="trade-table">
      <thead>
        <tr>
          <th>#</th><th>Dir</th>
          <th>Entrada</th><th>P.Entrada</th>
          <th>Salida</th><th>P.Salida</th>
          <th>PnL ($)</th><th>ROI</th>
          <th>Velas</th><th>Motivo</th>
        </tr>
      </thead>
      <tbody id="trade-tbody"></tbody>
    </table>
  </div>
</div>
<div id="tooltip"></div>
{tv_script}
<script>
(function(){{
'use strict';
const D={data_json};

// ── Métricas ─────────────────────────────────────────────────────────────────
const metricDefs=[
  ['Score',       D.score,                  v=>v.toFixed(6),           false],
  ['ROI',         D.metricas.roi_total,     v=>(v*100).toFixed(2)+'%', true],
  ['Win Rate',    D.metricas.win_rate,      v=>(v*100).toFixed(2)+'%', true],
  ['PF',          D.metricas.profit_factor, v=>v.toFixed(3),           true],
  ['Sharpe',      D.metricas.sharpe_ratio,  v=>v.toFixed(3),           true],
  ['Max DD',      D.metricas.max_drawdown,  v=>(v*100).toFixed(2)+'%', true],
  ['Trades',      D.metricas.total_trades,  v=>v.toLocaleString(),     false],
];
const metricsEl=document.getElementById('metrics');
metricDefs.forEach(([label,val,fmt,signed])=>{{
  const div=document.createElement('div');
  div.className='metric';
  const cls=signed?(val<0?'neg':'pos'):'';
  div.innerHTML=`<span class="metric-label">${{label}}</span><span class="metric-val ${{cls}}">${{fmt(val)}}</span>`;
  metricsEl.appendChild(div);
}});

// ── Parámetros ────────────────────────────────────────────────────────────────
const paramsBar=document.getElementById('params-bar');
Object.entries(D.parametros||{{}}).forEach(([k,v])=>{{
  const chip=document.createElement('div');
  chip.className='param-chip';
  chip.innerHTML=`<b>${{k}}</b> ${{v}}`;
  paramsBar.appendChild(chip);
}});

// ── Configuración de charts ───────────────────────────────────────────────────
const chartsEl=document.getElementById('charts');
const allCharts=[];
let isSyncing=false;

function baseOpts(h){{
  return {{
    layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#787b86',fontSize:12}},
    grid:{{vertLines:{{color:'#1e2130'}},horzLines:{{color:'#1e2130'}}}},
    crosshair:{{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{{color:'#758696',labelBackgroundColor:'#2a2e39'}},horzLine:{{color:'#758696',labelBackgroundColor:'#2a2e39'}}}},
    rightPriceScale:{{borderColor:'#2a2e39',scaleMargins:{{top:0.06,bottom:0.06}}}},
    timeScale:{{borderColor:'#2a2e39',timeVisible:true,secondsVisible:false,rightOffset:10}},
    handleScroll:{{mouseWheel:true,pressedMouseMove:true}},
    handleScale:{{mouseWheel:true,pinch:true}},
    width:chartsEl.clientWidth,
    height:h,
  }};
}}

function syncTime(src){{
  src.timeScale().subscribeVisibleLogicalRangeChange(range=>{{
    if(isSyncing||!range)return;
    isSyncing=true;
    allCharts.forEach(c=>{{ if(c!==src) c.timeScale().setVisibleLogicalRange(range); }});
    isSyncing=false;
  }});
}}

// ── Gráfico principal (velas) ─────────────────────────────────────────────────
const priceEl=document.getElementById('chart-price');
const mainChart=LightweightCharts.createChart(priceEl,baseOpts(520));
allCharts.push(mainChart);
syncTime(mainChart);

const candleSeries=mainChart.addCandlestickSeries({{
  upColor:'#26a69a',downColor:'#ef5350',
  borderVisible:false,
  wickUpColor:'#26a69a',wickDownColor:'#ef5350',
}});
candleSeries.setData(D.candles);

// ── Overlays (EMAs, SMAs, Bollinger…) ─────────────────────────────────────────
(D.indicadores||[]).filter(i=>i.tipo==='overlay').forEach(ind=>{{
  const s=mainChart.addLineSeries({{
    color:ind.color,lineWidth:1.5,title:ind.nombre,
    lastValueVisible:true,priceLineVisible:false,
  }});
  s.setData(ind.data.map(d=>({{time:d.t,value:d.v}})));
}});

// ── Markers de trades ─────────────────────────────────────────────────────────
const tvMarkers=(D.markers||[]).map(m=>{{
  const isEntry=m.tipo==='entrada';
  const isLong=m.direccion==='LONG';
  return {{
    time:m.time,
    position:isEntry?(isLong?'belowBar':'aboveBar'):(isLong?'aboveBar':'belowBar'),
    color:isEntry?(isLong?'#22c55e':'#f97316'):(m.pnl>=0?'#22c55e':'#ef5350'),
    shape:isEntry?(isLong?'arrowUp':'arrowDown'):'circle',
    text:String(m.trade),
    size:1.5,
  }};
}}).sort((a,b)=>a.time-b.time);
candleSeries.setMarkers(tvMarkers);

// ── Paneles de indicadores (RSI, MACD…) ───────────────────────────────────────
(D.indicadores||[]).filter(i=>i.tipo==='pane').forEach(ind=>{{
  const wrapper=document.createElement('div');
  wrapper.className='pane-wrapper';
  chartsEl.appendChild(wrapper);

  const lbl=document.createElement('div');
  lbl.className='pane-label';
  lbl.textContent=ind.nombre;
  wrapper.appendChild(lbl);

  const paneEl=document.createElement('div');
  wrapper.appendChild(paneEl);

  const paneChart=LightweightCharts.createChart(paneEl,baseOpts(140));
  allCharts.push(paneChart);
  syncTime(paneChart);

  const paneSeries=paneChart.addLineSeries({{
    color:ind.color,lineWidth:1.5,
    lastValueVisible:true,priceLineVisible:false,
  }});
  paneSeries.setData(ind.data.map(d=>({{time:d.t,value:d.v}})));

  if(ind.min!==undefined&&ind.max!==undefined){{
    paneSeries.applyOptions({{
      autoscaleInfoProvider:()=>({{
        priceRange:{{minValue:ind.min,maxValue:ind.max}},
        margins:{{above:8,below:8}},
      }}),
    }});
  }}

  (ind.niveles||[]).forEach(n=>{{
    paneSeries.createPriceLine({{
      price:n.valor,color:n.color,lineWidth:1,
      lineStyle:LightweightCharts.LineStyle.Dashed,
      axisLabelVisible:true,
    }});
  }});
}});

// ── Tooltip ───────────────────────────────────────────────────────────────────
const tooltipEl=document.getElementById('tooltip');
const byTime={{}};
(D.markers||[]).forEach(m=>{{
  if(!byTime[m.time])byTime[m.time]=[];
  byTime[m.time].push(m);
}});

mainChart.subscribeCrosshairMove(param=>{{
  if(!param.time||!param.point){{tooltipEl.style.display='none';return;}}
  const ms=byTime[param.time];
  if(!ms||!ms.length){{tooltipEl.style.display='none';return;}}

  tooltipEl.innerHTML=ms.map((m,i)=>{{
    const pnlStr=(m.pnl>=0?'+':'')+m.pnl.toFixed(2);
    const roiStr=(m.roi>=0?'+':'')+(m.roi*100).toFixed(2)+'%';
    const pColor=m.pnl>=0?'#26a69a':'#ef5350';
    const sep=i>0?'<hr class="tt-sep">':'';
    if(m.tipo==='entrada'){{
      const dColor=m.direccion==='LONG'?'#26a69a':'#ef5350';
      return `${{sep}}<div class="tt-head" style="color:${{dColor}}">${{m.direccion}} ENTRADA #${{m.trade}}</div>
        <div class="tt-row"><span>Precio</span><b>${{m.precio.toFixed(2)}}</b></div>`;
    }}else{{
      return `${{sep}}<div class="tt-head">SALIDA #${{m.trade}} — ${{m.motivo}}</div>
        <div class="tt-row"><span>Precio</span><b>${{m.precio.toFixed(2)}}</b></div>
        <div class="tt-row"><span>PnL</span><b style="color:${{pColor}}">${{pnlStr}} (${{roiStr}})</b></div>
        <div class="tt-row"><span>Duración</span><b>${{m.duracion}} velas</b></div>`;
    }}
  }}).join('');

  tooltipEl.style.display='block';
  let lx=param.point.x+16, ly=param.point.y+8;
  const tw=tooltipEl.offsetWidth||220;
  if(lx+tw>priceEl.clientWidth-10) lx=param.point.x-tw-8;
  tooltipEl.style.left=lx+'px';
  tooltipEl.style.top=ly+'px';
}});

// ── Tabla de trades ───────────────────────────────────────────────────────────
const tbody=document.getElementById('trade-tbody');
(D.trades||[]).forEach(t=>{{
  const tr=document.createElement('tr');
  tr.className=t.pnl>=0?'win':'loss';
  const fmtTs=ts=>ts?new Date(ts*1000).toISOString().replace('T',' ').slice(0,16)+'Z':'—';
  const pnlStr=(t.pnl>=0?'+':'')+t.pnl.toFixed(2);
  const roiStr=(t.roi>=0?'+':'')+(t.roi*100).toFixed(2)+'%';
  tr.innerHTML=`
    <td>${{t.n}}</td>
    <td class="${{t.direccion.toLowerCase()}}">${{t.direccion}}</td>
    <td>${{fmtTs(t.time_entrada)}}</td>
    <td>${{t.precio_entrada.toFixed(2)}}</td>
    <td>${{fmtTs(t.time_salida)}}</td>
    <td>${{t.precio_salida.toFixed(2)}}</td>
    <td class="${{t.pnl>=0?'pos':'neg'}}">${{pnlStr}}</td>
    <td class="${{t.pnl>=0?'pos':'neg'}}">${{roiStr}}</td>
    <td>${{t.duracion}}</td>
    <td>${{t.motivo}}</td>`;
  tbody.appendChild(tr);
}});

// ── Resize responsive ─────────────────────────────────────────────────────────
function fitAll(){{
  const w=chartsEl.clientWidth;
  allCharts.forEach(c=>c.applyOptions({{width:w}}));
}}
const ro=new ResizeObserver(fitAll);
ro.observe(chartsEl);
fitAll();

}})();
</script>
</body>
</html>"""

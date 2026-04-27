from __future__ import annotations

import html as _html
import json
import math
from pathlib import Path


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def generar_informe(
    *,
    run_dir: Path,
    trials: list,
    estrategia,
    activo: str,
    timeframe: str,
    salida_tipo: str,
) -> Path:
    if not trials:
        raise ValueError("[INFORME] No hay trials para generar el informe.")

    informe_dir = _base_resultados(run_dir) / "INFORME"
    informe_dir.mkdir(parents=True, exist_ok=True)
    path = _unique_path(informe_dir / "INFORME ROBUSTEZ.html")

    payload = _crear_payload(
        trials=trials,
        estrategia=estrategia,
        activo=activo,
        timeframe=timeframe,
        salida_tipo=salida_tipo,
    )

    path.write_text(_render_html(payload), encoding="utf-8")
    _verificar_informe(path)
    return path


def _verificar_informe(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"[INFORME] No se generó {path}.")
    contenido = path.read_text(encoding="utf-8")
    for token in ("Plotly", "params_keys", "metric_keys", "trials"):
        if token not in contenido:
            raise ValueError(f"[INFORME] {path.name} no contiene bloque requerido: {token}.")


def _crear_payload(
    *,
    trials,
    estrategia,
    activo: str,
    timeframe: str,
    salida_tipo: str,
) -> dict:
    params_keys = sorted({k for t in trials for k in (t.parametros or {}).keys()})
    metric_keys = sorted(
        {k for t in trials for k, v in (t.metricas or {}).items() if _is_numeric(v)}
    )

    rows = []
    for t in trials:
        rows.append(
            {
                "trial": int(t.numero),
                "score": _safe_num(t.score),
                "params": {k: _to_jsonable((t.parametros or {}).get(k)) for k in params_keys},
                "metricas": {k: _to_jsonable((t.metricas or {}).get(k)) for k in metric_keys},
            }
        )

    mejor = max(trials, key=lambda x: x.score)

    return {
        "titulo": f"{activo} {timeframe} | {estrategia.NOMBRE} | {salida_tipo}",
        "activo": activo,
        "timeframe": timeframe,
        "estrategia": estrategia.NOMBRE,
        "salida": salida_tipo,
        "params_keys": params_keys,
        "metric_keys": metric_keys,
        "trials": rows,
        "mejor_trial": int(mejor.numero),
        "mejor_score": _safe_num(mejor.score),
        "n_trials": len(trials),
    }


def _is_numeric(v) -> bool:
    if v is None or isinstance(v, bool):
        return False
    return isinstance(v, (int, float))


def _safe_num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def _to_jsonable(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return str(v)


def _render_html(payload: dict) -> str:
    titulo = _html.escape(payload["titulo"])
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    return (
        _TEMPLATE.replace("__DATA_JSON__", data_json)
        .replace("__TITULO__", titulo)
        .replace("__PLOTLY_CDN__", PLOTLY_CDN)
    )


def _base_resultados(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    if run_dir.parent.name.upper() == "DATOS":
        return run_dir.parent.parent
    return run_dir


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{idx:02d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"[INFORME] No se pudo crear nombre único para {path}.")


_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITULO__ — Informe Robustez</title>
<script src="__PLOTLY_CDN__"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#d1d4dc;font-size:13px;min-height:100vh}
header{padding:16px 24px;background:#131722;border-bottom:1px solid #2a2e39}
.title-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
#titulo{font-size:16px;font-weight:700;color:#fff;flex:1;min-width:240px}
.badge{background:#1e2130;color:#818cf8;font-size:11px;font-weight:600;padding:4px 10px;border-radius:12px;white-space:nowrap;letter-spacing:.4px}
.meta-row{display:flex;gap:20px;flex-wrap:wrap;margin-top:10px;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
.meta-item b{color:#d1d4dc;font-weight:600;margin-left:4px;text-transform:none;letter-spacing:0}
nav{display:flex;background:#0d1117;border-bottom:1px solid #2a2e39;padding:0 24px;overflow-x:auto}
.tab{padding:13px 18px;font-size:11px;font-weight:600;color:#64748b;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:color .15s,border-color .15s;letter-spacing:.6px;text-transform:uppercase}
.tab:hover{color:#d1d4dc}
.tab.active{color:#818cf8;border-bottom-color:#818cf8}
.help{font-size:11px;color:#64748b;padding:10px 24px;background:#0d1117;border-bottom:1px solid #1e2130;line-height:1.5}
.help b{color:#818cf8;font-weight:600}
.controls{display:flex;gap:14px;flex-wrap:wrap;padding:14px 24px;background:#131722;border-bottom:1px solid #2a2e39}
.ctl{display:flex;flex-direction:column;gap:5px}
.ctl label{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;font-weight:600}
.ctl select{background:#1a1d29;color:#d1d4dc;border:1px solid #2a2e39;border-radius:4px;padding:7px 10px;font-size:12px;font-family:inherit;min-width:160px;cursor:pointer;transition:border-color .15s}
.ctl select:hover{border-color:#3f4453}
.ctl select:focus{outline:none;border-color:#818cf8}
#chart-wrapper{padding:18px 24px 0;background:#0d1117}
#chart{width:100%;min-height:600px;background:#131722;border:1px solid #2a2e39;border-radius:6px;overflow:hidden}
.stats-row{display:flex;gap:24px;flex-wrap:wrap;padding:14px 24px;background:#131722;border-top:1px solid #2a2e39;margin-top:18px;margin-left:24px;margin-right:24px;margin-bottom:24px;border-radius:0 0 6px 6px;border:1px solid #2a2e39;border-top:none}
.stat{display:flex;flex-direction:column;gap:3px;min-width:90px}
.stat-label{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.stat-val{font-size:13px;color:#d1d4dc;font-weight:600;font-variant-numeric:tabular-nums}
.pos{color:#26a69a!important}
.neg{color:#ef5350!important}
#offline{display:none;padding:24px;text-align:center;color:#ef5350;background:#131722;margin:18px 24px;border:1px solid #2a2e39;border-radius:6px}
</style>
</head>
<body>
<header>
  <div class="title-row">
    <div id="titulo">__TITULO__</div>
    <div class="badge" id="best-badge"></div>
  </div>
  <div class="meta-row" id="meta-row"></div>
</header>
<nav id="tabs"></nav>
<div class="help" id="help"></div>
<div class="controls" id="controls"></div>
<div id="chart-wrapper"><div id="chart"></div></div>
<div class="stats-row" id="stats"></div>
<div id="offline">No se pudo cargar Plotly. Comprueba la conexión a internet o usa la versión local.</div>
<script>
'use strict';
const D = __DATA_JSON__;

if (typeof Plotly === 'undefined') {
  document.getElementById('offline').style.display = 'block';
  document.getElementById('chart-wrapper').style.display = 'none';
  document.getElementById('controls').style.display = 'none';
  document.getElementById('help').style.display = 'none';
}

// ── Paleta y formato ──────────────────────────────────────────────────
const C_BG='#131722',C_PAPER='#131722',C_GRID='#1e2130',C_AXIS='#2a2e39';
const C_TEXT='#d1d4dc',C_MUTED='#9099a8',C_DIM='#64748b';
const C_ACCENT='#818cf8',C_POS='#26a69a',C_NEG='#ef5350',C_GOLD='#fbbf24';

const METRIC_LABELS={
  roi_total:'ROI',expectancy:'Expectancy',win_rate:'Win Rate',profit_factor:'Profit Factor',sharpe_ratio:'Sharpe',
  max_drawdown:'Max DD',total_trades:'Trades',trades_long:'Long',trades_short:'Short',
  trades_ganadores:'Win',trades_perdedores:'Loss',trades_neutros:'Break Even',trades_por_dia:'Trades/Dia',pnl_bruto_total:'PnL Bruto',
  pnl_total:'PnL Neto',pnl_promedio:'PnL Prom',saldo_inicial:'Saldo Ini',
  saldo_final:'Saldo Fin',duracion_media_seg:'Dur Media',parado_por_saldo:'Stop Saldo',
  score:'Score',
};

const PCT_METRICS=new Set(['roi_total','expectancy','win_rate','max_drawdown']);
const MONEY_METRICS=new Set(['pnl_bruto_total','pnl_total','pnl_promedio','saldo_inicial','saldo_final']);
const INT_METRICS=new Set(['total_trades','trades_long','trades_short','trades_ganadores','trades_perdedores','trades_neutros']);
const MIN_BETTER=new Set(['max_drawdown','parado_por_saldo','duracion_media_seg']);

function metricLabel(k){return METRIC_LABELS[k]||k.toUpperCase().replace(/_/g,' ')}
function paramLabel(k){return k.toUpperCase().replace(/_/g,' ')}
function metricDir(k){return MIN_BETTER.has(k)?'min':'max'}

function fmtMetric(k,v){
  if(v==null||isNaN(v))return '—';
  if(PCT_METRICS.has(k))return (v*100).toFixed(2)+'%';
  if(MONEY_METRICS.has(k))return '$'+v.toLocaleString(undefined,{maximumFractionDigits:2});
  if(INT_METRICS.has(k))return Math.round(v).toLocaleString();
  if(k==='score')return v.toFixed(6);
  if(k==='profit_factor'||k==='sharpe_ratio')return v.toFixed(3);
  if(typeof v==='number')return Number.isInteger(v)?String(v):v.toFixed(4);
  return String(v);
}

function trialValue(t,key){
  if(key==='score')return t.score;
  if(key==='trial')return t.trial;
  if(key in t.params)return t.params[key];
  if(key in t.metricas)return t.metricas[key];
  return null;
}
function trialKind(key){
  if(key==='score')return 'score';
  if(D.params_keys.includes(key))return 'param';
  if(D.metric_keys.includes(key))return 'metric';
  return 'unknown';
}
function axisLabel(k){return trialKind(k)==='param'?paramLabel(k):metricLabel(k)}

function aggregate(values,mode){
  const vs=values.filter(v=>v!=null&&!isNaN(v));
  if(!vs.length)return null;
  if(mode==='count')return vs.length;
  if(mode==='mean')return vs.reduce((a,b)=>a+b,0)/vs.length;
  if(mode==='median'){const s=[...vs].sort((a,b)=>a-b);const m=Math.floor(s.length/2);return s.length%2?s[m]:(s[m-1]+s[m])/2}
  if(mode==='min')return Math.min(...vs);
  if(mode==='max')return Math.max(...vs);
  if(mode==='std'){const m=vs.reduce((a,b)=>a+b,0)/vs.length;return Math.sqrt(vs.reduce((a,b)=>a+(b-m)*(b-m),0)/vs.length)}
  return null;
}
const AGG_LABELS={mean:'Media',median:'Mediana',min:'Mínimo',max:'Máximo',std:'Desv. típica',count:'Nº trials'};

function pearson(xs,ys){
  const pairs=xs.map((x,i)=>[x,ys[i]]).filter(([x,y])=>x!=null&&y!=null&&!isNaN(x)&&!isNaN(y));
  if(pairs.length<2)return NaN;
  const n=pairs.length;
  const mx=pairs.reduce((a,[x])=>a+x,0)/n;
  const my=pairs.reduce((a,[,y])=>a+y,0)/n;
  let num=0,dx=0,dy=0;
  for(const [x,y] of pairs){const a=x-mx,b=y-my;num+=a*b;dx+=a*a;dy+=b*b}
  if(dx===0||dy===0)return 0;
  return num/Math.sqrt(dx*dy);
}

// ── Header ────────────────────────────────────────────────────────────
document.getElementById('best-badge').textContent='Mejor Trial '+D.mejor_trial;
const meta=document.getElementById('meta-row');
[
  ['Activo',D.activo],['TF',D.timeframe],['Estrategia',D.estrategia],
  ['Exit',D.salida],['Trials',D.n_trials.toLocaleString()],
  ['Mejor Score',D.mejor_score.toFixed(6)],
].forEach(([k,v])=>{
  const e=document.createElement('div');
  e.className='meta-item';
  e.innerHTML=k+'<b>'+v+'</b>';
  meta.appendChild(e);
});

// ── Tabs ─────────────────────────────────────────────────────────────
const TABS=[
  {id:'heatmap',label:'Robustez 2D',
   help:'Mapa de calor: cada celda agrega la métrica seleccionada para los trials con esos parámetros. <b>Zonas robustas</b> = áreas con color uniforme. Cambia agregación a <b>Desv. típica</b> para identificar variabilidad — bajo = robusto.',
   render:renderHeatmap},
  {id:'surface',label:'Superficie 3D',
   help:'Superficie 3D de la métrica frente a 2 parámetros. Zonas de meseta (planas y altas/bajas) indican configuraciones robustas; picos aislados indican parámetros frágiles.',
   render:renderSurface},
  {id:'scatter',label:'Dispersión 3D',
   help:'Scatter 3D con cualquier combinación de ejes (parámetros o métricas). El color codifica una métrica adicional. Útil para detectar clusters de configuraciones.',
   render:renderScatter3D},
  {id:'parallel',label:'Coord. Paralelas',
   help:'Cada línea representa un trial. Permite identificar combinaciones ganadoras a través de todos los parámetros simultáneamente. Arrastra los ejes para filtrar interactivamente.',
   render:renderParallel},
  {id:'pareto',label:'Frontera Pareto',
   help:'Trade-off entre 2 métricas. Los puntos en la <b>frontera de Pareto</b> son no dominados — ningún trial es mejor en ambas dimensiones simultáneamente.',
   render:renderPareto},
  {id:'sens',label:'Sensibilidad',
   help:'Correlación de Pearson entre cada parámetro y la métrica objetivo. Parámetros con correlación |alta| son los más influyentes en el resultado; correlaciones cercanas a 0 indican parámetros poco relevantes.',
   render:renderSensitivity},
  {id:'dist',label:'Distribuciones',
   help:'Distribución de la métrica seleccionada en todos los trials. Útil para detectar outliers, sesgos del optimizador y entender la forma del espacio de soluciones.',
   render:renderDistribution},
];

const tabsEl=document.getElementById('tabs');
TABS.forEach((t,i)=>{
  const e=document.createElement('div');
  e.className='tab'+(i===0?' active':'');
  e.textContent=t.label;
  e.dataset.id=t.id;
  e.addEventListener('click',()=>switchTab(t.id));
  tabsEl.appendChild(e);
});
let currentTab=TABS[0].id;
function switchTab(id){
  if(id===currentTab)return;
  currentTab=id;
  document.querySelectorAll('.tab').forEach(el=>el.classList.toggle('active',el.dataset.id===id));
  rebuild();
}

// ── State ────────────────────────────────────────────────────────────
const PK=D.params_keys, MK=D.metric_keys;
const State={
  hm:{x:PK[0],y:PK[1]||PK[0],metric:'score',agg:'mean'},
  surface:{x:PK[0],y:PK[1]||PK[0],metric:'score',agg:'mean'},
  scatter:{x:PK[0],y:PK[1]||PK[0],z:'score',color:MK.includes('roi_total')?'roi_total':'score'},
  parallel:{color:'score'},
  pareto:{
    a:MK.includes('roi_total')?'roi_total':'score',
    b:MK.includes('max_drawdown')?'max_drawdown':(MK.find(k=>k!=='roi_total')||'score'),
  },
  dist:{metric:'score'},
  sens:{metric:'score'},
};

// ── Selectores ───────────────────────────────────────────────────────
function makeSelect(id,label,opts,def){
  const ctl=document.createElement('div');ctl.className='ctl';
  const lbl=document.createElement('label');lbl.htmlFor=id;lbl.textContent=label;
  const sel=document.createElement('select');sel.id=id;
  opts.forEach(([v,t])=>{
    const o=document.createElement('option');o.value=v;o.textContent=t;
    if(v===def)o.selected=true;sel.appendChild(o);
  });
  sel.addEventListener('change',rebuild);
  ctl.appendChild(lbl);ctl.appendChild(sel);return ctl;
}
const paramOpts=()=>PK.map(k=>[k,paramLabel(k)]);
const metricOpts=(includeScore=true)=>{
  const a=MK.map(k=>[k,metricLabel(k)]);
  if(includeScore)a.unshift(['score','SCORE']);
  return a;
};
const allAxesOpts=()=>[
  ['score','SCORE'],
  ...PK.map(k=>[k,'⚙ '+paramLabel(k)]),
  ...MK.map(k=>[k,'∑ '+metricLabel(k)]),
];

function readControls(){
  const t=currentTab;
  const v=id=>document.getElementById(id)?document.getElementById(id).value:null;
  if(t==='heatmap')State.hm={x:v('hm-x'),y:v('hm-y'),metric:v('hm-metric'),agg:v('hm-agg')};
  else if(t==='surface')State.surface={x:v('s3-x'),y:v('s3-y'),metric:v('s3-metric'),agg:v('s3-agg')};
  else if(t==='scatter')State.scatter={x:v('sc-x'),y:v('sc-y'),z:v('sc-z'),color:v('sc-color')};
  else if(t==='parallel')State.parallel={color:v('pc-color')};
  else if(t==='pareto')State.pareto={a:v('pa-a'),b:v('pa-b')};
  else if(t==='dist')State.dist={metric:v('di-metric')};
  else if(t==='sens')State.sens={metric:v('se-metric')};
}

function buildControls(){
  const c=document.getElementById('controls');c.innerHTML='';
  const aggOpts=[['mean','Media'],['median','Mediana'],['min','Mínimo'],['max','Máximo'],['std','Desv. típica (robustez)'],['count','Nº trials']];
  if(currentTab==='heatmap'){
    c.appendChild(makeSelect('hm-x','Eje X (parámetro)',paramOpts(),State.hm.x));
    c.appendChild(makeSelect('hm-y','Eje Y (parámetro)',paramOpts(),State.hm.y));
    c.appendChild(makeSelect('hm-metric','Métrica',metricOpts(),State.hm.metric));
    c.appendChild(makeSelect('hm-agg','Agregación',aggOpts,State.hm.agg));
  } else if(currentTab==='surface'){
    c.appendChild(makeSelect('s3-x','Eje X (parámetro)',paramOpts(),State.surface.x));
    c.appendChild(makeSelect('s3-y','Eje Y (parámetro)',paramOpts(),State.surface.y));
    c.appendChild(makeSelect('s3-metric','Métrica Z',metricOpts(),State.surface.metric));
    c.appendChild(makeSelect('s3-agg','Agregación',aggOpts.filter(o=>o[0]!=='count'&&o[0]!=='std'),State.surface.agg));
  } else if(currentTab==='scatter'){
    c.appendChild(makeSelect('sc-x','Eje X',allAxesOpts(),State.scatter.x));
    c.appendChild(makeSelect('sc-y','Eje Y',allAxesOpts(),State.scatter.y));
    c.appendChild(makeSelect('sc-z','Eje Z',allAxesOpts(),State.scatter.z));
    c.appendChild(makeSelect('sc-color','Color',metricOpts(),State.scatter.color));
  } else if(currentTab==='parallel'){
    c.appendChild(makeSelect('pc-color','Color (métrica)',metricOpts(),State.parallel.color));
  } else if(currentTab==='pareto'){
    c.appendChild(makeSelect('pa-a','Métrica A',metricOpts(),State.pareto.a));
    c.appendChild(makeSelect('pa-b','Métrica B',metricOpts(),State.pareto.b));
  } else if(currentTab==='dist'){
    c.appendChild(makeSelect('di-metric','Métrica',metricOpts(),State.dist.metric));
  } else if(currentTab==='sens'){
    c.appendChild(makeSelect('se-metric','Métrica objetivo',metricOpts(),State.sens.metric));
  }
}

function rebuild(){
  if(typeof Plotly==='undefined')return;
  if(document.querySelector('#controls select'))readControls();
  buildControls();
  document.getElementById('help').innerHTML=TABS.find(t=>t.id===currentTab).help;
  TABS.find(t=>t.id===currentTab).render();
}

// ── Plotly base ──────────────────────────────────────────────────────
function baseLayout(extra={}){
  return Object.assign({
    paper_bgcolor:C_PAPER,plot_bgcolor:C_BG,
    font:{family:'-apple-system,BlinkMacSystemFont,sans-serif',color:C_TEXT,size:12},
    margin:{l:70,r:30,t:50,b:60},
    hoverlabel:{bgcolor:'#1a1d29',bordercolor:'#2a2e39',font:{color:C_TEXT,size:12}},
  },extra);
}
function ax(extra={}){
  return Object.assign({
    gridcolor:C_GRID,zerolinecolor:C_AXIS,linecolor:C_AXIS,tickcolor:C_AXIS,
    color:C_MUTED,tickfont:{size:11},
  },extra);
}
const PLOT_CFG={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{format:'png',filename:'informe',height:900,width:1600,scale:2}};

const SCALE_DIVERGING=[[0,'#ef5350'],[0.5,'#1a1d29'],[1,'#26a69a']];
const SCALE_SEQ=[[0,'#1a1d29'],[0.4,'#3b3f5c'],[0.7,'#818cf8'],[1,'#26a69a']];
const SCALE_INVERTED=[[0,'#26a69a'],[0.5,'#1a1d29'],[1,'#ef5350']];

function colorscaleFor(metric,agg){
  if(agg==='std'||agg==='count')return SCALE_SEQ;
  if(metricDir(metric)==='min')return SCALE_INVERTED;
  return SCALE_DIVERGING;
}

// ── Renderers ────────────────────────────────────────────────────────
function buildCellMap(xKey,yKey,mKey){
  const cells=new Map();
  for(const t of D.trials){
    const xv=t.params[xKey],yv=t.params[yKey];
    if(xv==null||yv==null)continue;
    const mv=mKey==='score'?t.score:t.metricas[mKey];
    if(mv==null||isNaN(mv))continue;
    const k=xv+'|'+yv;
    if(!cells.has(k))cells.set(k,[]);
    cells.get(k).push(mv);
  }
  const xUnique=[...new Set(D.trials.map(t=>t.params[xKey]).filter(v=>v!=null))].sort((a,b)=>a-b);
  const yUnique=[...new Set(D.trials.map(t=>t.params[yKey]).filter(v=>v!=null))].sort((a,b)=>a-b);
  return {cells,xUnique,yUnique};
}

function renderHeatmap(){
  const {x,y,metric,agg}=State.hm;
  const {cells,xUnique,yUnique}=buildCellMap(x,y,metric);
  const z=yUnique.map(yv=>xUnique.map(xv=>{
    const arr=cells.get(xv+'|'+yv);
    return arr?aggregate(arr,agg):null;
  }));
  const flat=z.flat().filter(v=>v!=null&&!isNaN(v));

  const trace={
    type:'heatmap',
    x:xUnique,y:yUnique,z:z,
    colorscale:colorscaleFor(metric,agg),
    showscale:true,
    hoverongaps:false,
    colorbar:{
      title:{text:AGG_LABELS[agg]+' '+metricLabel(metric),font:{size:11,color:C_MUTED}},
      tickcolor:C_AXIS,tickfont:{color:C_MUTED,size:11},outlinecolor:C_AXIS,
      thickness:14,len:0.85,
    },
    hovertemplate:paramLabel(x)+': %{x}<br>'+paramLabel(y)+': %{y}<br>'+
                  AGG_LABELS[agg]+' '+metricLabel(metric)+': %{z:.4f}<extra></extra>',
  };

  Plotly.newPlot('chart',[trace],baseLayout({
    title:{text:AGG_LABELS[agg]+' de '+metricLabel(metric)+' — '+paramLabel(x)+' × '+paramLabel(y),
           font:{size:13,color:C_TEXT},x:0.02,xanchor:'left'},
    xaxis:ax({title:{text:paramLabel(x),font:{color:C_MUTED}}}),
    yaxis:ax({title:{text:paramLabel(y),font:{color:C_MUTED}}}),
    height:620,margin:{l:90,r:30,t:60,b:70},
  }),PLOT_CFG);

  if(!flat.length){renderStats([['Sin datos','—']]);return}
  const stats=[['Celdas con datos',String(flat.length)+' / '+(xUnique.length*yUnique.length)]];
  if(agg==='count'){
    stats.push(['Mín por celda',String(Math.min(...flat))]);
    stats.push(['Máx por celda',String(Math.max(...flat))]);
    stats.push(['Media por celda',(flat.reduce((a,b)=>a+b,0)/flat.length).toFixed(1)]);
  } else {
    const fmt=v=>fmtMetric(metric,v);
    stats.push(['Mín celda',fmt(Math.min(...flat))]);
    stats.push(['Máx celda',fmt(Math.max(...flat))]);
    stats.push(['Media celdas',fmt(flat.reduce((a,b)=>a+b,0)/flat.length)]);
  }
  renderStats(stats);
}

function renderSurface(){
  const {x,y,metric,agg}=State.surface;
  const {cells,xUnique,yUnique}=buildCellMap(x,y,metric);
  const z=yUnique.map(yv=>xUnique.map(xv=>{
    const arr=cells.get(xv+'|'+yv);
    return arr?aggregate(arr,agg):null;
  }));

  const trace={
    type:'surface',
    x:xUnique,y:yUnique,z:z,
    colorscale:colorscaleFor(metric,agg),
    showscale:true,
    contours:{z:{show:true,usecolormap:true,highlightcolor:'#ffffff',project:{z:true},width:1}},
    colorbar:{title:{text:metricLabel(metric),font:{color:C_MUTED}},tickfont:{color:C_MUTED},thickness:14,len:0.7},
    hovertemplate:paramLabel(x)+': %{x}<br>'+paramLabel(y)+': %{y}<br>'+metricLabel(metric)+': %{z:.4f}<extra></extra>',
  };

  Plotly.newPlot('chart',[trace],baseLayout({
    title:{text:AGG_LABELS[agg]+' de '+metricLabel(metric),font:{size:13,color:C_TEXT},x:0.02,xanchor:'left'},
    scene:{
      xaxis:{title:paramLabel(x),color:C_MUTED,gridcolor:C_GRID,backgroundcolor:C_BG,showbackground:true},
      yaxis:{title:paramLabel(y),color:C_MUTED,gridcolor:C_GRID,backgroundcolor:C_BG,showbackground:true},
      zaxis:{title:metricLabel(metric),color:C_MUTED,gridcolor:C_GRID,backgroundcolor:C_BG,showbackground:true},
      bgcolor:C_BG,
    },
    height:680,
  }),PLOT_CFG);
  renderStats([]);
}

function renderScatter3D(){
  const {x,y,z,color}=State.scatter;
  const xs=D.trials.map(t=>trialValue(t,x));
  const ys=D.trials.map(t=>trialValue(t,y));
  const zs=D.trials.map(t=>trialValue(t,z));
  const cs=D.trials.map(t=>trialValue(t,color));

  const trace={
    type:'scatter3d',mode:'markers',
    x:xs,y:ys,z:zs,
    text:D.trials.map(t=>'Trial '+t.trial),
    marker:{
      size:4,color:cs,
      colorscale:metricDir(color)==='min'?SCALE_INVERTED:SCALE_DIVERGING,
      showscale:true,
      colorbar:{title:{text:axisLabel(color),font:{color:C_MUTED}},tickfont:{color:C_MUTED},thickness:14,len:0.7},
      line:{width:0},opacity:0.85,
    },
    hovertemplate:'%{text}<br>'+axisLabel(x)+': %{x}<br>'+axisLabel(y)+': %{y}<br>'+axisLabel(z)+': %{z}<br>'+axisLabel(color)+': %{marker.color:.4f}<extra></extra>',
  };

  const traces=[trace];
  const best=D.trials.find(t=>t.trial===D.mejor_trial);
  if(best){
    traces.push({
      type:'scatter3d',mode:'markers',
      x:[trialValue(best,x)],y:[trialValue(best,y)],z:[trialValue(best,z)],
      marker:{size:11,color:C_GOLD,symbol:'diamond',line:{color:'#fff',width:1.5}},
      text:['Mejor — Trial '+best.trial],
      hovertemplate:'%{text}<extra></extra>',
      showlegend:false,
    });
  }

  Plotly.newPlot('chart',traces,baseLayout({
    scene:{
      xaxis:{title:axisLabel(x),color:C_MUTED,gridcolor:C_GRID,backgroundcolor:C_BG,showbackground:true},
      yaxis:{title:axisLabel(y),color:C_MUTED,gridcolor:C_GRID,backgroundcolor:C_BG,showbackground:true},
      zaxis:{title:axisLabel(z),color:C_MUTED,gridcolor:C_GRID,backgroundcolor:C_BG,showbackground:true},
      bgcolor:C_BG,
    },
    height:680,showlegend:false,
  }),PLOT_CFG);
  renderStats([]);
}

function renderParallel(){
  const {color}=State.parallel;
  const dims=PK.map(k=>{
    const vs=D.trials.map(t=>t.params[k]);
    return {label:paramLabel(k),values:vs};
  });
  dims.push({label:'SCORE',values:D.trials.map(t=>t.score)});
  const cs=D.trials.map(t=>color==='score'?t.score:t.metricas[color]);

  const trace={
    type:'parcoords',
    line:{
      color:cs,
      colorscale:metricDir(color)==='min'?SCALE_INVERTED:SCALE_DIVERGING,
      showscale:true,
      colorbar:{title:{text:metricLabel(color),font:{color:C_MUTED}},tickfont:{color:C_MUTED},thickness:14,len:0.85},
    },
    dimensions:dims,
    labelfont:{color:C_TEXT,size:11},
    tickfont:{color:C_MUTED,size:10},
    rangefont:{color:C_DIM,size:10},
  };

  Plotly.newPlot('chart',[trace],baseLayout({
    height:620,margin:{l:90,r:90,t:60,b:50},
  }),PLOT_CFG);
  renderStats([]);
}

function renderPareto(){
  const {a,b}=State.pareto;
  const N=D.trials.length;
  const as=D.trials.map(t=>a==='score'?t.score:t.metricas[a]);
  const bs=D.trials.map(t=>b==='score'?t.score:t.metricas[b]);
  const dirA=metricDir(a),dirB=metricDir(b);

  const dom=new Array(N).fill(false);
  const valid=new Array(N).fill(false);
  for(let i=0;i<N;i++){
    if(as[i]==null||bs[i]==null||isNaN(as[i])||isNaN(bs[i]))continue;
    valid[i]=true;
  }
  for(let i=0;i<N;i++){
    if(!valid[i])continue;
    for(let j=0;j<N;j++){
      if(i===j||!valid[j])continue;
      const aBet=dirA==='max'?as[j]>=as[i]:as[j]<=as[i];
      const bBet=dirB==='max'?bs[j]>=bs[i]:bs[j]<=bs[i];
      const aStr=dirA==='max'?as[j]>as[i]:as[j]<as[i];
      const bStr=dirB==='max'?bs[j]>bs[i]:bs[j]<bs[i];
      if(aBet&&bBet&&(aStr||bStr)){dom[i]=true;break}
    }
  }
  const front=[];
  for(let i=0;i<N;i++)if(valid[i]&&!dom[i])front.push(i);
  front.sort((i,j)=>as[i]-as[j]);

  const dominados={
    type:'scatter',mode:'markers',
    x:D.trials.map((_,i)=>(valid[i]&&dom[i])?as[i]:null),
    y:D.trials.map((_,i)=>(valid[i]&&dom[i])?bs[i]:null),
    text:D.trials.map(t=>'Trial '+t.trial),
    marker:{size:6,color:C_DIM,opacity:0.45,line:{width:0}},
    name:'Dominados',
    hovertemplate:'%{text}<br>'+metricLabel(a)+': %{x:.4f}<br>'+metricLabel(b)+': %{y:.4f}<extra></extra>',
  };
  const frontera={
    type:'scatter',mode:'lines+markers',
    x:front.map(i=>as[i]),y:front.map(i=>bs[i]),
    text:front.map(i=>'Trial '+D.trials[i].trial),
    marker:{size:9,color:C_ACCENT,line:{color:'#fff',width:1.2}},
    line:{color:C_ACCENT,width:1.5,dash:'dot'},
    name:'Frontera Pareto',
    hovertemplate:'%{text}<br>'+metricLabel(a)+': %{x:.4f}<br>'+metricLabel(b)+': %{y:.4f}<extra></extra>',
  };

  Plotly.newPlot('chart',[dominados,frontera],baseLayout({
    title:{text:metricLabel(a)+' ('+(dirA==='max'?'maximizar':'minimizar')+') vs '+metricLabel(b)+' ('+(dirB==='max'?'maximizar':'minimizar')+')',
           font:{size:13},x:0.02,xanchor:'left'},
    xaxis:ax({title:{text:metricLabel(a),font:{color:C_MUTED}}}),
    yaxis:ax({title:{text:metricLabel(b),font:{color:C_MUTED}}}),
    height:620,
    legend:{font:{color:C_TEXT,size:11},bgcolor:'rgba(19,23,34,.85)',bordercolor:C_AXIS,borderwidth:1},
  }),PLOT_CFG);

  renderStats([
    ['Total trials',String(N)],
    ['En la frontera',String(front.length)],
    ['Dominados',String(N-front.length-(N-valid.filter(Boolean).length))],
    ['Sin datos',String(N-valid.filter(Boolean).length)],
  ]);
}

function renderDistribution(){
  const {metric}=State.dist;
  const vs=D.trials.map(t=>metric==='score'?t.score:t.metricas[metric]).filter(v=>v!=null&&!isNaN(v));
  if(!vs.length){Plotly.purge('chart');renderStats([['Sin datos','—']]);return}

  const sorted=[...vs].sort((a,b)=>a-b);
  const mean=vs.reduce((a,b)=>a+b,0)/vs.length;
  const std=Math.sqrt(vs.reduce((a,b)=>a+(b-mean)*(b-mean),0)/vs.length);
  const median=sorted[Math.floor(sorted.length/2)];
  const p25=sorted[Math.floor(sorted.length*0.25)];
  const p75=sorted[Math.floor(sorted.length*0.75)];

  const hist={
    type:'histogram',x:vs,
    nbinsx:Math.min(50,Math.max(12,Math.floor(Math.sqrt(vs.length)))),
    marker:{color:C_ACCENT,line:{color:'#1a1d29',width:1}},
    opacity:0.88,
    hovertemplate:'Rango: %{x}<br>Trials: %{y}<extra></extra>',
  };

  Plotly.newPlot('chart',[hist],baseLayout({
    title:{text:'Distribución de '+metricLabel(metric),font:{size:13},x:0.02,xanchor:'left'},
    xaxis:ax({title:{text:metricLabel(metric),font:{color:C_MUTED}}}),
    yaxis:ax({title:{text:'Trials',font:{color:C_MUTED}}}),
    height:580,bargap:0.04,
    shapes:[
      {type:'line',x0:mean,x1:mean,yref:'paper',y0:0,y1:1,line:{color:C_GOLD,width:1.5,dash:'dot'}},
      {type:'line',x0:median,x1:median,yref:'paper',y0:0,y1:1,line:{color:C_POS,width:1.5,dash:'dash'}},
    ],
    annotations:[
      {x:mean,y:1.04,yref:'paper',text:'media',showarrow:false,font:{color:C_GOLD,size:10}},
      {x:median,y:1.08,yref:'paper',text:'mediana',showarrow:false,font:{color:C_POS,size:10}},
    ],
  }),PLOT_CFG);

  renderStats([
    ['Media',fmtMetric(metric,mean)],
    ['Mediana',fmtMetric(metric,median)],
    ['Desv. típica',fmtMetric(metric,std)],
    ['P25',fmtMetric(metric,p25)],
    ['P75',fmtMetric(metric,p75)],
    ['Mín',fmtMetric(metric,sorted[0])],
    ['Máx',fmtMetric(metric,sorted[sorted.length-1])],
  ]);
}

function renderSensitivity(){
  const {metric}=State.sens;
  const ms=D.trials.map(t=>metric==='score'?t.score:t.metricas[metric]);
  const corrs=PK.map(p=>{
    const ps=D.trials.map(t=>t.params[p]);
    return {param:p,corr:pearson(ps,ms)};
  }).filter(d=>!isNaN(d.corr)).sort((a,b)=>Math.abs(b.corr)-Math.abs(a.corr));

  const trace={
    type:'bar',orientation:'h',
    x:corrs.map(d=>d.corr),
    y:corrs.map(d=>paramLabel(d.param)),
    marker:{color:corrs.map(d=>d.corr>=0?C_POS:C_NEG)},
    text:corrs.map(d=>d.corr.toFixed(3)),
    textposition:'outside',
    textfont:{color:C_TEXT,size:11},
    hovertemplate:'%{y}<br>Correlación: %{x:.4f}<extra></extra>',
  };

  Plotly.newPlot('chart',[trace],baseLayout({
    title:{text:'Sensibilidad de '+metricLabel(metric)+' a cada parámetro (Pearson)',font:{size:13},x:0.02,xanchor:'left'},
    xaxis:ax({title:{text:'Correlación',font:{color:C_MUTED}},range:[-1.1,1.1],zeroline:true,zerolinewidth:1.5,zerolinecolor:'#3f4453'}),
    yaxis:ax({automargin:true,categoryorder:'array',categoryarray:corrs.slice().reverse().map(d=>paramLabel(d.param))}),
    height:Math.max(320,corrs.length*42+120),showlegend:false,margin:{l:160,r:40,t:60,b:60},
  }),PLOT_CFG);

  if(!corrs.length){renderStats([['Sin datos','—']]);return}
  renderStats([
    ['Más sensible',paramLabel(corrs[0].param)],
    ['|Correlación| máx',Math.abs(corrs[0].corr).toFixed(3)],
    ['Menos sensible',paramLabel(corrs[corrs.length-1].param)],
    ['|Correlación| mín',Math.abs(corrs[corrs.length-1].corr).toFixed(3)],
  ]);
}

function renderStats(items){
  const el=document.getElementById('stats');
  el.innerHTML='';
  if(!items||!items.length){el.style.display='none';return}
  el.style.display='flex';
  items.forEach(([k,v])=>{
    const e=document.createElement('div');
    e.className='stat';
    e.innerHTML='<span class="stat-label">'+k+'</span><span class="stat-val">'+v+'</span>';
    el.appendChild(e);
  });
}

if(typeof Plotly!=='undefined'){
  rebuild();
  window.addEventListener('resize',()=>{
    const el=document.getElementById('chart');
    if(el&&el.data)Plotly.Plots.resize(el);
  });
}
</script>
</body>
</html>"""

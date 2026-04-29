from __future__ import annotations

import html as _html
import json
import math
from pathlib import Path


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

PARAM_LABELS = {
    "risk_max_pct": "RIESGO %",
    "risk_vol_halflife": "VOL HL",
    "risk_sl_ewma_mult": "SL xVOL",
    "risk_tp_ewma_mult": "TP xVOL",
    "risk_trail_act_ewma_mult": "TRAIL ACT xVOL",
    "risk_trail_dist_ewma_mult": "TRAIL DIST xVOL",
    "exit_sl_pct": "SL %",
    "exit_tp_pct": "TP %",
    "exit_velas": "VELAS",
    "exit_trail_act_pct": "TRAIL ACT %",
    "exit_trail_dist_pct": "TRAIL DIST %",
    "halflife_bars": "HALFLIFE BARS",
    "normalization_multiplier": "NORMALIZATION",
    "umbral_cvd": "UMBRAL CVD",
    "vwap_clip_sigmas": "VWAP CLIP SIGM",
}

METRIC_LABELS = {
    "roi_total": "ROI",
    "expectancy": "EXPECTANCY",
    "win_rate": "WIN RATE",
    "profit_factor": "PROFIT FACTOR",
    "sharpe_ratio": "SHARPE",
    "max_drawdown": "MAX DD",
    "total_trades": "TRADES",
    "trades_long": "LONG",
    "trades_short": "SHORT",
    "trades_ganadores": "WIN",
    "trades_perdedores": "LOSS",
    "trades_neutros": "BE",
    "trades_por_dia": "TRADES/DIA",
    "pnl_bruto_total": "PNL BRUTO",
    "pnl_total": "PNL NETO",
    "pnl_promedio": "PNL PROM",
    "saldo_inicial": "SALDO INICIAL",
    "saldo_final": "SALDO FINAL",
    "duracion_media_seg": "DUR MEDIA",
    "parado_por_saldo": "STOP SALDO",
    "score": "SCORE",
}

DERIVED_LABELS = {
    "return_dd_ratio": "ROI / DD",
    "pnl_por_trade": "PNL / TRADE",
    "score_dd_ratio": "SCORE / DD",
    "xvol_rr": "TP/SL xVOL",
    "trail_act_sl_ratio": "TRAIL ACT/SL",
    "trail_dist_sl_ratio": "TRAIL DIST/SL",
}

DERIVED_KEYS = tuple(DERIVED_LABELS.keys())


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
    derived_keys = [
        key
        for key in DERIVED_KEYS
        if any(_crear_derivados(t).get(key) is not None for t in trials)
    ]

    rows = []
    for t in trials:
        derivados = _crear_derivados(t)
        rows.append(
            {
                "trial": int(t.numero),
                "score": _safe_num(t.score),
                "params": {k: _to_jsonable((t.parametros or {}).get(k)) for k in params_keys},
                "metricas": {k: _to_jsonable((t.metricas or {}).get(k)) for k in metric_keys},
                "derived": {k: _to_jsonable(derivados.get(k)) for k in derived_keys},
            }
        )

    mejor = max(trials, key=lambda x: x.score)
    field_labels = _field_labels(params_keys, metric_keys, derived_keys)

    return {
        "titulo": f"{activo} {timeframe} | {estrategia.NOMBRE} | {salida_tipo}",
        "activo": activo,
        "timeframe": timeframe,
        "estrategia": estrategia.NOMBRE,
        "salida": salida_tipo,
        "params_keys": params_keys,
        "metric_keys": metric_keys,
        "derived_keys": derived_keys,
        "field_labels": field_labels,
        "trials": rows,
        "mejor_trial": int(mejor.numero),
        "mejor_score": _safe_num(mejor.score),
        "mejor_metricas": {
            k: _to_jsonable((mejor.metricas or {}).get(k)) for k in metric_keys
        },
        "mejor_parametros": {
            k: _to_jsonable((mejor.parametros or {}).get(k)) for k in params_keys
        },
        "n_trials": len(trials),
    }


def _crear_derivados(trial) -> dict[str, float | None]:
    metricas = trial.metricas or {}
    parametros = trial.parametros or {}
    roi = _num_or_none(metricas.get("roi_total"))
    dd = _num_or_none(metricas.get("max_drawdown"))
    pnl = _num_or_none(metricas.get("pnl_total"))
    trades = _num_or_none(metricas.get("total_trades"))
    score = _num_or_none(trial.score)
    sl_mult = _num_or_none(parametros.get("risk_sl_ewma_mult"))
    tp_mult = _num_or_none(parametros.get("risk_tp_ewma_mult"))
    trail_act = _num_or_none(parametros.get("risk_trail_act_ewma_mult"))
    trail_dist = _num_or_none(parametros.get("risk_trail_dist_ewma_mult"))

    return {
        "return_dd_ratio": _safe_div(roi, dd),
        "pnl_por_trade": _safe_div(pnl, trades),
        "score_dd_ratio": _safe_div(score, dd),
        "xvol_rr": _safe_div(tp_mult, sl_mult),
        "trail_act_sl_ratio": _safe_div(trail_act, sl_mult),
        "trail_dist_sl_ratio": _safe_div(trail_dist, sl_mult),
    }


def _field_labels(
    params_keys: list[str],
    metric_keys: list[str],
    derived_keys: list[str],
) -> dict[str, str]:
    labels = {"score": "SCORE", "trial": "TRIAL"}
    labels.update({k: PARAM_LABELS.get(k, _label_generico(k)) for k in params_keys})
    labels.update({k: METRIC_LABELS.get(k, _label_generico(k)) for k in metric_keys})
    labels.update({k: DERIVED_LABELS.get(k, _label_generico(k)) for k in derived_keys})
    return labels


def _label_generico(key: str) -> str:
    return str(key).replace("_", " ").upper()


def _num_or_none(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    return float(num) / float(den)


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
<title>__TITULO__ - Informe Robustez</title>
<script src="__PLOTLY_CDN__"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#000;--panel:#0a0a0a;--panel2:#101010;--border:#1f1f1f;--border2:#2a2a2a;--text:#e8e8e8;--mute:#b8b8b8;--dim:#7d7d7d;--accent:#ffb000;--accent2:#ff8a00;--pos:#5cdb5c;--neg:#ff4d4d;--blue:#3aa3ff;--grid:#101010;--font:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;--mono:"JetBrains Mono","SFMono-Regular",Menlo,Consolas,monospace}
html,body{min-height:100%;background:var(--bg)}
body{font-family:var(--font);background:var(--bg);color:var(--text);font-size:12px;-webkit-font-smoothing:antialiased}
.mono{font-family:var(--mono);font-feature-settings:"tnum" 1,"zero" 1}.pos{color:var(--pos)!important}.neg{color:var(--neg)!important}.muted{color:var(--mute)}
.topbar{display:flex;align-items:center;gap:14px;background:var(--accent);color:#000;padding:3px 14px;border-bottom:2px solid #000;font-family:var(--mono);font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.topbar .clock{margin-left:auto}
.title-strip{display:flex;align-items:baseline;gap:16px;background:var(--panel);border-bottom:1px solid var(--border);padding:8px 14px}.title-strip h1{font-family:var(--mono);font-size:13px;color:var(--text);letter-spacing:.05em;text-transform:uppercase}.title-strip .sub{font-family:var(--mono);font-size:10px;color:var(--mute);letter-spacing:.08em;text-transform:uppercase}.title-strip .badge{margin-left:auto;border:1px solid var(--accent);color:var(--accent);font-family:var(--mono);font-size:10px;padding:3px 9px;letter-spacing:.1em}
.headline{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));background:var(--panel);border-bottom:1px solid var(--border)}.m-cell{position:relative;min-width:0;padding:8px 14px;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:2px}.m-cell:last-child{border-right:none}.m-cell:first-child:before{content:"";position:absolute;left:0;top:0;bottom:0;width:2px;background:var(--accent)}.m-label{font-family:var(--mono);font-size:9px;color:var(--mute);letter-spacing:.14em;text-transform:uppercase}.m-val{font-family:var(--mono);font-size:17px;font-weight:800;white-space:nowrap}
.params-row{display:flex;align-items:center;gap:8px;min-height:31px;padding:5px 14px;background:#000;border-bottom:1px solid var(--border);font-family:var(--mono);overflow:hidden}.params-row .lbl{font-size:9px;color:var(--mute);letter-spacing:.14em;margin-right:4px}.param-chip{display:flex;align-items:baseline;gap:6px;padding:2px 8px;background:var(--panel);border:1px solid var(--border);font-size:10px;white-space:nowrap}.param-chip span{color:var(--mute);font-size:9px;letter-spacing:.08em;text-transform:uppercase}.param-chip b{color:var(--accent);font-weight:800}
.toolbar{display:flex;flex-wrap:wrap;background:var(--panel);border-bottom:1px solid var(--border)}.tool-label{font-family:var(--mono);font-size:9px;color:var(--mute);letter-spacing:.14em;padding:8px 12px;border-right:1px solid var(--border);text-transform:uppercase}.view-btn{font-family:var(--mono);background:transparent;color:var(--mute);border:none;border-right:1px solid var(--border);padding:8px 11px;font-size:10px;letter-spacing:.08em;cursor:pointer;text-transform:uppercase;font-weight:700}.view-btn:hover{background:var(--panel2);color:var(--text)}.view-btn.active{color:var(--accent);background:#1a1100}
.help{font-family:var(--mono);font-size:10px;line-height:1.55;color:var(--mute);padding:8px 14px;background:#000;border-bottom:1px solid var(--border)}.help b{color:var(--accent)}
.controls{display:flex;gap:0;flex-wrap:wrap;background:var(--panel);border-bottom:1px solid var(--border)}.ctl{display:flex;align-items:center;gap:7px;padding:7px 12px;border-right:1px solid var(--border);font-family:var(--mono)}.ctl label{font-size:9px;color:var(--mute);letter-spacing:.12em;text-transform:uppercase;white-space:nowrap}.ctl select{background:#000;color:var(--text);border:1px solid var(--border2);border-radius:0;padding:5px 8px;font-family:var(--mono);font-size:10px;min-width:150px;outline:none}.ctl select:hover,.ctl select:focus{border-color:var(--accent)}
#stage{padding:12px 14px 22px;background:#000}.chart-box{min-height:640px;background:#000;border:1px solid var(--border);position:relative}.stats-row{display:flex;gap:0;flex-wrap:wrap;background:var(--panel);border:1px solid var(--border);border-top:none}.stat{padding:8px 14px;border-right:1px solid var(--border);min-width:120px;font-family:var(--mono)}.stat-label{display:block;color:var(--mute);font-size:9px;letter-spacing:.12em;text-transform:uppercase}.stat-val{display:block;color:var(--text);font-size:13px;font-weight:800;margin-top:2px}.data-table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:10.5px}.data-table th{position:sticky;top:0;background:var(--panel2);color:var(--mute);font-size:9px;letter-spacing:.1em;text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);text-transform:uppercase}.data-table td{padding:6px 10px;border-bottom:1px solid #080808;white-space:nowrap}.data-table tr:hover td{background:#111}.num{text-align:right}.rank-wrap{max-height:640px;overflow:auto;background:#000}.offline{display:none;color:var(--neg);font-family:var(--mono);padding:18px 14px;background:#150000;border-bottom:1px solid #3a0000}
@media(max-width:1000px){.headline{grid-template-columns:repeat(2,minmax(0,1fr))}.title-strip{flex-wrap:wrap}.title-strip .badge{margin-left:0}.ctl{width:100%;justify-content:space-between}.ctl select{min-width:180px}.view-btn{font-size:9px;padding:8px}.chart-box{min-height:520px}}
</style>
</head>
<body>
<div class="topbar"><div>MNFRESCOX TERMINAL</div><div>ROBUSTEZ REPORT - V2</div><div class="clock" id="clock">--:--:-- UTC</div></div>
<div class="title-strip"><h1 id="title">__TITULO__</h1><div class="sub" id="subtitle"></div><div class="badge" id="best-badge">#-</div></div>
<div class="headline" id="headline"></div>
<div class="params-row"><span class="lbl">PARAMS</span><div id="param-chips"></div></div>
<div class="toolbar"><span class="tool-label">VISTAS</span><div id="view-buttons"></div></div>
<div class="help" id="help"></div>
<div class="controls" id="controls"></div>
<div class="offline" id="offline">No se pudo cargar Plotly. Las tablas siguen disponibles, pero los graficos avanzados necesitan conexion al CDN.</div>
<div id="stage"><div id="chart" class="chart-box"></div><div class="stats-row" id="stats"></div></div>
<script>
'use strict';
const D=__DATA_JSON__;
const T={bg:'#000000',panel:'#0a0a0a',panel2:'#101010',grid:'#101010',border:'#1f1f1f',text:'#e8e8e8',mute:'#b8b8b8',dim:'#7d7d7d',accent:'#ffb000',accent2:'#ff8a00',pos:'#5cdb5c',neg:'#ff4d4d',blue:'#3aa3ff',mono:'JetBrains Mono, SFMono-Regular, Menlo, Consolas, monospace'};
const PARAM_KEYS=D.params_keys||[],METRIC_KEYS=D.metric_keys||[],DERIVED_KEYS=D.derived_keys||[];
const FIELD_LABELS=D.field_labels||{};
const PCT_FIELDS=new Set(['roi_total','expectancy','win_rate','max_drawdown']);
const MONEY_FIELDS=new Set(['pnl_bruto_total','pnl_total','pnl_promedio','saldo_inicial','saldo_final','pnl_por_trade']);
const INT_FIELDS=new Set(['total_trades','trades_long','trades_short','trades_ganadores','trades_perdedores','trades_neutros','risk_vol_halflife','exit_velas']);
const DURATION_FIELDS=new Set(['duracion_media_seg']);
const MIN_BETTER=new Set(['max_drawdown','parado_por_saldo','duracion_media_seg']);
const FIELD_OPTS=()=>[['score',label('score')],...METRIC_KEYS.map(k=>[k,label(k)]),...DERIVED_KEYS.map(k=>[k,label(k)])];
const AXIS_OPTS=()=>[['score',label('score')],...PARAM_KEYS.map(k=>[k,label(k)]),...METRIC_KEYS.map(k=>[k,label(k)]),...DERIVED_KEYS.map(k=>[k,label(k)])];
const PARAM_OPTS=()=>PARAM_KEYS.map(k=>[k,label(k)]);
const AGG_OPTS=[['mean','MEDIA'],['median','MEDIANA'],['max','MAX'],['min','MIN'],['std','DESV'],['count','N']];
function esc(v){return String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function label(k){return FIELD_LABELS[k]||String(k).replaceAll('_',' ').toUpperCase()}
function kind(k){if(k==='score')return 'score';if(PARAM_KEYS.includes(k))return 'param';if(METRIC_KEYS.includes(k))return 'metric';if(DERIVED_KEYS.includes(k))return 'derived';return 'unknown'}
function value(t,k){if(k==='score')return t.score;if(t.params&&k in t.params)return t.params[k];if(t.metricas&&k in t.metricas)return t.metricas[k];if(t.derived&&k in t.derived)return t.derived[k];return null}
function dir(k){return MIN_BETTER.has(k)?'min':'max'}
function num(v){const n=Number(v);return Number.isFinite(n)?n:null}
function fmt(k,v){const n=num(v);if(n===null)return '-';if(PCT_FIELDS.has(k))return (n*100).toFixed(2)+'%';if(MONEY_FIELDS.has(k))return '$'+n.toLocaleString(undefined,{maximumFractionDigits:2});if(INT_FIELDS.has(k))return Math.round(n).toLocaleString();if(DURATION_FIELDS.has(k))return fmtDur(n);if(k==='score')return n.toFixed(6);if(Math.abs(n)>=1000)return n.toLocaleString(undefined,{maximumFractionDigits:2});if(Math.abs(n)>=10)return n.toFixed(2);return n.toFixed(4)}
function fmtDur(s){if(!Number.isFinite(s)||s<=0)return '-';const d=Math.floor(s/86400),h=Math.floor((s%86400)/3600),m=Math.floor((s%3600)/60);if(d)return d+'d '+String(h).padStart(2,'0')+'h';if(h)return h+'h '+String(m).padStart(2,'0')+'m';if(m)return m+'m';return Math.round(s)+'s'}
function cls(k,v){const n=num(v);if(n===null)return '';if(k==='max_drawdown'||k.includes('dd'))return n>0?'neg':'';return n>=0?'pos':'neg'}
function sortedTrials(k='score',desc=true){return [...D.trials].sort((a,b)=>{const av=num(value(a,k))??-Infinity,bv=num(value(b,k))??-Infinity;return desc?bv-av:av-bv})}
function cleanValues(k){return D.trials.map(t=>num(value(t,k))).filter(v=>v!==null)}
function aggregate(arr,mode){const vs=arr.map(num).filter(v=>v!==null);if(!vs.length)return null;if(mode==='count')return vs.length;if(mode==='mean')return vs.reduce((a,b)=>a+b,0)/vs.length;if(mode==='median'){const s=[...vs].sort((a,b)=>a-b),m=Math.floor(s.length/2);return s.length%2?s[m]:(s[m-1]+s[m])/2}if(mode==='min')return Math.min(...vs);if(mode==='max')return Math.max(...vs);if(mode==='std'){const mean=aggregate(vs,'mean');return Math.sqrt(vs.reduce((a,b)=>a+(b-mean)**2,0)/vs.length)}return null}
function pearson(xs,ys){const p=xs.map((x,i)=>[num(x),num(ys[i])]).filter(([x,y])=>x!==null&&y!==null);if(p.length<2)return null;const mx=p.reduce((a,[x])=>a+x,0)/p.length,my=p.reduce((a,[,y])=>a+y,0)/p.length;let n=0,dx=0,dy=0;for(const [x,y]of p){const a=x-mx,b=y-my;n+=a*b;dx+=a*a;dy+=b*b}return dx&&dy?n/Math.sqrt(dx*dy):0}
function colorscale(k){return dir(k)==='min'?[[0,T.pos],[.5,T.panel2],[1,T.neg]]:[[0,T.neg],[.5,T.panel2],[1,T.pos]]}
function plot(data,layout){if(typeof Plotly==='undefined'){document.getElementById('offline').style.display='block';return}const el=document.getElementById('chart');el.innerHTML='';Plotly.newPlot(el,data,Object.assign({paper_bgcolor:T.bg,plot_bgcolor:T.bg,font:{family:T.mono,color:T.text,size:11},margin:{l:72,r:28,t:42,b:64},hoverlabel:{bgcolor:'#000',bordercolor:T.accent,font:{color:T.text}},xaxis:axis(),yaxis:axis()},layout||{}),{responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{format:'png',filename:'robustez',height:900,width:1600,scale:2}})}
function axis(extra){return Object.assign({gridcolor:T.grid,zerolinecolor:T.border,linecolor:T.border,tickcolor:T.border,color:T.mute,tickfont:{color:T.mute,size:10},titlefont:{color:T.mute,size:10}},extra||{})}
function purge(){if(typeof Plotly!=='undefined')Plotly.purge('chart');document.getElementById('chart').innerHTML=''}
function stats(items){const el=document.getElementById('stats');el.innerHTML='';if(!items||!items.length){el.style.display='none';return}el.style.display='flex';for(const [k,v,c]of items){el.insertAdjacentHTML('beforeend',`<div class="stat"><span class="stat-label">${esc(k)}</span><span class="stat-val ${c||''}">${esc(v)}</span></div>`)}}
function metricOptions(){return FIELD_OPTS().filter(([k])=>kind(k)!=='param')}
function best(){return D.trials.find(t=>t.trial===D.mejor_trial)||sortedTrials()[0]||{params:{},metricas:{},derived:{}}}

function initHeader(){const b=best();document.getElementById('subtitle').textContent=[D.activo,D.timeframe,D.estrategia,D.salida,D.n_trials+' trials'].join(' | ');document.getElementById('best-badge').textContent='#'+D.mejor_trial;const cells=[['ROI','roi_total'],['PROFIT F','profit_factor'],['SHARPE','sharpe_ratio'],['MAX DD','max_drawdown'],['WIN RATE','win_rate'],['SCORE','score']];document.getElementById('headline').innerHTML=cells.map(([name,k])=>{const v=value(b,k);return `<div class="m-cell"><div class="m-label">${name}</div><div class="m-val ${cls(k,v)}">${fmt(k,v)}</div></div>`}).join('');const params=PARAM_KEYS.slice(0,10).map(k=>`<div class="param-chip"><span>${esc(label(k))}</span><b>${esc(fmt(k,b.params?.[k]))}</b></div>`).join('');document.getElementById('param-chips').innerHTML=params||'<div class="param-chip"><span>SIN PARAMS</span><b>-</b></div>';document.getElementById('clock').textContent=new Date().toISOString().slice(11,19)+' UTC'}

const VIEWS=[
{id:'ranking',label:'RANKING',help:'Tabla operativa de mejores trials con metricas clave y parametros principales.',render:renderRanking},
{id:'heatmap',label:'ROBUSTEZ 2D',help:'Mapa de calor por <b>2 parametros</b>. Usa media, mediana, desviacion o conteo para detectar zonas robustas y no picos aislados.',render:renderHeatmap},
{id:'topzone',label:'TOP-ZONE',help:'Mapa de concentracion de mejores trials. Muestra donde se agrupan las configuraciones del percentil superior.',render:renderTopZone},
{id:'scatter2d',label:'SCATTER 2D',help:'Dispersión simple de dos variables con color por metrica. Es la vista rapida cuando quieres analizar solo 2 ejes.',render:renderScatter2D},
{id:'scatter3d',label:'SCATTER 3D',help:'Dispersión 3D para clusters y trade-offs con una cuarta variable en color.',render:renderScatter3D},
{id:'pareto',label:'PARETO',help:'Frontera no dominada entre dos metricas. Util para ver ROI contra DD, expectancy contra DD o score contra riesgo.',render:renderPareto},
{id:'correlation',label:'CORRELACION',help:'Matriz de correlacion entre parametros y metricas/derivados. Ayuda a detectar que controla realmente el resultado.',render:renderCorrelation},
{id:'convergence',label:'CONVERGENCIA',help:'Evolucion por numero de trial y mejor valor acumulado. Sirve para evaluar si Optuna sigue encontrando mejora.',render:renderConvergence},
{id:'distribution',label:'DISTRIBUCION',help:'Histograma y caja de una metrica para ver dispersion, outliers y percentiles.',render:renderDistribution},
{id:'sensitivity',label:'SENSIBILIDAD',help:'Ranking de correlaciones parametro -> metrica objetivo. Valores cercanos a cero implican menor sensibilidad.',render:renderSensitivity},
{id:'parallel',label:'MULTI',help:'Coordenadas paralelas de todos los parametros y score, coloreadas por la metrica seleccionada.',render:renderParallel},
];
const State={
ranking:{sort:'score'},heatmap:{x:PARAM_KEYS[0],y:PARAM_KEYS[1]||PARAM_KEYS[0],metric:'score',agg:'median'},topzone:{x:PARAM_KEYS[0],y:PARAM_KEYS[1]||PARAM_KEYS[0],metric:'score',top:'20'},scatter2d:{x:PARAM_KEYS[0]||'score',y:'score',color:METRIC_KEYS.includes('roi_total')?'roi_total':'score'},scatter3d:{x:PARAM_KEYS[0]||'score',y:PARAM_KEYS[1]||'score',z:'score',color:METRIC_KEYS.includes('roi_total')?'roi_total':'score'},pareto:{x:METRIC_KEYS.includes('roi_total')?'roi_total':'score',y:METRIC_KEYS.includes('max_drawdown')?'max_drawdown':'score'},correlation:{target:'score'},convergence:{metric:'score'},distribution:{metric:'score'},sensitivity:{metric:'score'},parallel:{color:'score'}
};
let currentView='ranking';
function buildViews(){const host=document.getElementById('view-buttons');host.innerHTML=VIEWS.map(v=>`<button class="view-btn ${v.id===currentView?'active':''}" data-view="${v.id}" type="button">${v.label}</button>`).join('');host.querySelectorAll('button').forEach(btn=>btn.addEventListener('click',()=>switchView(btn.dataset.view)))}
function switchView(id){if(id===currentView)return;currentView=id;buildViews();renderCurrent()}
function makeSelect(id,text,opts,val,setter){const wrap=document.createElement('div');wrap.className='ctl';const lab=document.createElement('label');lab.textContent=text;lab.htmlFor=id;const sel=document.createElement('select');sel.id=id;for(const [v,t]of opts){const o=document.createElement('option');o.value=v;o.textContent=t;if(v===val)o.selected=true;sel.appendChild(o)}sel.addEventListener('change',()=>{setter(sel.value);renderCurrent()});wrap.appendChild(lab);wrap.appendChild(sel);return wrap}
function buildControls(){const c=document.getElementById('controls');c.innerHTML='';const s=State[currentView];if(currentView==='ranking'){c.appendChild(makeSelect('ranking-sort','ORDEN',metricOptions(),s.sort,v=>s.sort=v))}else if(currentView==='heatmap'){c.appendChild(makeSelect('hm-x','X PARAM',PARAM_OPTS(),s.x,v=>s.x=v));c.appendChild(makeSelect('hm-y','Y PARAM',PARAM_OPTS(),s.y,v=>s.y=v));c.appendChild(makeSelect('hm-metric','METRICA',metricOptions(),s.metric,v=>s.metric=v));c.appendChild(makeSelect('hm-agg','AGG',AGG_OPTS,s.agg,v=>s.agg=v))}else if(currentView==='topzone'){c.appendChild(makeSelect('tz-x','X PARAM',PARAM_OPTS(),s.x,v=>s.x=v));c.appendChild(makeSelect('tz-y','Y PARAM',PARAM_OPTS(),s.y,v=>s.y=v));c.appendChild(makeSelect('tz-metric','METRICA',metricOptions(),s.metric,v=>s.metric=v));c.appendChild(makeSelect('tz-top','TOP',[['10','TOP 10%'],['20','TOP 20%'],['30','TOP 30%']],s.top,v=>s.top=v))}else if(currentView==='scatter2d'){c.appendChild(makeSelect('sc2-x','X',AXIS_OPTS(),s.x,v=>s.x=v));c.appendChild(makeSelect('sc2-y','Y',AXIS_OPTS(),s.y,v=>s.y=v));c.appendChild(makeSelect('sc2-color','COLOR',metricOptions(),s.color,v=>s.color=v))}else if(currentView==='scatter3d'){c.appendChild(makeSelect('sc3-x','X',AXIS_OPTS(),s.x,v=>s.x=v));c.appendChild(makeSelect('sc3-y','Y',AXIS_OPTS(),s.y,v=>s.y=v));c.appendChild(makeSelect('sc3-z','Z',AXIS_OPTS(),s.z,v=>s.z=v));c.appendChild(makeSelect('sc3-color','COLOR',metricOptions(),s.color,v=>s.color=v))}else if(currentView==='pareto'){c.appendChild(makeSelect('pa-x','METRICA X',metricOptions(),s.x,v=>s.x=v));c.appendChild(makeSelect('pa-y','METRICA Y',metricOptions(),s.y,v=>s.y=v))}else if(currentView==='correlation'){c.appendChild(makeSelect('co-target','FOCO',metricOptions(),s.target,v=>s.target=v))}else if(currentView==='convergence'){c.appendChild(makeSelect('cv-metric','METRICA',metricOptions(),s.metric,v=>s.metric=v))}else if(currentView==='distribution'){c.appendChild(makeSelect('di-metric','METRICA',metricOptions(),s.metric,v=>s.metric=v))}else if(currentView==='sensitivity'){c.appendChild(makeSelect('se-metric','METRICA',metricOptions(),s.metric,v=>s.metric=v))}else if(currentView==='parallel'){c.appendChild(makeSelect('pc-color','COLOR',metricOptions(),s.color,v=>s.color=v))}}
function readControls(){return true}
function renderCurrent(){
  buildControls();
  readControls();
  const view=VIEWS.find(v=>v.id===currentView)||VIEWS[0];
  document.getElementById('help').innerHTML=view.help;
  view.render();
}

function cellGrid(x,y,m,agg){const xs=[...new Set(D.trials.map(t=>value(t,x)).filter(v=>v!=null))].sort((a,b)=>a-b);const ys=[...new Set(D.trials.map(t=>value(t,y)).filter(v=>v!=null))].sort((a,b)=>a-b);const map=new Map();for(const t of D.trials){const xv=value(t,x),yv=value(t,y),mv=value(t,m);if(xv==null||yv==null||mv==null)continue;const key=xv+'|'+yv;if(!map.has(key))map.set(key,[]);map.get(key).push(mv)}const z=ys.map(yv=>xs.map(xv=>aggregate(map.get(xv+'|'+yv)||[],agg)));return{xs,ys,z,map}}
function renderRanking(){purge();const s=State.ranking;const trials=sortedTrials(s.sort,dir(s.sort)==='max').slice(0,250);const cols=['trial','score','roi_total','max_drawdown','profit_factor','sharpe_ratio','total_trades',...PARAM_KEYS.slice(0,7)];const head=cols.map(c=>`<th class="${c==='trial'?'':'num'}">${esc(label(c))}</th>`).join('');const rows=trials.map(t=>`<tr>${cols.map(c=>{const v=c==='trial'?t.trial:value(t,c);return `<td class="${c==='trial'?'':'num'} ${cls(c,v)}">${esc(c==='trial'?'#'+v:fmt(c,v))}</td>`}).join('')}</tr>`).join('');document.getElementById('chart').innerHTML=`<div class="rank-wrap"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;const b=best();stats([['MEJOR TRIAL','#'+b.trial],['SCORE',fmt('score',b.score),'pos'],['ORDEN',label(s.sort)],['MOSTRADOS',String(trials.length)]])}
function renderHeatmap(){const s=State.heatmap,g=cellGrid(s.x,s.y,s.metric,s.agg);plot([{type:'heatmap',x:g.xs,y:g.ys,z:g.z,colorscale:colorscale(s.metric),hoverongaps:false,colorbar:{title:{text:label(s.metric),font:{color:T.mute}},tickfont:{color:T.mute}},hovertemplate:label(s.x)+': %{x}<br>'+label(s.y)+': %{y}<br>'+label(s.metric)+': %{z:.4f}<extra></extra>'}],{title:{text:s.agg.toUpperCase()+' '+label(s.metric)+' - '+label(s.x)+' x '+label(s.y),font:{size:12}},xaxis:axis({title:label(s.x)}),yaxis:axis({title:label(s.y)}),height:650});stats([['X',label(s.x)],['Y',label(s.y)],['METRICA',label(s.metric)],['AGG',s.agg.toUpperCase()]])}
function renderTopZone(){const s=State.topzone;const vals=D.trials.map(t=>num(value(t,s.metric))).filter(v=>v!==null).sort((a,b)=>dir(s.metric)==='max'?b-a:a-b);const cut=vals[Math.max(0,Math.floor(vals.length*(Number(s.top)/100))-1)]??null;const top=D.trials.filter(t=>{const v=num(value(t,s.metric));return v!==null&&(dir(s.metric)==='max'?v>=cut:v<=cut)});const xs=[...new Set(D.trials.map(t=>value(t,s.x)).filter(v=>v!=null))].sort((a,b)=>a-b);const ys=[...new Set(D.trials.map(t=>value(t,s.y)).filter(v=>v!=null))].sort((a,b)=>a-b);const counts=new Map();for(const t of top){const key=value(t,s.x)+'|'+value(t,s.y);counts.set(key,(counts.get(key)||0)+1)}const z=ys.map(yv=>xs.map(xv=>counts.get(xv+'|'+yv)||0));plot([{type:'heatmap',x:xs,y:ys,z,colorscale:[[0,T.panel2],[.5,T.accent2],[1,T.accent]],colorbar:{title:{text:'TOP TRIALS',font:{color:T.mute}},tickfont:{color:T.mute}},hovertemplate:label(s.x)+': %{x}<br>'+label(s.y)+': %{y}<br>Top trials: %{z}<extra></extra>'}],{title:{text:'CONCENTRACION TOP '+s.top+'% - '+label(s.metric),font:{size:12}},xaxis:axis({title:label(s.x)}),yaxis:axis({title:label(s.y)}),height:650});stats([['CORTE',fmt(s.metric,cut)],['TRIALS TOP',String(top.length)],['TOTAL',String(D.trials.length)]])}
function renderScatter2D(){const s=State.scatter2d;plot([{type:'scattergl',mode:'markers',x:D.trials.map(t=>value(t,s.x)),y:D.trials.map(t=>value(t,s.y)),text:D.trials.map(t=>'#'+t.trial),marker:{size:7,color:D.trials.map(t=>value(t,s.color)),colorscale:colorscale(s.color),showscale:true,colorbar:{title:{text:label(s.color),font:{color:T.mute}},tickfont:{color:T.mute}},line:{color:'#000',width:.5}},hovertemplate:'Trial %{text}<br>'+label(s.x)+': %{x}<br>'+label(s.y)+': %{y}<extra></extra>'}],{title:{text:label(s.x)+' vs '+label(s.y),font:{size:12}},xaxis:axis({title:label(s.x)}),yaxis:axis({title:label(s.y)}),height:650});stats([['MODO','2 EJES'],['X',label(s.x)],['Y',label(s.y)],['COLOR',label(s.color)]])}
function renderScatter3D(){const s=State.scatter3d;plot([{type:'scatter3d',mode:'markers',x:D.trials.map(t=>value(t,s.x)),y:D.trials.map(t=>value(t,s.y)),z:D.trials.map(t=>value(t,s.z)),text:D.trials.map(t=>'#'+t.trial),marker:{size:4,color:D.trials.map(t=>value(t,s.color)),colorscale:colorscale(s.color),showscale:true,colorbar:{title:{text:label(s.color),font:{color:T.mute}},tickfont:{color:T.mute}},opacity:.85},hovertemplate:'Trial %{text}<br>'+label(s.x)+': %{x}<br>'+label(s.y)+': %{y}<br>'+label(s.z)+': %{z}<extra></extra>'}],{scene:{xaxis:{title:label(s.x),color:T.mute,gridcolor:T.grid,backgroundcolor:T.bg,showbackground:true},yaxis:{title:label(s.y),color:T.mute,gridcolor:T.grid,backgroundcolor:T.bg,showbackground:true},zaxis:{title:label(s.z),color:T.mute,gridcolor:T.grid,backgroundcolor:T.bg,showbackground:true},bgcolor:T.bg},height:700});stats([['MODO','3 EJES'],['COLOR',label(s.color)]])}
function renderPareto(){const s=State.pareto,N=D.trials.length,xs=D.trials.map(t=>num(value(t,s.x))),ys=D.trials.map(t=>num(value(t,s.y)));const valid=xs.map((x,i)=>x!==null&&ys[i]!==null),dom=new Array(N).fill(false);for(let i=0;i<N;i++){if(!valid[i])continue;for(let j=0;j<N;j++){if(i===j||!valid[j])continue;const xb=dir(s.x)==='max'?xs[j]>=xs[i]:xs[j]<=xs[i],yb=dir(s.y)==='max'?ys[j]>=ys[i]:ys[j]<=ys[i],xs2=dir(s.x)==='max'?xs[j]>xs[i]:xs[j]<xs[i],ys2=dir(s.y)==='max'?ys[j]>ys[i]:ys[j]<ys[i];if(xb&&yb&&(xs2||ys2)){dom[i]=true;break}}}const front=D.trials.map((t,i)=>({t,i,x:xs[i],y:ys[i]})).filter(o=>valid[o.i]&&!dom[o.i]).sort((a,b)=>a.x-b.x);plot([{type:'scatter',mode:'markers',x:xs.map((x,i)=>valid[i]&&dom[i]?x:null),y:ys.map((y,i)=>valid[i]&&dom[i]?y:null),marker:{size:7,color:T.dim,opacity:.45},text:D.trials.map(t=>'#'+t.trial),name:'Dominados',hovertemplate:'Trial %{text}<br>'+label(s.x)+': %{x}<br>'+label(s.y)+': %{y}<extra></extra>'},{type:'scatter',mode:'lines+markers',x:front.map(o=>o.x),y:front.map(o=>o.y),text:front.map(o=>'#'+o.t.trial),marker:{size:10,color:T.accent,line:{color:'#fff',width:1}},line:{color:T.accent,width:1.5,dash:'dot'},name:'Frontera',hovertemplate:'Trial %{text}<br>'+label(s.x)+': %{x}<br>'+label(s.y)+': %{y}<extra></extra>'}],{title:{text:'PARETO - '+label(s.x)+' vs '+label(s.y),font:{size:12}},xaxis:axis({title:label(s.x)}),yaxis:axis({title:label(s.y)}),legend:{font:{color:T.text},bgcolor:T.panel,bordercolor:T.border,borderwidth:1},height:650});stats([['FRONTERA',String(front.length)],['DOMINADOS',String(dom.filter(Boolean).length)]])}
function renderCorrelation(){const ys=[...PARAM_KEYS],xs=['score',...METRIC_KEYS,...DERIVED_KEYS];const z=ys.map(p=>xs.map(m=>pearson(D.trials.map(t=>value(t,p)),D.trials.map(t=>value(t,m)))));plot([{type:'heatmap',x:xs.map(label),y:ys.map(label),z,colorscale:[[0,T.neg],[.5,T.panel2],[1,T.pos]],zmin:-1,zmax:1,colorbar:{title:{text:'r',font:{color:T.mute}},tickfont:{color:T.mute}},hovertemplate:'%{y}<br>%{x}<br>r=%{z:.3f}<extra></extra>'}],{title:{text:'CORRELACION PARAMETRO - RESULTADO',font:{size:12}},xaxis:axis({tickangle:-35}),yaxis:axis({automargin:true}),height:Math.max(560,ys.length*34+160),margin:{l:170,r:30,t:48,b:130}});stats([['PARAMS',String(ys.length)],['VARIABLES',String(xs.length)]])}
function renderConvergence(){const s=State.convergence;const ordered=[...D.trials].sort((a,b)=>a.trial-b.trial);let bestVal=null;const bestLine=[];for(const t of ordered){const v=num(value(t,s.metric));if(v!==null&&(bestVal===null||(dir(s.metric)==='max'?v>bestVal:v<bestVal)))bestVal=v;bestLine.push(bestVal)}plot([{type:'scatter',mode:'markers',x:ordered.map(t=>t.trial),y:ordered.map(t=>value(t,s.metric)),marker:{size:5,color:T.dim},name:'Trial'},{type:'scatter',mode:'lines',x:ordered.map(t=>t.trial),y:bestLine,line:{color:T.accent,width:2},name:'Mejor acumulado'}],{title:{text:'CONVERGENCIA - '+label(s.metric),font:{size:12}},xaxis:axis({title:'TRIAL'}),yaxis:axis({title:label(s.metric)}),height:620,legend:{font:{color:T.text},bgcolor:T.panel,bordercolor:T.border,borderwidth:1}});stats([['ULTIMO BEST',fmt(s.metric,bestLine[bestLine.length-1])],['TRIALS',String(ordered.length)]])}
function renderDistribution(){const s=State.distribution,vs=cleanValues(s.metric).sort((a,b)=>a-b);if(!vs.length){purge();stats([['SIN DATOS','-']]);return}const mean=aggregate(vs,'mean'),med=aggregate(vs,'median'),std=aggregate(vs,'std');plot([{type:'histogram',x:vs,nbinsx:Math.min(60,Math.max(12,Math.floor(Math.sqrt(vs.length)))),marker:{color:T.accent,line:{color:'#000',width:1}},name:'Histograma'},{type:'box',x:vs,marker:{color:T.pos},boxpoints:'outliers',name:'Box'}],{title:{text:'DISTRIBUCION - '+label(s.metric),font:{size:12}},xaxis:axis({title:label(s.metric)}),yaxis:axis({title:'TRIALS'}),height:620,bargap:.04,showlegend:false,shapes:[{type:'line',x0:mean,x1:mean,yref:'paper',y0:0,y1:1,line:{color:T.accent2,width:1.5,dash:'dot'}}]});stats([['MEDIA',fmt(s.metric,mean)],['MEDIANA',fmt(s.metric,med)],['DESV',fmt(s.metric,std)],['MIN',fmt(s.metric,vs[0])],['MAX',fmt(s.metric,vs[vs.length-1])]])}
function renderSensitivity(){const s=State.sensitivity;const ms=D.trials.map(t=>value(t,s.metric));const rows=PARAM_KEYS.map(p=>({p,c:pearson(D.trials.map(t=>value(t,p)),ms)})).filter(r=>r.c!==null).sort((a,b)=>Math.abs(b.c)-Math.abs(a.c));plot([{type:'bar',orientation:'h',x:rows.map(r=>r.c),y:rows.map(r=>label(r.p)),marker:{color:rows.map(r=>r.c>=0?T.pos:T.neg)},text:rows.map(r=>r.c.toFixed(3)),textposition:'outside',hovertemplate:'%{y}<br>r=%{x:.4f}<extra></extra>'}],{title:{text:'SENSIBILIDAD - '+label(s.metric),font:{size:12}},xaxis:axis({title:'CORRELACION',range:[-1.1,1.1],zeroline:true,zerolinecolor:T.border}),yaxis:axis({automargin:true,categoryorder:'array',categoryarray:rows.slice().reverse().map(r=>label(r.p))}),height:Math.max(420,rows.length*38+130),margin:{l:170,r:50,t:48,b:60}});stats(rows.length?[['MAS INFLUYE',label(rows[0].p)],['R',rows[0].c.toFixed(3)]]:[]) }
function renderParallel(){const s=State.parallel;const dims=PARAM_KEYS.map(k=>({label:label(k),values:D.trials.map(t=>value(t,k))}));dims.push({label:'SCORE',values:D.trials.map(t=>t.score)});plot([{type:'parcoords',line:{color:D.trials.map(t=>value(t,s.color)),colorscale:colorscale(s.color),showscale:true,colorbar:{title:{text:label(s.color),font:{color:T.mute}},tickfont:{color:T.mute}}},dimensions:dims,labelfont:{color:T.text,size:10},tickfont:{color:T.mute,size:9},rangefont:{color:T.dim,size:9}}],{height:650,margin:{l:80,r:80,t:45,b:45}});stats([['EJES',String(dims.length)],['COLOR',label(s.color)]])}
initHeader();buildViews();renderCurrent();window.addEventListener('resize',()=>{const el=document.getElementById('chart');if(typeof Plotly!=='undefined'&&el&&el.data)Plotly.Plots.resize(el)});
</script>
</body>
</html>"""

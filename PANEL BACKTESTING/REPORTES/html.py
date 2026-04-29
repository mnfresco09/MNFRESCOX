from __future__ import annotations

import html as _html
import json
from datetime import date, datetime, timezone
from math import isfinite
from pathlib import Path

import polars as pl

from MOTOR.wrapper import MOTIVOS
from REPORTES.formatos import formatear_duracion
from REPORTES.tv_library import obtener_script_libreria


def generar_htmls(
    *,
    run_dir: Path,
    df: pl.DataFrame,
    df_indicadores: pl.DataFrame | None = None,
    trials: list,
    estrategia,
    max_plots: int,
    grafica_rango: str,
    grafica_desde: str,
    grafica_hasta: str,
) -> list[Path]:
    if max_plots <= 0:
        return []
    df_indicadores = df if df_indicadores is None else df_indicadores

    html_dir = _base_resultados(run_dir) / "GRAFICA"
    html_dir.mkdir(parents=True, exist_ok=True)
    # Sólo trials con replay materializado tienen trades para dibujar.
    candidatos = [t for t in trials if t.replay is not None]
    mejores = sorted(candidatos, key=lambda t: t.score, reverse=True)[: min(max_plots, 5)]
    tv_script = obtener_script_libreria()

    paths = []
    for trial in mejores:
        df_trial = _df_replay(trial, "df_tf", df)
        df_indicadores_trial = _df_replay(trial, "df_tf", df_indicadores)
        df_idx = df_trial.with_row_index("_i_")
        path = _unique_path(
            html_dir
            / (
                f"TRIAL {int(trial.numero)} - {_score_nombre(trial.score)}.html"
            )
        )
        payload = _crear_payload(
            df=df_trial,
            df_idx=df_idx,
            df_indicadores=df_indicadores_trial,
            trial=trial,
            estrategia=estrategia,
            indicadores_precalculados=getattr(trial.replay, "indicadores", None),
            grafica_rango=grafica_rango,
            grafica_desde=grafica_desde,
            grafica_hasta=grafica_hasta,
        )
        path.write_text(_render_html(payload, tv_script), encoding="utf-8")
        paths.append(path)

    verificar_htmls(paths)
    return paths


def _df_replay(trial, atributo: str, fallback: pl.DataFrame) -> pl.DataFrame:
    replay = getattr(trial, "replay", None)
    if replay is None:
        return fallback
    df = getattr(replay, atributo, None)
    return fallback if df is None else df


def verificar_htmls(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise ValueError(f"[HTML] No se genero {path}.")
        contenido = path.read_text(encoding="utf-8")
        tokens = (
            "candles",
            "markers",
            "createChart",
            "addCandlestickSeries",
            "chart-equity-dd",
            "addEquityDrawdownPane",
            "btn-reset-view",
            "resetToDefaultView",
            "paneMarkerSeries",
            "trade-tbody",
        )
        for token in tokens:
            if token not in contenido:
                raise ValueError(f"[HTML] {path.name} no contiene bloque requerido: {token}.")
        if "exit-strip" in contenido or "var-exit-stats" in contenido:
            raise ValueError(f"[HTML] {path.name} conserva el bloque legacy de TP/SL/END.")


def _crear_payload(
    *,
    df: pl.DataFrame,
    df_idx: pl.DataFrame,
    df_indicadores: pl.DataFrame,
    trial,
    estrategia,
    indicadores_precalculados: list[dict] | None,
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
    candle_times = [int(c["time"]) for c in candles]

    # Indicadores: calculados sobre el df completo, filtrados al conjunto exacto
    # de timestamps de las velas visibles. Restringir por `set(candle_times)` (no
    # por rango ts_min/ts_max) garantiza que el espacio lógico de los subpaneles
    # coincida exactamente con el del gráfico principal y elimina puntos huérfanos
    # que TradingView ancla al borde izquierdo desincronizando la timescale.
    indicadores_raw = (
        indicadores_precalculados
        if indicadores_precalculados is not None
        else estrategia.indicadores_para_grafica(df_indicadores, trial.parametros)
    )
    candle_times_set = set(candle_times)
    indicadores = [
        {**ind, "data": [d for d in ind["data"] if int(d["t"]) in candle_times_set]}
        for ind in indicadores_raw
    ]

    # Trades y markers — leídos directamente desde columnas numpy del replay.
    trades = []
    markers = []
    cols = trial.replay.trades
    n_trades = int(cols["idx_salida"].shape[0])
    motivos_lookup = MOTIVOS

    for i in range(n_trades):
        idx_signal = int(cols["idx_senal"][i])
        idx_e = int(cols["idx_entrada"][i])
        idx_s = int(cols["idx_salida"][i])
        t_signal = _segundos_desde_us(cols["ts_senal"][i])
        t_e = _segundos_desde_us(cols["ts_entrada"][i])
        t_s = _segundos_desde_us(cols["ts_salida"][i])
        dir_int = int(cols["direccion"][i])
        direccion = "LONG" if dir_int == 1 else "SHORT"
        pnl = float(cols["pnl"][i])
        comision = float(cols["comision_total"][i])
        pnl_bruto = pnl + comision
        roi = float(cols["roi"][i])
        precio_e = float(cols["precio_entrada"][i])
        precio_s = float(cols["precio_salida"][i])
        colateral = float(cols["colateral"][i])
        apalancamiento = float(cols["apalancamiento"][i])
        tamano_posicion = float(cols["tamano_posicion"][i])
        risk_vol_ewma = float(cols["risk_vol_ewma"][i])
        risk_sl_dist_pct = float(cols["risk_sl_dist_pct"][i])
        saldo_post = float(cols["saldo_post"][i])
        duracion_velas = int(cols["duracion_velas"][i])
        duracion_seg = _segundos_entre_us(
            int(cols["ts_entrada"][i]),
            int(cols["ts_salida"][i]),
        )
        duracion_txt = formatear_duracion(duracion_seg, duracion_velas)
        motivo_codigo = int(cols["motivo_salida"][i])
        motivo = motivos_lookup[motivo_codigo]
        n = i + 1

        trade = {
            "n":              n,
            "idx_senal":      idx_signal,
            "idx_entrada":    idx_e,
            "idx_salida":     idx_s,
            "direccion":      direccion,
            "time_senal":     t_signal,
            "time_entrada":   t_e,
            "time_salida":    t_s,
            "precio_entrada": precio_e,
            "precio_salida":  precio_s,
            "colateral":      colateral,
            "apalancamiento":  apalancamiento,
            "tamano_posicion": tamano_posicion,
            "risk_vol_ewma":   risk_vol_ewma,
            "risk_sl_dist_pct": risk_sl_dist_pct,
            "comision_total": comision,
            "pnl_bruto":      pnl_bruto,
            "pnl":            pnl,
            "roi":            roi,
            "saldo_post":     saldo_post,
            "duracion":       duracion_velas,
            "duracion_seg":   duracion_seg,
            "duracion_txt":   duracion_txt,
            "motivo_codigo":  motivo_codigo,
            "motivo":         motivo,
        }
        trades.append(trade)

        if t_e and ts_min <= t_e <= ts_max:
            markers.append({
                "trade":        n,
                "tipo":         "entrada",
                "time":         t_e,
                "direccion":    direccion,
                "precio":       precio_e,
                "precio_entrada": precio_e,
                "precio_salida": precio_s,
                "colateral":    colateral,
                "apalancamiento": apalancamiento,
                "tamano_posicion": tamano_posicion,
                "risk_vol_ewma": risk_vol_ewma,
                "risk_sl_dist_pct": risk_sl_dist_pct,
                "comision_total": comision,
                "pnl_bruto":    pnl_bruto,
                "pnl":          pnl,
                "roi":          roi,
                "saldo_post":   saldo_post,
                "duracion":     duracion_velas,
                "duracion_seg": duracion_seg,
                "duracion_txt": duracion_txt,
                "motivo":       motivo,
            })
        if t_s and ts_min <= t_s <= ts_max:
            markers.append({
                "trade":        n,
                "tipo":         "salida",
                "time":         t_s,
                "direccion":    direccion,
                "precio":       precio_s,
                "precio_entrada": precio_e,
                "precio_salida": precio_s,
                "colateral":    colateral,
                "apalancamiento": apalancamiento,
                "tamano_posicion": tamano_posicion,
                "risk_vol_ewma": risk_vol_ewma,
                "risk_sl_dist_pct": risk_sl_dist_pct,
                "comision_total": comision,
                "pnl_bruto":    pnl_bruto,
                "pnl":          pnl,
                "roi":          roi,
                "saldo_post":   saldo_post,
                "duracion":     duracion_velas,
                "duracion_seg": duracion_seg,
                "duracion_txt": duracion_txt,
                "motivo":       motivo,
            })

    equity_drawdown = _crear_equity_drawdown(
        trial=trial,
        tiempos_candles=candle_times,
        ts_inicio_total=int(df_idx["timestamp"][0].timestamp()),
    )

    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "titulo":       f"{trial.activo} {trial.timeframe} | {trial.estrategia_nombre} | {trial.salida.tipo}",
        "activo":       str(trial.activo),
        "timeframe":    str(trial.timeframe),
        "timeframe_ejecucion": str(getattr(trial, "timeframe_ejecucion", trial.timeframe)),
        "estrategia":   {
            "id": int(trial.estrategia_id),
            "nombre": str(trial.estrategia_nombre),
        },
        "salida":       _salida_payload(trial.salida),
        "trial":        int(trial.numero),
        "score":        _json_safe(float(trial.score)),
        "metricas":     _json_safe(trial.metricas),
        "parametros":   _json_safe(trial.parametros),
        "conteo_senales": _conteo_payload(getattr(trial, "conteo_senales", None)),
        "conteo_salidas": _conteo_payload(getattr(trial, "conteo_salidas", None)),
        "rango": {
            "inicio": ts_min,
            "fin": ts_max,
            "velas": len(candles),
        },
        "resumen_trades": _resumen_trades(trades),
        "candles":      candles,
        "markers":      markers,
        "trades":       trades,
        "equity_drawdown": equity_drawdown,
        "indicadores":  indicadores,
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


def _segundos_entre_us(inicio_us: int, fin_us: int) -> int:
    return max(0, int((fin_us - inicio_us) / 1_000_000))


def _segundos_desde_us(valor_us) -> int:
    return int(int(valor_us) / 1_000_000)


def _salida_payload(salida) -> dict:
    return {
        "tipo": str(salida.tipo),
        "sl_pct": _json_safe(getattr(salida, "sl_pct", None)),
        "tp_pct": _json_safe(getattr(salida, "tp_pct", None)),
        "velas": _json_safe(getattr(salida, "velas", None)),
        "trail_act_pct": _json_safe(getattr(salida, "trail_act_pct", None)),
        "trail_dist_pct": _json_safe(getattr(salida, "trail_dist_pct", None)),
        "optimizar": bool(getattr(salida, "optimizar", False)),
        "sl_min": _json_safe(getattr(salida, "sl_min", None)),
        "sl_max": _json_safe(getattr(salida, "sl_max", None)),
        "tp_min": _json_safe(getattr(salida, "tp_min", None)),
        "tp_max": _json_safe(getattr(salida, "tp_max", None)),
        "velas_min": _json_safe(getattr(salida, "velas_min", None)),
        "velas_max": _json_safe(getattr(salida, "velas_max", None)),
    }


def _conteo_payload(conteo: dict[int, int] | None) -> dict | None:
    if conteo is None:
        return None
    return {
        "long": int(conteo.get(1, 0)),
        "short": int(conteo.get(-1, 0)),
        "neutral": int(conteo.get(0, 0)),
        "total": int(sum(conteo.values())),
    }


def _resumen_trades(trades: list[dict]) -> dict:
    total = len(trades)
    pnl_neto = sum(float(t["pnl"]) for t in trades)
    pnl_bruto = sum(float(t["pnl_bruto"]) for t in trades)
    comisiones = sum(float(t["comision_total"]) for t in trades)
    ganadores = sum(1 for t in trades if float(t["pnl"]) > 0)
    perdedores = sum(1 for t in trades if float(t["pnl"]) < 0)
    neutros = total - ganadores - perdedores
    por_motivo: dict[str, dict] = {}
    for trade in trades:
        motivo = str(trade["motivo"])
        bucket = por_motivo.setdefault(
            motivo,
            {"trades": 0, "pnl": 0.0, "pnl_bruto": 0.0, "comisiones": 0.0, "win": 0, "loss": 0},
        )
        pnl = float(trade["pnl"])
        bucket["trades"] += 1
        bucket["pnl"] += pnl
        bucket["pnl_bruto"] += float(trade["pnl_bruto"])
        bucket["comisiones"] += float(trade["comision_total"])
        if pnl > 0:
            bucket["win"] += 1
        elif pnl < 0:
            bucket["loss"] += 1

    return {
        "total": total,
        "ganadores": ganadores,
        "perdedores": perdedores,
        "neutros": neutros,
        "pnl_neto": pnl_neto,
        "pnl_bruto": pnl_bruto,
        "comisiones": comisiones,
        "win_rate": (ganadores / total) if total else 0.0,
        "por_motivo": por_motivo,
    }


def _crear_equity_drawdown(
    *,
    trial,
    tiempos_candles: list[int],
    ts_inicio_total: int,
) -> list[dict]:
    equity_curve = trial.replay.equity_curve
    if equity_curve.shape[0] == 0 or not tiempos_candles:
        return []

    cols = trial.replay.trades
    inicial = float(equity_curve[0])
    peak = inicial
    eventos = [{"time": int(ts_inicio_total), "saldo": inicial, "peak": peak}]
    for i in range(int(cols["idx_salida"].shape[0])):
        saldo = float(equity_curve[i + 1])
        peak = max(peak, saldo)
        ts = _segundos_desde_us(cols["ts_salida"][i])
        eventos.append({"time": int(ts), "saldo": saldo, "peak": peak})

    eventos = _deduplicar_por_tiempo(sorted(eventos, key=lambda p: p["time"]))
    salida: list[dict] = []
    evento_idx = 0
    estado = eventos[0]
    for ts in sorted(set(int(t) for t in tiempos_candles)):
        while evento_idx + 1 < len(eventos) and eventos[evento_idx + 1]["time"] <= ts:
            evento_idx += 1
            estado = eventos[evento_idx]

        punto = {"time": ts, "saldo": estado["saldo"], "peak": estado["peak"]}
        saldo = float(punto["saldo"])
        peak = max(float(punto["peak"]), inicial)
        equity_pct = ((saldo / inicial) - 1.0) * 100.0 if inicial else 0.0
        drawdown_pct = ((saldo - peak) / peak) * 100.0 if peak else 0.0
        salida.append(
            {
                "time": int(punto["time"]),
                "saldo": saldo,
                "equity_pct": equity_pct,
                "drawdown_pct": min(0.0, drawdown_pct),
            }
        )
    return salida


def _deduplicar_por_tiempo(puntos: list[dict]) -> list[dict]:
    por_tiempo = {int(p["time"]): p for p in puntos}
    return [por_tiempo[t] for t in sorted(por_tiempo)]


def _json_safe(valor):
    if valor is None or isinstance(valor, (str, bool, int)):
        return valor
    if isinstance(valor, float):
        return valor if isfinite(valor) else None
    if isinstance(valor, dict):
        return {str(k): _json_safe(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_json_safe(v) for v in valor]
    if hasattr(valor, "item"):
        return _json_safe(valor.item())
    return str(valor)


def _base_resultados(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    if run_dir.parent.name.upper() == "DATOS":
        return run_dir.parent.parent
    return run_dir


def _score_nombre(score: float) -> str:
    valor = f"{abs(float(score)):.6f}".rstrip("0").rstrip(".")
    return f"NEG {valor}" if float(score) < 0 else valor


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{idx:02d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"[HTML] No se pudo crear nombre unico para {path}.")


def _render_html(payload: dict, tv_script: str) -> str:
    titulo = _html.escape(payload["titulo"])
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    template = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ - Trial __TRIAL__</title>
<style>
:root{
  --bg:#000000;--panel:#0a0a0a;--panel-2:#101010;
  --border:#1f1f1f;--border-strong:#2a2a2a;
  --text:#e8e8e8;--text-mute:#9a9a9a;--text-dim:#646464;
  --accent:#ffb000;--accent-2:#ff8a00;
  --pos:#5cdb5c;--neg:#ff4d4d;
  --long:#3aa3ff;--short:#ff8a00;--trailing:#ffd84d;
  --grid:#0d0d0d;--topbar-fg:#000;
  --font-body:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;
  --font-mono:"JetBrains Mono","SFMono-Regular",Menlo,Consolas,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{min-height:100%;background:var(--bg)}
body{font-family:var(--font-body);background:var(--bg);color:var(--text);font-size:12px;-webkit-font-smoothing:antialiased}
.mono{font-family:var(--font-mono);font-feature-settings:"tnum" 1,"zero" 1}.num{text-align:right}
.pos{color:var(--pos)!important}.neg{color:var(--neg)!important}.long{color:var(--long)}.short{color:var(--short)}
.topbar{display:flex;align-items:center;gap:12px;background:var(--accent);color:var(--topbar-fg);padding:3px 14px;font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;font-family:var(--font-mono);border-bottom:2px solid #000}
.topbar .brand{margin-right:8px}.topbar .cmd{flex:1;opacity:.82}.topbar .clock{font-feature-settings:"tnum" 1}
.title-strip{display:flex;align-items:baseline;gap:18px;padding:7px 14px 6px;background:var(--panel);border-bottom:1px solid var(--border)}
.title-strip h1{font-family:var(--font-mono);font-size:12px;font-weight:600;letter-spacing:.05em;color:var(--text);text-transform:uppercase}
.title-strip .sub{font-family:var(--font-mono);font-size:10px;color:var(--text-mute);letter-spacing:.08em;text-transform:uppercase}
.title-strip .badge{margin-left:auto;font-family:var(--font-mono);font-size:10px;letter-spacing:.1em;padding:3px 9px;border:1px solid var(--accent);color:var(--accent);text-transform:uppercase}
.headline{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));background:var(--panel);border-bottom:1px solid var(--border)}
.m-cell{padding:7px 14px 8px;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:2px;position:relative;min-width:0}
.m-cell:last-child{border-right:none}.m-cell::before{content:"";position:absolute;left:0;top:0;height:100%;width:2px;background:var(--accent);opacity:0}.m-cell:nth-child(1)::before{opacity:1}
.m-label{font-family:var(--font-mono);font-size:9px;color:var(--text-mute);letter-spacing:.14em;font-weight:500;white-space:nowrap}
.m-val{font-family:var(--font-mono);font-size:17px;font-weight:700;letter-spacing:0;font-feature-settings:"tnum" 1,"zero" 1;white-space:nowrap}
.secondary{display:flex;flex-wrap:wrap;background:var(--panel-2);border-bottom:1px solid var(--border);padding:0}
.s-cell{display:flex;align-items:baseline;gap:7px;padding:5px 14px;border-right:1px solid var(--border);font-family:var(--font-mono);min-height:24px}
.s-k{color:var(--text-mute);font-size:9px;letter-spacing:.12em;white-space:nowrap}.s-v{color:var(--text);font-size:10.5px;font-weight:600;white-space:nowrap}
.params-row{display:flex;align-items:center;gap:8px;min-height:31px;padding:5px 14px;background:var(--bg);border-bottom:1px solid var(--border);font-family:var(--font-mono);overflow:hidden}
.params-row .lbl{font-size:9px;color:var(--text-mute);letter-spacing:.14em;margin-right:4px;white-space:nowrap}#var-params{display:flex;gap:6px;flex-wrap:wrap}
.param{display:flex;align-items:baseline;gap:6px;padding:2px 8px;background:var(--panel);border:1px solid var(--border);font-size:10px}.param span{color:var(--text-mute);text-transform:uppercase;letter-spacing:.06em;font-size:9px}.param b{color:var(--accent);font-weight:700}
.toolbar{display:flex;flex-wrap:wrap;gap:0;align-items:stretch;background:var(--panel);border-bottom:1px solid var(--border)}
.tool-group{display:flex;align-items:center;gap:0;border-right:1px solid var(--border)}.tool-group .lbl{font-family:var(--font-mono);font-size:9px;color:var(--text-mute);letter-spacing:.14em;padding:0 12px;white-space:nowrap}.tool-inline{display:flex;align-items:stretch;gap:0}
.tbtn{font-family:var(--font-mono);background:transparent;color:var(--text-mute);border:none;border-left:1px solid var(--border);padding:7px 11px;font-size:10px;letter-spacing:.08em;cursor:pointer;text-transform:uppercase;font-weight:600;white-space:nowrap}.tbtn:hover{background:var(--panel-2);color:var(--text)}.tbtn.active{color:var(--accent);background:#1a1100}.tbtn[data-toggle]:not(.active){color:var(--text-dim);text-decoration:line-through}
.legend-inline{display:flex;align-items:center;gap:14px;padding:0 14px;margin-left:auto;font-family:var(--font-mono);font-size:10px;color:var(--text-mute);letter-spacing:.06em;text-transform:uppercase;min-height:31px}.legend-inline .li{display:flex;align-items:center;gap:6px;white-space:nowrap}.legend-inline .ldot{width:8px;height:8px;border-radius:50%;flex:none}.legend-inline .ltri{width:0;height:0;border-left:4px solid transparent;border-right:4px solid transparent;flex:none}.legend-inline .ltri.up{border-bottom:7px solid var(--long)}.legend-inline .ltri.down{border-top:7px solid var(--short)}
#charts{background:var(--bg);position:relative;min-height:0;overflow:hidden;display:flex;flex-direction:column}#chart-price{position:relative;overflow:hidden;border-bottom:1px solid var(--border);height:clamp(500px,calc(100vh - 390px),740px)}
.pane-wrapper{position:relative;border-top:1px solid var(--border);overflow:hidden;background:var(--bg)}.pane-chart{position:relative;height:132px;overflow:hidden}.equity-dd-wrapper .pane-chart{height:230px}.pane-label{position:absolute;top:6px;left:10px;z-index:5;font-family:var(--font-mono);font-size:9.5px;color:var(--accent);background:transparent;padding:0;letter-spacing:.16em;font-weight:700;pointer-events:none;text-transform:uppercase}.pane-label .meta{color:var(--text-mute);margin-left:8px;font-weight:400;letter-spacing:.1em}.global-crosshair{position:absolute;top:0;bottom:0;width:0;border-left:1px dotted var(--accent);z-index:30;pointer-events:none;display:none;opacity:.95}
#tv-attr-logo,[id^="tv-attr-logo"],a[href*="tradingview.com"]{display:none!important}
#tooltip{position:absolute;display:none;pointer-events:none;background:#000;border:1px solid var(--accent);padding:0;font-family:var(--font-mono);font-size:11px;line-height:1.55;min-width:300px;max-width:360px;z-index:200;box-shadow:0 12px 30px rgba(0,0,0,.9)}.tt-head{font-weight:700;font-size:11px;padding:6px 10px;letter-spacing:.1em;border-bottom:1px solid var(--accent);background:#0a0a0a}.tt-grid{padding:8px 10px;display:grid;grid-template-columns:1fr 1fr;gap:4px 14px}.tt-row{display:flex;justify-content:space-between;gap:8px;font-size:10.5px}.tt-row span{color:var(--text-mute);letter-spacing:.05em;font-size:9.5px;text-transform:uppercase}.tt-row b{color:var(--text);font-weight:500;font-feature-settings:"tnum" 1}.tt-sep{display:none}
.table-section{background:var(--panel);border-top:1px solid var(--border-strong)}.table-header{display:flex;align-items:center;gap:12px;padding:8px 14px;border-bottom:1px solid var(--border)}.table-header h3{font-family:var(--font-mono);font-size:10px;letter-spacing:.14em;color:var(--accent);font-weight:700}.table-header .count{font-family:var(--font-mono);font-size:10px;color:var(--text-mute);white-space:nowrap}.table-filters{margin-left:auto;display:flex;gap:0;flex-wrap:wrap}.table-filters .tbtn{border-left:1px solid var(--border)}.table-wrap{max-height:340px;overflow:auto}.table-wrap::-webkit-scrollbar{width:8px;height:8px}.table-wrap::-webkit-scrollbar-thumb{background:var(--border-strong)}.table-wrap::-webkit-scrollbar-track{background:var(--bg)}
#trade-table{width:max-content;min-width:100%;border-collapse:collapse;font-size:11px}#trade-table th{background:var(--panel-2);color:var(--text-mute);font-weight:700;padding:7px 12px;text-align:left;font-size:9px;border-bottom:1px solid var(--border);font-family:var(--font-mono);letter-spacing:.08em;text-transform:uppercase;position:sticky;top:0;z-index:2;white-space:nowrap}#trade-table td{padding:5px 12px;border-bottom:1px solid #0a0a0a;font-family:var(--font-mono);font-feature-settings:"tnum" 1;white-space:nowrap}#trade-table td.num,#trade-table th.num{text-align:right}#trade-table tbody tr:hover td{background:#111}#trade-table tbody tr.win:hover td{background:#0a1a0d}#trade-table tbody tr.loss:hover td{background:#1a0a0a}
@media (max-width:900px){.headline{grid-template-columns:repeat(2,minmax(0,1fr))}#chart-price{height:560px}.legend-inline{margin-left:0;width:100%;border-top:1px solid var(--border)}}
</style>
</head>
<body>
<div class="topbar"><div class="brand">MNFRESCOX TERMINAL</div><div class="cmd">BACKTEST REPORT - V1</div><div class="clock" id="clock">--:--:-- UTC</div></div>
<div class="title-strip"><h1 id="var-title">-</h1><div class="sub" id="var-subtitle">-</div><div class="badge" id="var-trial">#-</div></div>
<div class="headline" id="var-headline"></div>
<div class="secondary" id="var-secondary"></div>
<div class="params-row"><span class="lbl">PARAMS</span><div id="var-params"></div></div>
<div class="toolbar">
  <div class="tool-group"><span class="lbl">VISTA</span><button class="tbtn" id="btn-reset-view" type="button">RESET</button></div>
  <div class="tool-group"><span class="lbl">MARKERS</span><button class="tbtn" data-toggle="entries">ENTRADAS</button><button class="tbtn" data-toggle="exits">SALIDAS</button><button class="tbtn" data-toggle="trailing">TRAILING</button></div>
  <div class="tool-group"><span class="lbl">CAPAS</span><div class="tool-inline" id="pane-buttons"></div></div>
  <div class="legend-inline"><div class="li"><span class="ltri up"></span>LONG</div><div class="li"><span class="ltri down"></span>SHORT</div><div class="li"><span class="ldot" style="background:var(--pos)"></span>WIN</div><div class="li"><span class="ldot" style="background:var(--neg)"></span>LOSS</div><div class="li"><span class="ldot" style="background:var(--trailing)"></span>TRAIL</div></div>
</div>
<div id="charts"><div id="chart-price"></div></div>
<div class="table-section"><div class="table-header"><h3>TRADES</h3><span class="count" id="trade-count">-</span><div class="table-filters" id="table-filters"></div></div><div class="table-wrap"><table id="trade-table"><thead><tr><th class="num">#</th><th>DIR</th><th>SEÑAL</th><th>ENTRADA</th><th class="num">P.E</th><th>SALIDA</th><th class="num">P.S</th><th class="num">COLLAT</th><th class="num">LEV</th><th class="num">SIZE</th><th class="num">COM</th><th class="num">PNL BR</th><th class="num">PNL NET</th><th class="num">ROI</th><th class="num">BALANCE</th><th>DUR</th><th>MOTIVO</th></tr></thead><tbody id="trade-tbody"></tbody></table></div></div>
<div id="tooltip"></div>
__TV_SCRIPT__
<script>
(function(){
'use strict';
window.TRIAL_DATA=__DATA_JSON__;
const D=window.TRIAL_DATA;
const T={chartBg:'#000000',border:'#1f1f1f',gridV:'#0d0d0d',gridH:'#0d0d0d',textMuted:'#9a9a9a',monoFont:'JetBrains Mono, SFMono-Regular, Menlo, Consolas, monospace',crosshair:'#ffb000',crosshairLabel:'#ffb000',up:'#5cdb5c',down:'#ff4d4d',entryLong:'#3aa3ff',entryShort:'#ff8a00',exitWin:'#5cdb5c',exitLoss:'#ff4d4d',exitTrailing:'#ffd84d',equityLine:'#5cdb5c',equityFillTop:'rgba(92,219,92,.45)',equityFillBottom:'rgba(92,219,92,.05)',ddLine:'#ff4d4d',ddFillBottom:'rgba(255,77,77,.05)'};
if(!D||!window.LightweightCharts){throw new Error('HTML report missing TRIAL_DATA or LightweightCharts');}
const num=v=>Number.isFinite(Number(v))?Number(v):0;
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtPct=(v,dec=2)=>(num(v)>=0?'+':'')+(num(v)*100).toFixed(dec)+'%';
const fmtPctU=(v,dec=2)=>(num(v)*100).toFixed(dec)+'%';
const fmtPctPoints=(v,dec=2)=>(num(v)>=0?'+':'')+num(v).toFixed(dec)+'%';
const fmtNum=(v,dec=2)=>(num(v)>=0?'+':'')+num(v).toFixed(dec);
const fmtMoney=(v,dec=2)=>(num(v)<0?'-':'')+'$'+Math.abs(num(v)).toLocaleString('en-US',{minimumFractionDigits:dec,maximumFractionDigits:dec});
const fmtMoney0=v=>(num(v)<0?'-':'')+'$'+Math.abs(num(v)).toLocaleString('en-US',{maximumFractionDigits:0});
const fmtTs=ts=>ts?new Date(ts*1000).toISOString().replace('T',' ').slice(0,16)+'Z':'-';
function fmtDuration(seconds,candles,text){if(text)return text;if(Number.isFinite(Number(seconds))&&Number(seconds)>0){const total=Math.round(Number(seconds));const d=Math.floor(total/86400),h=Math.floor((total%86400)/3600),m=Math.floor((total%3600)/60);if(d)return d+'d '+String(h).padStart(2,'0')+'h';if(h)return h+'h '+String(m).padStart(2,'0')+'m';if(m)return m+'m';return total+'s';}if(candles!=null)return String(candles)+' velas';return '-';}
function renderHeader(){
  document.getElementById('var-title').textContent=D.titulo;
  document.getElementById('var-subtitle').textContent=`Trial ${D.trial} - Score ${num(D.score).toFixed(4)} - Exec ${D.timeframe_ejecucion}`;
  document.getElementById('var-trial').textContent='#'+D.trial;
  const m=D.metricas||{};
  const headlineDefs=[['ROI',num(m.roi_total),v=>fmtPct(v),true],['PROFIT FACT',num(m.profit_factor),v=>v.toFixed(2),'pf'],['SHARPE',num(m.sharpe_ratio),v=>v.toFixed(2),'signed'],['MAX DD',num(m.max_drawdown),v=>'-'+fmtPctU(v),'dd'],['WIN RATE',num(m.win_rate),v=>fmtPctU(v),false],['EXPECT',num(m.expectancy),v=>fmtPct(v),true]];
  document.getElementById('var-headline').innerHTML=headlineDefs.map(([label,val,fmt,signed])=>{let cls='';if(signed===true||signed==='signed')cls=val>=0?'pos':'neg';else if(signed==='pf')cls=val>=1?'pos':'neg';else if(signed==='dd')cls='neg';return `<div class="m-cell"><div class="m-label">${esc(label)}</div><div class="m-val ${cls}">${esc(fmt(val))}</div></div>`;}).join('');
  const sig=D.conteo_senales||{};
  const secDefs=[['TRADES',num(m.total_trades).toLocaleString()],['LONG',num(m.trades_long).toLocaleString()],['SHORT',num(m.trades_short).toLocaleString()],['WIN',num(m.trades_ganadores).toLocaleString()],['LOSS',num(m.trades_perdedores).toLocaleString()],['T/DAY',num(m.trades_por_dia).toFixed(3)],['AVG DUR',fmtDuration(m.duracion_media_seg,m.duracion_media_velas)],['PNL NET',fmtMoney(m.pnl_total)],['PNL GROSS',fmtMoney(m.pnl_bruto_total)],['BAL FINAL',fmtMoney0(m.saldo_final)],['TF EXEC',D.timeframe_ejecucion],['VELAS',num(D.rango?.velas).toLocaleString()],['SIG L/S',`${num(sig.long).toLocaleString()}/${num(sig.short).toLocaleString()}`]];
  document.getElementById('var-secondary').innerHTML=secDefs.map(([k,v])=>`<div class="s-cell"><span class="s-k">${esc(k)}</span><span class="s-v">${esc(v)}</span></div>`).join('');
  const params={...(D.parametros||{}),exit_type:D.salida?.tipo,sl_pct:D.salida?.sl_pct,tp_pct:D.salida?.tp_pct,exit_velas:D.salida?.velas,trail_act_pct:D.salida?.trail_act_pct,trail_dist_pct:D.salida?.trail_dist_pct};
  document.getElementById('var-params').innerHTML=Object.entries(params).filter(([,v])=>v!==null&&v!==undefined).map(([k,v])=>`<div class="param"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
}
const chartsEl=document.getElementById('charts');
const globalCrosshair=document.createElement('div');globalCrosshair.className='global-crosshair';chartsEl.appendChild(globalCrosshair);
const allCharts=[];const overlaySeriesMap={};const paneSeriesMap={};const visState={entries:true,exits:true,trailing:true};let mainChart=null;let syncBusy=false;let syncFrame=0;let pendingRange=null;
const RIGHT_SCALE_WIDTH=108;
function baseOpts(el,h,main=false){return {layout:{background:{type:'solid',color:T.chartBg},textColor:T.textMuted,fontSize:11,fontFamily:T.monoFont},grid:{vertLines:{color:T.gridV},horzLines:{color:T.gridH}},crosshair:{mode:LightweightCharts.CrosshairMode.Normal,vertLine:{visible:main,color:T.crosshair,labelBackgroundColor:T.crosshairLabel,width:1,style:LightweightCharts.LineStyle.Dotted},horzLine:{visible:main,color:T.crosshair,labelBackgroundColor:T.crosshairLabel,width:1,style:LightweightCharts.LineStyle.Dotted}},rightPriceScale:{borderColor:T.border,scaleMargins:{top:0.06,bottom:0.06},minimumWidth:RIGHT_SCALE_WIDTH},timeScale:{visible:main,borderColor:T.border,timeVisible:true,secondsVisible:false,rightOffset:6},handleScroll:main?{mouseWheel:true,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false}:{mouseWheel:false,pressedMouseMove:false,horzTouchDrag:false,vertTouchDrag:false},handleScale:main?{mouseWheel:true,pinch:true,axisPressedMouseMove:true}:{mouseWheel:false,pinch:false,axisPressedMouseMove:false},width:el.clientWidth||chartsEl.clientWidth||1200,height:h||el.clientHeight||120};}
function validRange(range){return !!range&&range.from!=null&&range.to!=null;}
function mainVisibleRange(){const r=mainChart?.timeScale().getVisibleLogicalRange?.();return validRange(r)?r:null;}
function applyTimeRange(range,source){if(!validRange(range)||syncBusy)return;pendingRange={range,source};if(syncFrame)return;syncFrame=requestAnimationFrame(()=>{const job=pendingRange;pendingRange=null;syncFrame=0;if(!job||!validRange(job.range))return;syncBusy=true;allCharts.forEach(c=>{if(c!==job.source){try{c.timeScale().setVisibleLogicalRange(job.range);}catch(_){}}});syncBusy=false;});}
function registerChart(chart,main=false){allCharts.push(chart);if(!main){const r=mainVisibleRange();if(r){try{chart.timeScale().setVisibleLogicalRange(r);}catch(_){}}return;}chart.timeScale().subscribeVisibleLogicalRangeChange(range=>applyTimeRange(range,chart));}
function moveGlobalCrosshair(param){if(!param||!param.point||!param.time){globalCrosshair.style.display='none';return;}globalCrosshair.style.display='block';globalCrosshair.style.transform=`translateX(${param.point.x}px)`;}
const priceEl=document.getElementById('chart-price');
mainChart=LightweightCharts.createChart(priceEl,baseOpts(priceEl,priceEl.clientHeight||620,true));registerChart(mainChart,true);
const candleSeries=mainChart.addCandlestickSeries({upColor:T.up,downColor:T.down,borderVisible:false,wickUpColor:T.up,wickDownColor:T.down});candleSeries.setData(D.candles);
const timeAnchorData=(D.candles||[]).map(c=>({time:c.time}));
function addTimeAnchor(chart){const anchor=chart.addLineSeries({color:'rgba(0,0,0,0)',lineWidth:1,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false});anchor.setData(timeAnchorData);return anchor;}
function addToggleButton(key,label){const host=document.getElementById('pane-buttons');if(!host||host.querySelector(`[data-toggle="${key.replace(/"/g,'\\"')}"]`))return;visState[key]=true;const btn=document.createElement('button');btn.className='tbtn';btn.dataset.toggle=key;btn.textContent=label;host.appendChild(btn);}
function shortLabel(name){return String(name||'IND').replace(/\\(.+\\)/,'').slice(0,12).toUpperCase();}
(D.indicadores||[]).filter(i=>i.tipo==='overlay').forEach(ind=>{const s=mainChart.addLineSeries({color:ind.color,lineWidth:1,title:ind.nombre,lastValueVisible:true,priceLineVisible:false});s.setData((ind.data||[]).map(d=>({time:d.t,value:d.v})));overlaySeriesMap[ind.nombre]=s;addToggleButton(ind.nombre,shortLabel(ind.nombre));});
function isTrailing(m){return /TRAIL|TS\\b/i.test(m.motivo||'');}
function buildMarkers(scale){const s=Number(scale)||1;return (D.markers||[]).filter(m=>{if(m.tipo==='entrada'&&!visState.entries)return false;if(m.tipo==='salida'&&!visState.exits)return false;if(m.tipo==='salida'&&isTrailing(m)&&!visState.trailing)return false;return true;}).map(m=>{const isEntry=m.tipo==='entrada';const isLong=m.direccion==='LONG';return {time:m.time,position:isEntry?(isLong?'belowBar':'aboveBar'):(isLong?'aboveBar':'belowBar'),color:isEntry?(isLong?T.entryLong:T.entryShort):(isTrailing(m)?T.exitTrailing:(m.pnl>=0?T.exitWin:T.exitLoss)),shape:isEntry?(isLong?'arrowUp':'arrowDown'):'circle',text:'',size:1.38*s};}).sort((a,b)=>a.time-b.time);}
const PANE_MARKER_SCALE=0.78;
const paneMarkerSeries=[];
candleSeries.setMarkers(buildMarkers(1));
(D.indicadores||[]).filter(i=>i.tipo==='pane').forEach(ind=>{const wrapper=document.createElement('div');wrapper.className='pane-wrapper';wrapper.dataset.indicator=ind.nombre;chartsEl.appendChild(wrapper);const lbl=document.createElement('div');lbl.className='pane-label';lbl.textContent=ind.nombre;wrapper.appendChild(lbl);const paneEl=document.createElement('div');paneEl.className='pane-chart';wrapper.appendChild(paneEl);const paneChart=LightweightCharts.createChart(paneEl,baseOpts(paneEl,132,false));addTimeAnchor(paneChart);const paneSeries=paneChart.addLineSeries({color:ind.color,lineWidth:1.2,lastValueVisible:true,priceLineVisible:false});const paneData=(ind.data||[]).map(d=>({time:d.t,value:d.v}));paneSeries.setData(paneData);if(ind.min!==undefined&&ind.max!==undefined){paneSeries.applyOptions({autoscaleInfoProvider:()=>({priceRange:{minValue:ind.min,maxValue:ind.max},margins:{above:8,below:8}})});} (ind.niveles||[]).forEach(n=>paneSeries.createPriceLine({price:n.valor,color:n.color,lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true}));paneSeries.setMarkers(buildMarkers(PANE_MARKER_SCALE));paneMarkerSeries.push(paneSeries);registerChart(paneChart,false);paneSeriesMap[ind.nombre]={wrapper,chart:paneChart,el:paneEl};addToggleButton(ind.nombre,shortLabel(ind.nombre));});
function addEquityDrawdownPane(){
  const eqArr=D.equity_drawdown||[];
  const wrapper=document.createElement('div');wrapper.className='pane-wrapper equity-dd-wrapper';chartsEl.appendChild(wrapper);
  const lbl=document.createElement('div');lbl.className='pane-label';
  const last=eqArr.length?eqArr[eqArr.length-1]:{};
  const eqLast=num(last.equity_pct),ddLast=num(last.drawdown_pct);
  lbl.innerHTML=`EQUITY / DRAWDOWN <span class="meta">equity <span class="${eqLast>=0?'pos':'neg'}">${fmtPctPoints(eqLast)}</span> · dd <span class="neg">${fmtPctPoints(ddLast)}</span></span>`;
  wrapper.appendChild(lbl);
  const paneEl=document.createElement('div');paneEl.id='chart-equity-dd';paneEl.className='pane-chart';wrapper.appendChild(paneEl);
  const chart=LightweightCharts.createChart(paneEl,baseOpts(paneEl,230,false));
  addTimeAnchor(chart);
  // Eje Y simétrico alrededor de 0 — equity ocupa la mitad superior, drawdown la inferior.
  const eqMax=eqArr.reduce((m,p)=>Math.max(m,num(p.equity_pct)),0);
  const ddMin=eqArr.reduce((m,p)=>Math.min(m,num(p.drawdown_pct)),0);
  const yLimit=Math.ceil(Math.max(eqMax,Math.abs(ddMin),1)*1.10*100)/100;
  const sharedScale=()=>({priceRange:{minValue:-yLimit,maxValue:yLimit},margins:{above:0,below:0}});
  // Equity: BaselineSeries con base 0 — sólo la mitad superior queda rellena (verde),
  // la inferior se descarta clipeando a 0 los valores negativos.
  const equitySeries=chart.addBaselineSeries({baseValue:{type:'price',price:0},topLineColor:T.equityLine,topFillColor1:T.equityFillTop,topFillColor2:T.equityFillBottom,bottomLineColor:T.equityLine,bottomFillColor1:'rgba(0,0,0,0)',bottomFillColor2:'rgba(0,0,0,0)',lineWidth:2,lastValueVisible:true,priceLineVisible:false});
  equitySeries.setData(eqArr.map(p=>({time:p.time,value:Math.max(0,num(p.equity_pct))})));
  equitySeries.applyOptions({autoscaleInfoProvider:sharedScale});
  // Drawdown: BaselineSeries con base 0 — sólo la mitad inferior queda rellena (rojo),
  // estilo "Running Maximum Drawdown" (Fidelity-style).
  const drawdownSeries=chart.addBaselineSeries({baseValue:{type:'price',price:0},topLineColor:T.ddLine,topFillColor1:'rgba(0,0,0,0)',topFillColor2:'rgba(0,0,0,0)',bottomLineColor:T.ddLine,bottomFillColor1:'rgba(255,77,77,.45)',bottomFillColor2:T.ddFillBottom,lineWidth:2,lastValueVisible:true,priceLineVisible:false});
  drawdownSeries.setData(eqArr.map(p=>({time:p.time,value:Math.min(0,num(p.drawdown_pct))})));
  drawdownSeries.applyOptions({autoscaleInfoProvider:sharedScale});
  // Línea de cero compartida.
  equitySeries.createPriceLine({price:0,color:'#8a8a8a88',lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true});
  registerChart(chart,false);
  paneSeriesMap.EQUITY_DD={wrapper,chart,el:paneEl};
  addToggleButton('EQUITY_DD','EQUITY/DD');
}
addEquityDrawdownPane();
const tooltipEl=document.getElementById('tooltip');const byTime={};(D.markers||[]).forEach(m=>{if(!byTime[m.time])byTime[m.time]=[];byTime[m.time].push(m);});const tradeByN={};(D.trades||[]).forEach(t=>{tradeByN[t.n]=t;});
mainChart.subscribeCrosshairMove(param=>{moveGlobalCrosshair(param);if(!param.time||!param.point){tooltipEl.style.display='none';return;}const ms=byTime[param.time];if(!ms||!ms.length){tooltipEl.style.display='none';return;}tooltipEl.innerHTML=ms.map(m=>{const t=tradeByN[m.trade]||m;const pnl=num(t.pnl),roi=num(t.roi),pnlCls=pnl>=0?'pos':'neg';const isEntry=m.tipo==='entrada';const headColor=isEntry?(m.direccion==='LONG'?T.entryLong:T.entryShort):(isTrailing(m)?T.exitTrailing:(pnl>=0?T.exitWin:T.exitLoss));const headLabel=isEntry?`${m.direccion} ENTRADA #${m.trade}`:`SALIDA ${t.motivo||m.motivo} #${m.trade}`;return `<div class="tt-head" style="color:${headColor}">${esc(headLabel)}</div><div class="tt-grid"><div class="tt-row"><span>P. ENTRADA</span><b>${fmtMoney(t.precio_entrada,2)}</b></div><div class="tt-row"><span>P. SALIDA</span><b>${fmtMoney(t.precio_salida,2)}</b></div><div class="tt-row"><span>COLLATERAL</span><b>${fmtMoney(t.colateral,2)}</b></div><div class="tt-row"><span>LEV</span><b>${num(t.apalancamiento).toFixed(2)}x</b></div><div class="tt-row"><span>SIZE</span><b>${num(t.tamano_posicion).toFixed(6)}</b></div><div class="tt-row"><span>VOL EWMA</span><b>${fmtPct(t.risk_vol_ewma)}</b></div><div class="tt-row"><span>COMISION</span><b>${fmtMoney(t.comision_total,2)}</b></div><div class="tt-row"><span>SL DIST</span><b>${fmtPct(t.risk_sl_dist_pct)}</b></div><div class="tt-row"><span>PNL BRUTO</span><b>${fmtNum(t.pnl_bruto,2)}</b></div><div class="tt-row"><span>PNL NETO</span><b class="${pnlCls}">${fmtNum(pnl,2)}</b></div><div class="tt-row"><span>ROI</span><b class="${pnlCls}">${fmtPct(roi)}</b></div><div class="tt-row"><span>BALANCE</span><b>${fmtMoney(t.saldo_post,2)}</b></div><div class="tt-row"><span>DURACION</span><b>${esc(t.duracion_txt||fmtDuration(t.duracion_seg,t.duracion))}</b></div></div>`;}).join('');tooltipEl.style.display='block';let lx=param.point.x+16,ly=param.point.y+8;const tw=tooltipEl.offsetWidth||300;if(lx+tw>priceEl.clientWidth-10)lx=param.point.x-tw-8;tooltipEl.style.left=lx+'px';tooltipEl.style.top=ly+'px';});
let tableFilter='ALL';
function renderFilterButtons(){const host=document.getElementById('table-filters');const motivos=[...new Set((D.trades||[]).map(t=>t.motivo).filter(Boolean))];const defs=['ALL','LONG','SHORT','WIN','LOSS',...motivos];host.innerHTML=defs.map((f,i)=>`<button class="tbtn ${i===0?'active':''}" data-filter="${esc(f)}">${esc(f)}</button>`).join('');}
function renderTable(){let trades=D.trades||[];if(tableFilter==='LONG')trades=trades.filter(t=>t.direccion==='LONG');else if(tableFilter==='SHORT')trades=trades.filter(t=>t.direccion==='SHORT');else if(tableFilter==='WIN')trades=trades.filter(t=>num(t.pnl)>=0);else if(tableFilter==='LOSS')trades=trades.filter(t=>num(t.pnl)<0);else if(tableFilter!=='ALL')trades=trades.filter(t=>t.motivo===tableFilter);document.getElementById('trade-tbody').innerHTML=trades.map(t=>{const pnlCls=num(t.pnl)>=0?'pos':'neg';const dirCls=t.direccion==='LONG'?'long':'short';return `<tr class="${num(t.pnl)>=0?'win':'loss'}"><td class="num">${esc(t.n)}</td><td class="${dirCls}">${esc(t.direccion)}</td><td class="mono">${esc(fmtTs(t.time_senal))}</td><td class="mono">${esc(fmtTs(t.time_entrada))}</td><td class="mono num">${num(t.precio_entrada).toFixed(2)}</td><td class="mono">${esc(fmtTs(t.time_salida))}</td><td class="mono num">${num(t.precio_salida).toFixed(2)}</td><td class="mono num">${fmtMoney(t.colateral,2)}</td><td class="mono num">${num(t.apalancamiento).toFixed(2)}x</td><td class="mono num">${num(t.tamano_posicion).toFixed(6)}</td><td class="mono num">${fmtMoney(t.comision_total,2)}</td><td class="mono num ${num(t.pnl_bruto)>=0?'pos':'neg'}">${fmtNum(t.pnl_bruto,2)}</td><td class="mono num ${pnlCls}">${fmtNum(t.pnl,2)}</td><td class="mono num ${pnlCls}">${fmtPct(t.roi)}</td><td class="mono num">${fmtMoney(t.saldo_post,2)}</td><td class="mono">${esc(t.duracion_txt||fmtDuration(t.duracion_seg,t.duracion))}</td><td>${esc(t.motivo)}</td></tr>`;}).join('');document.getElementById('trade-count').textContent=`${trades.length} / ${(D.trades||[]).length}`;}
function wireFilters(){document.querySelectorAll('[data-filter]').forEach(btn=>btn.addEventListener('click',()=>{tableFilter=btn.dataset.filter;document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b===btn));renderTable();}));}
function applyMarkerVisibility(){candleSeries.setMarkers(buildMarkers(1));paneMarkerSeries.forEach(s=>s.setMarkers(buildMarkers(PANE_MARKER_SCALE)));}
function applyPaneVisibility(key){const entry=paneSeriesMap[key];if(!entry)return;entry.wrapper.style.display=visState[key]?'':'none';setTimeout(()=>{fitAll();const r=mainVisibleRange();if(r)applyTimeRange(r,mainChart);},30);}
function applyOverlayVisibility(key){const s=overlaySeriesMap[key];if(s&&typeof s.applyOptions==='function')s.applyOptions({visible:!!visState[key]});}
function wireToggles(){document.querySelectorAll('[data-toggle]').forEach(btn=>{const key=btn.dataset.toggle;if(!(key in visState))visState[key]=true;btn.classList.toggle('active',!!visState[key]);btn.addEventListener('click',()=>{visState[key]=!visState[key];btn.classList.toggle('active',visState[key]);if(key==='entries'||key==='exits'||key==='trailing')applyMarkerVisibility();else if(paneSeriesMap[key])applyPaneVisibility(key);else if(overlaySeriesMap[key])applyOverlayVisibility(key);});});}
function fitAll(){const priceW=priceEl.clientWidth||chartsEl.clientWidth;mainChart.applyOptions({width:priceW,height:priceEl.clientHeight||620});Object.values(paneSeriesMap).forEach(({chart,el})=>chart.applyOptions({width:(el?.clientWidth||priceW),height:(el?.clientHeight||120)}));}
const DEFAULT_VISIBLE_BARS=300;
function resetToDefaultView(){const total=(D.candles||[]).length;if(!total)return;const from=Math.max(0,total-DEFAULT_VISIBLE_BARS);const range={from,to:total-1+6};try{mainChart.timeScale().setVisibleLogicalRange(range);}catch(_){}applyTimeRange(range,mainChart);}
function wireResetButton(){const btn=document.getElementById('btn-reset-view');if(btn)btn.addEventListener('click',resetToDefaultView);}
const ro=new ResizeObserver(fitAll);ro.observe(chartsEl);window.addEventListener('resize',fitAll);
function tick(){const d=new Date(),pad=n=>String(n).padStart(2,'0');document.getElementById('clock').textContent=pad(d.getUTCHours())+':'+pad(d.getUTCMinutes())+':'+pad(d.getUTCSeconds())+' UTC';}
renderHeader();renderFilterButtons();renderTable();wireFilters();wireToggles();wireResetButton();fitAll();setTimeout(()=>{fitAll();resetToDefaultView();},50);setInterval(tick,1000);tick();
})();
</script>
</body>
</html>"""
    return (
        template
        .replace("__TITLE__", titulo)
        .replace("__TRIAL__", str(int(payload["trial"])))
        .replace("__TV_SCRIPT__", tv_script)
        .replace("__DATA_JSON__", data_json)
    )

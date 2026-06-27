"""Reporte PDF de rendimiento — estilo prop firm.

Genera un fichero PDF en landscape A4 con:
  Pagina 1:
    - Cabecera (estrategia, activo, timeframe, periodo, score)
    - Aviso IS (los parametros se optimizaron sobre este mismo dato)
    - Fila de parametros del trial
    - Grid de KPIs (18 metricas en 3 filas x 6 columnas)
    - Tabla de estadisticas en dos columnas (izquierda: estrategia; derecha: trades)
  Pagina 2:
    - Grafico equity curve + drawdown (matplotlib → imagen embebida)
    - Tabla de rentabilidad mensual con colores rojo/verde

Dependencias externas (deben estar instaladas en el entorno):
  pip install reportlab matplotlib

Fuentes de datos: analitica_avanzada() + tabla_monthly_returns() de analitica.py.
No se duplcan calculos: este modulo solo formatea lo que analitica.py ya calculo.
"""

from __future__ import annotations

import io
import math
from datetime import date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Intentar importar dependencias con mensaje claro si faltan
# ---------------------------------------------------------------------------
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _RL_OK = True
except ImportError:
    _RL_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from REPORTES.analitica import analitica_avanzada, tabla_monthly_returns, distribuciones

# ---------------------------------------------------------------------------
# Paleta de colores
# ---------------------------------------------------------------------------
_NAVY    = colors.HexColor("#0F172A")
_NAVY2   = colors.HexColor("#1E293B")
_SLATE   = colors.HexColor("#334155")
_MUTED   = colors.HexColor("#64748B")
_TEXT    = colors.HexColor("#F1F5F9")
_GREEN   = colors.HexColor("#16A34A")
_GREEN_L = colors.HexColor("#DCFCE7")
_RED     = colors.HexColor("#DC2626")
_RED_L   = colors.HexColor("#FEE2E2")
_AMBER   = colors.HexColor("#B45309")
_AMBER_L = colors.HexColor("#FEF3C7")
_BLUE    = colors.HexColor("#2563EB")
_BLUE_L  = colors.HexColor("#DBEAFE")
_WHITE   = colors.white
_LGRAY   = colors.HexColor("#F8FAFC")
_DGRAY   = colors.HexColor("#E2E8F0")


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def generar_reporte_pdf(
    *,
    run_dir: Path,
    trial,
    estrategia,
    activo: str,
    timeframe: str,
    fecha_inicio: date,
    fecha_fin: date,
) -> Path:
    """Genera el PDF del mejor trial y lo guarda en run_dir/PDF/.

    Si reportlab o matplotlib no estan instalados, lanza ImportError con
    instrucciones de instalacion claras.
    """
    if not _RL_OK:
        raise ImportError(
            "[PDF] reportlab no esta instalado. "
            "Instalar con: pip install reportlab"
        )
    if not _MPL_OK:
        raise ImportError(
            "[PDF] matplotlib no esta instalado. "
            "Instalar con: pip install matplotlib"
        )

    if trial.replay is None:
        raise ValueError("[PDF] El trial no tiene replay materializado.")

    replay   = trial.replay
    an       = analitica_avanzada(
        metricas=trial.metricas,
        trades=replay.trades,
        equity_curve=replay.equity_curve,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
    )
    monthly  = tabla_monthly_returns(
        replay.trades,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
    )
    dist     = distribuciones(replay.trades)

    pdf_dir  = run_dir / "PDF"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    path     = _unique_path(pdf_dir / f"REPORTE TRIAL {int(trial.numero)}.pdf")

    _build_pdf(
        path=path,
        trial=trial,
        an=an,
        monthly=monthly,
        dist=dist,
        estrategia=estrategia,
        activo=activo,
        timeframe=timeframe,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        equity=np.asarray(replay.equity_curve, dtype=float),
        ts_salida=replay.trades.get("ts_salida"),
        pnl=np.asarray(replay.trades.get("pnl", []), dtype=float),
    )
    return path


# ---------------------------------------------------------------------------
# Constructor del PDF
# ---------------------------------------------------------------------------

def _build_pdf(
    *,
    path: Path,
    trial,
    an: dict[str, Any],
    monthly: dict,
    dist: dict,
    estrategia,
    activo: str,
    timeframe: str,
    fecha_inicio: date,
    fecha_fin: date,
    equity,
    ts_salida,
    pnl,
) -> None:
    PAGE  = landscape(A4)
    W, H  = PAGE

    doc   = SimpleDocTemplate(
        str(path),
        pagesize=PAGE,
        leftMargin=1.2*cm, rightMargin=1.2*cm,
        topMargin=1.2*cm,  bottomMargin=1.2*cm,
        title=f"Reporte {estrategia.NOMBRE} — {activo}",
        author="Sistema de Backtesting",
    )

    story = []
    m     = trial.metricas
    params= trial.parametros

    # ── Pagina 1 ──────────────────────────────────────────────────────────
    story += _bloque_cabecera(
        estrategia=estrategia, activo=activo, timeframe=timeframe,
        fecha_inicio=fecha_inicio, fecha_fin=fecha_fin,
        trial=trial, W=W,
    )
    story.append(Spacer(1, 3*mm))
    story.append(_aviso_is(W))
    story.append(Spacer(1, 2*mm))
    story += _bloque_params(params, W)
    story.append(Spacer(1, 3*mm))
    story += _bloque_kpis(m, an, W)
    story.append(Spacer(1, 4*mm))
    story += _bloque_stats(m, an, dist, W)
    story.append(PageBreak())

    # ── Pagina 2 ──────────────────────────────────────────────────────────
    story += _bloque_charts(equity=equity, ts_salida=ts_salida, pnl=pnl, W=W)
    story.append(Spacer(1, 5*mm))
    story += _bloque_monthly(monthly, W)
    story.append(Spacer(1, 4*mm))
    story.append(_footer())

    doc.build(story)


# ---------------------------------------------------------------------------
# Bloques de contenido
# ---------------------------------------------------------------------------

def _bloque_cabecera(
    *, estrategia, activo, timeframe, fecha_inicio, fecha_fin, trial, W,
) -> list:
    score = f"{trial.score:.4f}"
    periodo = f"{fecha_inicio.strftime('%d %b %Y')}  →  {fecha_fin.strftime('%d %b %Y')}"

    data = [[
        Paragraph(f"<b>{estrategia.NOMBRE}</b>",
                  _ps(16, _TEXT, bold=True)),
        Paragraph(f"{activo} · {timeframe} · Trial #{trial.numero}",
                  _ps(10, _MUTED)),
        Paragraph(periodo, _ps(10, _MUTED)),
        Paragraph(f"SCORE  {score}", _ps(11, _BLUE, bold=True)),
    ]]
    t = Table(data, colWidths=[W*0.35, W*0.22, W*0.22, W*0.15])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _NAVY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_NAVY]),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 2, _BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return [t]


def _aviso_is(W) -> Table:
    txt = Paragraph(
        "<b>⚠  Resultados IN-SAMPLE.</b>  Los parametros se seleccionaron "
        "optimizando sobre este mismo periodo historico. La ejecucion barra "
        "a barra es libre de lookahead (entrada en open de la vela siguiente "
        "a la senal), pero las metricas reflejan el mejor ajuste observado. "
        "Validar con Walk-Forward Analysis para obtener metricas fuera de muestra.",
        _ps(8, _AMBER),
    )
    t = Table([[txt]], colWidths=[W - 2.4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _AMBER_L),
        ("LINEABOVE",     (0, 0), (-1, 0),  1.5, _AMBER),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.5, _AMBER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    return t


def _bloque_params(params: dict, W) -> list:
    partes = []
    for k, v in params.items():
        label = k.replace("exit_", "").replace("_", " ").upper()
        val   = f"{v:.2f}" if isinstance(v, float) else str(v)
        partes.append(f"<b>{label}</b>: {val}")
    txt = "  ·  ".join(partes) if partes else "—"
    t = Table(
        [[Paragraph(txt, _ps(8, _SLATE))]],
        colWidths=[W - 2.4*cm],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _LGRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 0.5, _DGRAY),
    ]))
    return [t]


def _bloque_kpis(m: dict, an: dict, W) -> list:
    """Tres filas de 6 KPI cards cada una."""
    pnl_total  = float(m.get("pnl_total",  0))
    n_trades   = int(m.get("total_trades", 0))
    win_rate   = float(m.get("win_rate",   0))
    profit_f   = float(m.get("profit_factor", 0))
    max_dd_pct = float(m.get("max_drawdown", 0))
    roi_total  = float(m.get("roi_total",  0))

    filas = [
        [
            ("PROFIT TOTAL",   _usd(pnl_total),                    _sign_c(pnl_total), True),
            ("# TRADES",       str(n_trades),                       None,   False),
            ("SHARPE ANUAL",   f"{an['sharpe_anualizado']:.2f}",    _sharpe_c(an["sharpe_anualizado"]), False),
            ("PROFIT FACTOR",  f"{profit_f:.2f}",                   _pf_c(profit_f),    False),
            ("RETURN / DD",    f"{an['return_dd_ratio']:.1f}",      None,   False),
            ("WIN RATE",       f"{win_rate*100:.1f}%",              _wr_c(win_rate),     False),
        ],
        [
            ("DRAWDOWN $",     _usd(an["max_drawdown_money"]),      _RED,   False),
            ("DRAWDOWN %",     f"{max_dd_pct*100:.2f}%",           _RED,   False),
            ("DAILY AVG",      _usd(an["daily_avg_profit"]),        _sign_c(an["daily_avg_profit"]), False),
            ("MONTHLY AVG",    _usd(an["monthly_avg_profit"]),      _sign_c(an["monthly_avg_profit"]), False),
            ("AVG TRADE",      _usd(float(m.get("expectancy", 0))), _sign_c(float(m.get("expectancy", 0))), False),
            ("CAGR",           f"{an['cagr']*100:.2f}%",           _sign_c(an["cagr"]), False),
        ],
        [
            ("ANNUAL% / DD%",  f"{an['annual_pct_over_maxdd']:.2f}", None,  False),
            ("R EXPECTANCY",   f"{an['r_expectancy']:.3f}",         _sign_c(an["r_expectancy"]), False),
            ("R EXP SCORE",    f"{an['r_expectancy_score']:.2f}",   None,   False),
            ("SQN",            f"{an['sqn']:.2f}",                  _sqn_c(an["sqn"]), False),
            ("PSR",            f"{an['psr']:.3f}",                  _psr_c(an["psr"]), False),
            ("CALMAR",         f"{an['calmar_ratio']:.2f}",         _sign_c(an["calmar_ratio"]), False),
        ],
    ]

    result = [Paragraph("RENDIMIENTO", _ps(7, _MUTED, bold=True))]
    cw = (W - 2.4*cm) / 6

    for fila in filas:
        row_data = []
        for label, value, color, big in fila:
            c = color if color else _SLATE
            sz = 14 if big else 12
            cell = [
                Paragraph(label,  _ps(6,  _MUTED)),
                Paragraph(value,  _ps(sz, c, bold=True)),
            ]
            row_data.append(cell)

        t = Table([row_data], colWidths=[cw] * 6)
        t.setStyle(TableStyle([
            ("BOX",           (0, 0), (-1, -1), 0.5, _DGRAY),
            ("INNERGRID",     (0, 0), (-1, -1), 0.5, _DGRAY),
            ("BACKGROUND",    (0, 0), (-1, -1), _WHITE),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        result.append(t)

    return result


def _bloque_stats(m: dict, an: dict, dist: dict, W) -> list:
    """Dos tablas lado a lado: estadisticas de estrategia y de trades."""
    cw_total = W - 2.4*cm
    cw = cw_total / 2

    # ── Estadisticas de estrategia ────────────────────────────────────────
    wins_losses = f"{an['wins_losses_ratio']:.2f}" if math.isfinite(an["wins_losses_ratio"]) else "∞"
    strat_rows = [
        ["Wins / Losses Ratio",    wins_losses,
         "Payoff (Avg Win/Loss)",   f"{an['payoff_ratio']:.2f}"],
        ["AHPR %",                  f"{an['ahpr']:.2f}%",
         "GHPR %",                  f"{an['ghpr']:.2f}%"],
        ["Expectancy",              _usd(float(m.get("expectancy", 0))),
         "Desv. Estandar PnL",      _usd(float(np.std([])))  # se calcula abajo
         ],
        ["Z-Score",                 f"{an['z_score']:.2f}",
         "Z-Probability",           f"{an['z_probability']:.2f}%"],
        ["Stagnation (dias)",       f"{an['stagnation_days']:.0f}",
         "Stagnation %",            f"{an['stagnation_pct']:.2f}%"],
        ["Exposicion",              f"{an['exposure']*100:.2f}%",
         "Avg Velas / Trade",       f"{float(m.get('duracion_media_velas',0)):.2f}"],
        ["Skewness PnL",            f"{an['skew']:.3f}",
         "Kurtosis (exceso)",       f"{an['kurtosis']:.3f}"],
        ["Recovery Factor",         f"{an['recovery_factor']:.2f}",
         "Sortino Anual",           f"{an['sortino_anualizado']:.2f}"],
        ["SQN",                     f"{an['sqn']:.2f}",
         "PSR",                     f"{an['psr']:.3f}"],
    ]

    # ── Estadisticas de trades ────────────────────────────────────────────
    motivos = dist.get("motivo", {})
    motivos_txt = "  ".join(f"{k}:{v}" for k, v in sorted(
        motivos.items(), key=lambda x: -x[1])[:4])

    trade_rows = [
        ["# Ganadores",    f"{int(m.get('trades_ganadores',0))}",
         "# Perdedores",   f"{int(m.get('trades_perdedores',0))}"],
        ["# Long",         f"{int(m.get('trades_long',0))}",
         "# Short",        f"{int(m.get('trades_short',0))}"],
        ["Gross Profit",   _usd(an["gross_profit"]),
         "Gross Loss",     _usd(an["gross_loss"])],
        ["Avg Win",        _usd(an["avg_win"]),
         "Avg Loss",       _usd(an["avg_loss"])],
        ["Mejor Trade",    _usd(an["best_trade"]),
         "Peor Trade",     _usd(an["worst_trade"])],
        ["Max Consec Wins", f"{an['max_win_streak']}",
         "Max Consec Loss", f"{an['max_loss_streak']}"],
        ["Avg Consec Wins", f"{an['avg_consec_wins']:.1f}",
         "Avg Consec Loss", f"{an['avg_consec_losses']:.1f}"],
        ["Avg Velas (Win)", f"{an['avg_bars_wins']:.2f}",
         "Avg Velas (Loss)", f"{an['avg_bars_losses']:.2f}"],
        ["Motivos salida",  motivos_txt, "", ""],
    ]

    left  = _tabla_stats("ESTADISTICAS ESTRATEGIA", strat_rows, cw)
    right = _tabla_stats("ESTADISTICAS TRADES",     trade_rows, cw)

    contenedor = Table([[left, right]], colWidths=[cw, cw])
    contenedor.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [contenedor]


def _tabla_stats(titulo: str, rows: list[list[str]], cw: float) -> Table:
    """Construye una tabla de stats con cabecera de titulo."""
    w4 = cw / 4
    col_w = [w4 * 1.4, w4 * 0.8, w4 * 1.4, w4 * 0.8]

    header = [[
        Paragraph(titulo, _ps(7, _TEXT, bold=True)),
        "", "", "",
    ]]
    data_rows = []
    for r in rows:
        if len(r) == 4:
            data_rows.append([
                Paragraph(str(r[0]), _ps(7, _MUTED)),
                Paragraph(str(r[1]), _ps(7, _SLATE, bold=True)),
                Paragraph(str(r[2]), _ps(7, _MUTED)),
                Paragraph(str(r[3]), _ps(7, _SLATE, bold=True)),
            ])
        else:
            data_rows.append([Paragraph(str(r[0]), _ps(7, _MUTED)), "", "", ""])

    all_rows = header + data_rows
    t = Table(all_rows, colWidths=col_w)

    style = [
        ("BACKGROUND",  (0, 0), (-1, 0),  _NAVY),
        ("SPAN",        (0, 0), (-1, 0)),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",(0, 0), (-1, -1), 5),
        ("BOX",         (0, 0), (-1, -1), 0.5, _DGRAY),
        ("LINEBELOW",   (0, 0), (-1, 0),  1,   _BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_WHITE, _LGRAY]),
        ("FONTNAME",    (1, 1), (1, -1), "Helvetica-Bold"),
        ("FONTNAME",    (3, 1), (3, -1), "Helvetica-Bold"),
        ("ALIGN",       (1, 1), (1, -1), "RIGHT"),
        ("ALIGN",       (3, 1), (3, -1), "RIGHT"),
    ]
    t.setStyle(TableStyle(style))
    return t


def _bloque_charts(*, equity, ts_salida, pnl, W) -> list:
    """Genera la imagen matplotlib con equity + drawdown y la embebe en el PDF."""
    img_buf = _render_charts(equity=equity, ts_salida=ts_salida, pnl=pnl, W=W)
    img = RLImage(img_buf, width=W - 2.4*cm, height=9*cm)
    return [Paragraph("EQUITY CURVE &amp; DRAWDOWN", _ps(7, _MUTED, bold=True)),
            Spacer(1, 2*mm),
            img]


def _render_charts(*, equity, ts_salida, pnl, W) -> io.BytesIO:
    """Genera equity + drawdown como PNG en memoria."""
    from datetime import datetime, timezone as tz

    n_eq  = len(equity)
    n_ts  = len(ts_salida) if ts_salida is not None and len(ts_salida) > 0 else 0

    # Fechas para el eje X
    if n_ts > 0:
        ts = np.asarray(ts_salida, dtype=np.int64)
        dates = [datetime.fromtimestamp(t / 1_000_000, tz=tz.utc) for t in ts.tolist()]
        if n_eq == n_ts + 1:
            dates = [dates[0]] + dates
        elif n_eq != n_ts:
            dates = list(range(n_eq))
    else:
        dates = list(range(n_eq))

    picos  = np.maximum.accumulate(equity)
    dd_pct = np.where(picos > 0, (equity - picos) / picos * 100.0, 0.0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 4.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
        facecolor="#0F172A",
    )
    fig.subplots_adjust(hspace=0.05)

    # Equity
    ax1.set_facecolor("#0F172A")
    ax1.plot(dates, equity, color="#3B82F6", linewidth=1.4, zorder=3)
    ax1.fill_between(dates, equity, equity[0], alpha=0.12, color="#3B82F6", zorder=2)
    ax1.axhline(y=equity[0], color="#334155", linewidth=0.8, linestyle="--")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.tick_params(colors="#64748B", labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1E293B")
    ax1.grid(True, color="#1E293B", linewidth=0.5, alpha=0.7)
    ax1.set_xticklabels([])

    # Drawdown
    ax2.set_facecolor("#0F172A")
    ax2.fill_between(dates, dd_pct, 0, alpha=0.5, color="#EF4444", zorder=2)
    ax2.plot(dates, dd_pct, color="#EF4444", linewidth=1.0, zorder=3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.tick_params(colors="#64748B", labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1E293B")
    ax2.grid(True, color="#1E293B", linewidth=0.5, alpha=0.7)
    ax2.invert_yaxis()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0F172A", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def _bloque_monthly(monthly: dict[int, dict[int, float]], W) -> list:
    """Tabla de retornos mensuales con fondo rojo/verde proporcional."""
    if not monthly:
        return [Paragraph("Sin datos mensuales.", _ps(8, _MUTED))]

    years  = sorted(monthly.keys())
    meses  = ["Ene","Feb","Mar","Abr","May","Jun",
               "Jul","Ago","Sep","Oct","Nov","Dic"]
    header = ["Año"] + meses + ["YTD"]

    all_vals = [v for yr in monthly.values() for v in yr.values() if v != 0.0]
    max_abs  = max((abs(v) for v in all_vals), default=1.0)

    rows   = [header]
    styles = [
        ("BACKGROUND",    (0, 0), (-1, 0),  _NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  _TEXT),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BOX",           (0, 0), (-1, -1), 0.5, _DGRAY),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, _DGRAY),
        ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
    ]

    for ri, yr in enumerate(years, start=1):
        ytd  = sum(monthly[yr].values())
        row  = [str(yr)]
        for m in range(1, 13):
            val = monthly[yr].get(m, 0.0)
            row.append(_usd_short(val) if val != 0.0 else "—")
        row.append(_usd_short(ytd))
        rows.append(row)

        # Colorear celdas mes a mes
        for ci, m in enumerate(range(1, 13), start=1):
            val = monthly[yr].get(m, 0.0)
            if val == 0.0:
                continue
            intensity = min(1.0, abs(val) / max_abs)
            if val > 0:
                r = int(220 - intensity * 120)
                g = int(255 - intensity * 80)
                b = int(220 - intensity * 120)
                bg = colors.Color(r/255, g/255, b/255, alpha=1)
            else:
                r = int(255 - intensity * 60)
                g = int(220 - intensity * 180)
                b = int(220 - intensity * 180)
                bg = colors.Color(r/255, g/255, b/255, alpha=1)
            styles.append(("BACKGROUND", (ci, ri), (ci, ri), bg))

        # YTD
        ytd_c = _GREEN_L if ytd > 0 else (_RED_L if ytd < 0 else _WHITE)
        styles.append(("BACKGROUND", (13, ri), (13, ri), ytd_c))
        if ytd != 0.0:
            styles.append(("TEXTCOLOR", (13, ri), (13, ri),
                           _GREEN if ytd > 0 else _RED))

    cw_total = W - 2.4*cm
    col_widths = [1.3*cm] + [(cw_total - 1.3*cm - 1.5*cm) / 12] * 12 + [1.5*cm]

    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle(styles))

    return [
        Paragraph("RENTABILIDAD MENSUAL ($)", _ps(7, _MUTED, bold=True)),
        Spacer(1, 2*mm),
        t,
    ]


def _footer() -> Paragraph:
    return Paragraph(
        "SQN: Van Tharp (2006)  ·  Z-Score: Wald–Wolfowitz (1940)  ·  "
        "PSR: Bailey &amp; Lopez de Prado (2012)  ·  "
        "R-Expectancy: Van Tharp (1998)  ·  AHPR/GHPR: Vince (1992)",
        _ps(6, _MUTED),
    )


# ---------------------------------------------------------------------------
# Helpers de estilo y formato
# ---------------------------------------------------------------------------

def _ps(size: int, color, *, bold: bool = False) -> ParagraphStyle:
    return ParagraphStyle(
        name=f"s{size}{'b' if bold else ''}",
        fontSize=size,
        leading=size * 1.3,
        textColor=color,
        fontName="Helvetica-Bold" if bold else "Helvetica",
        spaceAfter=0,
        spaceBefore=0,
    )


def _usd(v: float) -> str:
    if not math.isfinite(v):
        return "N/A"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _usd_short(v: float) -> str:
    if not math.isfinite(v) or v == 0.0:
        return "—"
    return f"{'−' if v < 0 else ''}{abs(v):,.0f}"


# ── Colores semaforo ───────────────────────────────────────────────────────

def _sign_c(v: float):
    return _GREEN if v > 0 else (_RED if v < 0 else _SLATE)


def _sharpe_c(v: float):
    if v >= 2.0: return _GREEN
    if v >= 1.0: return _AMBER
    return _RED if v <= 0 else _SLATE


def _pf_c(v: float):
    if v >= 1.5: return _GREEN
    if v >= 1.0: return _AMBER
    return _RED


def _wr_c(v: float):
    if v >= 0.55: return _GREEN
    if v >= 0.45: return _AMBER
    return _RED


def _sqn_c(v: float):
    if v >= 2.5: return _GREEN
    if v >= 1.6: return _AMBER
    return _RED if v < 0 else _SLATE


def _psr_c(v: float):
    if v >= 0.90: return _GREEN
    if v >= 0.75: return _AMBER
    return _RED if v < 0.5 else _SLATE


# ---------------------------------------------------------------------------
# Utilidades de fichero
# ---------------------------------------------------------------------------

def _unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    i = 2
    while True:
        candidate = p.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1

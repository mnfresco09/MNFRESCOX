"""Excel del backtest: una hoja por trial con tabla de trades, grafico de
balance y un panel de analisis/robustez (veredicto + valoraciones).

Diseno minimalista. La presentacion de metricas (etiquetas, formato y juicio
BUENO/REGULAR/MALO) viene de `REPORTES.metricas_presentacion`, la misma fuente
que usa el report HTML, de modo que ambos coinciden siempre.
"""

from __future__ import annotations

import re
import unicodedata
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import xlsxwriter

from REPORTES.analitica import analitica_avanzada
from REPORTES.metricas_presentacion import construir_analitica
from REPORTES.persistencia import trades_dataframe

MAX_DETALLES_EXCEL = 5

# ── Paleta minimalista profesional (claro) ──────────────────────────────────
INK = "#0F172A"; SUB = "#64748B"; LINE = "#E2E8F0"; BAND = "#F1F5F9"
ACC = "#2563EB"; POS = "#15803D"; NEG = "#DC2626"; WARN = "#B45309"; HEAD = "#0F172A"
INFO = "#94A3B8"
TINTE = {"good": "#DCFCE7", "ok": "#FEF3C7", "bad": "#FEE2E2", "info": "#FFFFFF"}
NIVEL_COLOR = {"good": POS, "ok": WARN, "bad": NEG, "info": INFO}

# Tabla de trades: columna -> (cabecera, formato_numero)
_TRADE_COLS = [
    ("n",             "#",            "0"),
    ("direccion_txt", "DIR",          None),
    ("ts_entrada",    "ENTRADA",      "dd/mm hh:mm"),
    ("ts_salida",     "SALIDA",       "dd/mm hh:mm"),
    ("precio_entrada", "P. ENTRADA",  "#,##0.00"),
    ("precio_salida", "P. SALIDA",    "#,##0.00"),
    ("apalancamiento", "LEV",         '0"x"'),
    ("comision_total", "COMISIÓN",    "$#,##0.00"),
    ("pnl",           "PNL NETO",     "$#,##0.00"),
    ("roi",           "ROI",          "0.0%"),
    ("saldo_post",    "BALANCE",      "$#,##0.00"),
    ("duracion_velas", "DUR",         "0"),
    ("motivo_salida", "MOTIVO",       None),
]
_DATE_COLS = {"ts_entrada", "ts_salida"}


# ═══════════════════════════════════════════════════════════════════════════
# API pública
# ═══════════════════════════════════════════════════════════════════════════

def generar_excel(
    run_dir: Path,
    trials: list,
    mejor,
    *,
    fecha_inicio: date | None = None,
    fecha_fin: date | None = None,
) -> Path | None:
    """Genera un Excel (una hoja) por cada uno de los top-N trials con replay.
    Devuelve la ruta del Excel del mejor trial, o None si no hay ninguno."""
    excel_dir = _base_resultados(run_dir) / "EXCEL"
    excel_dir.mkdir(parents=True, exist_ok=True)

    top = [
        t for t in sorted(trials, key=lambda t: t.score, reverse=True)
        if t.replay is not None
    ][:MAX_DETALLES_EXCEL]
    if not top:
        return None

    primero: Path | None = None
    for trial in top:
        path = _generar_excel_trial(excel_dir, trial, fecha_inicio, fecha_fin)
        if primero is None:
            primero = path
    return primero


def verificar_excel(path: Path, filas_trades: int) -> None:
    if not path.exists():
        raise ValueError(f"[EXCEL] No se genero {path}.")
    with zipfile.ZipFile(path) as zf:
        presentes = set(zf.namelist())
        sheet = "xl/worksheets/sheet1.xml"
        if sheet not in presentes:
            raise ValueError(f"[EXCEL] Falta hoja interna {sheet}.")
        contenido = zf.read(sheet).decode("utf-8")
        filas_xml = contenido.count("<row ")
        if filas_xml < filas_trades + 1:
            raise ValueError(
                f"[EXCEL] {sheet} no conserva las filas de trades: {filas_xml} < {filas_trades + 1}."
            )
        if filas_trades > 0:
            charts = [p for p in presentes if p.startswith("xl/charts/chart") and p.endswith(".xml")]
            if not charts:
                raise ValueError("[EXCEL] No se genero el grafico de balance.")


# ═══════════════════════════════════════════════════════════════════════════
# Generación
# ═══════════════════════════════════════════════════════════════════════════

def _generar_excel_trial(excel_dir: Path, trial, fecha_inicio, fecha_fin) -> Path:
    path = _unique_path(excel_dir / _nombre_detalle(trial))
    trades = trades_dataframe(trial.replay)
    fi, ff = _resolver_fechas(trial, fecha_inicio, fecha_fin)
    avanzada = analitica_avanzada(
        metricas=trial.metricas,
        trades=trial.replay.trades,
        equity_curve=trial.replay.equity_curve,
        fecha_inicio=fi,
        fecha_fin=ff,
    )
    panel = construir_analitica(trial.metricas, avanzada)

    workbook = xlsxwriter.Workbook(
        str(path),
        {"nan_inf_to_errors": True, "default_date_format": "dd/mm/yy hh:mm"},
    )
    try:
        fmt = _crear_formatos(workbook)
        _write_hoja_trades(workbook, trades, trial, panel, fmt)
    finally:
        workbook.close()

    verificar_excel(path, trades.height)
    return path


def _write_hoja_trades(workbook, trades: pl.DataFrame, trial, panel: dict, fmt: dict) -> None:
    ws = workbook.add_worksheet("TRADES")
    ws.hide_gridlines(2)
    ws.set_tab_color(ACC)

    ws.set_row(0, 24)
    ws.write(0, 0, f"TRADES · {_nombre_visible(trial.estrategia_nombre)}", fmt["title"])
    ws.write(
        1, 0,
        f"{trial.activo} · {trial.timeframe} · {trial.salida.tipo}    ·    "
        f"Trial #{int(trial.numero)}    ·    "
        + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        fmt["sub"],
    )

    hr = 3
    cols = [c for c in _TRADE_COLS if c[0] == "n" or c[0] in trades.columns]
    for c, (name, header, _) in enumerate(cols):
        ws.write(hr, c, header, fmt["header"])
    ws.set_row(hr, 22)

    filas = trades.rows(named=True)
    for i, row in enumerate(filas):
        rr = hr + 1 + i
        band = i % 2 == 1
        ws.set_row(rr, 17)
        for c, (name, _h, numfmt) in enumerate(cols):
            valor = (i + 1) if name == "n" else row.get(name)
            celda = _fmt_celda(fmt, name, numfmt, band)
            _write_cell(ws, rr, c, valor, celda, name)

    nT = trades.height
    last = hr + nT
    col_pnl = next((c for c, (nm, *_r) in enumerate(cols) if nm == "pnl"), None)
    if col_pnl is not None and nT > 0:
        ws.conditional_format(hr + 1, col_pnl, last, col_pnl, {"type": "cell", "criteria": ">", "value": 0, "format": fmt["pos"]})
        ws.conditional_format(hr + 1, col_pnl, last, col_pnl, {"type": "cell", "criteria": "<", "value": 0, "format": fmt["neg"]})
    ws.freeze_panes(hr + 1, 0)
    ws.autofilter(hr, 0, max(last, hr + 1), len(cols) - 1)
    for c, w in enumerate([5, 7, 13, 13, 11, 11, 6, 11, 12, 8, 12, 7, 10][:len(cols)]):
        ws.set_column(c, c, w)

    _write_panel(ws, panel, trial, fmt)

    col_balance = next((c for c, (nm, *_r) in enumerate(cols) if nm == "saldo_post"), None)
    if nT > 0 and col_balance is not None:
        _insert_balance_chart(workbook, ws, fila0=hr + 1, fila_fin=last, col_balance=col_balance, start_row=last + 3)


def _write_panel(ws, panel: dict, trial, fmt: dict) -> None:
    O = 14
    ws.set_column(13, 13, 2)
    ws.set_column(O, O, 21); ws.set_column(O + 1, O + 1, 13); ws.set_column(O + 2, O + 2, 11)

    ws.merge_range(0, O, 0, O + 2, f"ANÁLISIS · TRIAL #{int(trial.numero)}", fmt["title2"])

    v = panel["veredicto"]
    ws.merge_range(
        1, O, 1, O + 2,
        f"VEREDICTO · {v['badge']} · {v['favorables']}/{v['total']} favorables",
        fmt["verdict"][v["nivel"]],
    )
    ws.set_row(1, 22)

    fila = 3
    for seccion in panel["secciones"]:
        ws.merge_range(fila, O, fila, O + 2, seccion["titulo"], fmt["sec"])
        ws.set_row(fila, 20)
        for i, f in enumerate(seccion["filas"]):
            rr = fila + 1 + i
            ws.set_row(rr, 18)
            band = i % 2 == 1
            ws.write(rr, O, f["label"], fmt["panel_lbl"][1 if band else 0])
            color = NIVEL_COLOR.get(f["nivel"], INK)
            valfmt = _panel_val(fmt, color, band)
            ws.write(rr, O + 1, f["valor"], valfmt)
            ws.write(rr, O + 2, _nivel_label(f["nivel"]), fmt["chip"][f["nivel"]])
        fila = fila + 1 + len(seccion["filas"]) + 1


def _insert_balance_chart(workbook, ws, *, fila0: int, fila_fin: int, col_balance: int, start_row: int) -> None:
    chart = workbook.add_chart({"type": "area"})
    chart.add_series({
        "name": "Balance",
        "categories": ["TRADES", fila0, 0, fila_fin, 0],
        "values": ["TRADES", fila0, col_balance, fila_fin, col_balance],
        "line": {"color": ACC, "width": 1.75},
        "fill": {"color": ACC, "transparency": 85},
    })
    chart.set_title({"name": "Evolución del balance", "name_font": {"name": "Aptos", "size": 11, "color": INK}})
    chart.set_legend({"none": True})
    chart.set_size({"width": 820, "height": 300})
    chart.set_y_axis({"num_format": "$#,##0", "num_font": {"color": SUB}, "major_gridlines": {"visible": True, "line": {"color": LINE}}})
    chart.set_x_axis({"name": "Trade #", "num_font": {"color": SUB}})
    chart.set_chartarea({"border": {"color": LINE}})
    chart.set_plotarea({"border": {"none": True}})
    ws.insert_chart(start_row, 0, chart, {"x_offset": 2, "y_offset": 8})


# ═══════════════════════════════════════════════════════════════════════════
# Formatos
# ═══════════════════════════════════════════════════════════════════════════

def _crear_formatos(workbook) -> dict:
    def base(**k): return workbook.add_format(k)
    fmt = {
        "title": base(font_name="Aptos Display", font_size=14, bold=True, font_color=INK),
        "title2": base(font_name="Aptos Display", font_size=12, bold=True, font_color=INK),
        "sub": base(font_name="Aptos", font_size=9, font_color=SUB),
        "header": base(font_name="Aptos", font_size=9, bold=True, font_color="#FFFFFF", bg_color=HEAD, align="center", valign="vcenter"),
        "sec": base(font_name="Aptos", font_size=9, bold=True, font_color="#FFFFFF", bg_color=ACC, align="center", valign="vcenter"),
        "pos": base(font_name="Aptos", font_size=10, bold=True, font_color=POS),
        "neg": base(font_name="Aptos", font_size=10, bold=True, font_color=NEG),
    }
    fmt["panel_lbl"] = (
        base(font_name="Aptos", font_size=10, font_color=SUB, align="left", valign="vcenter", bottom=1, bottom_color=LINE),
        base(font_name="Aptos", font_size=10, font_color=SUB, align="left", valign="vcenter", bottom=1, bottom_color=LINE, bg_color=BAND),
    )
    fmt["verdict"] = {
        lvl: base(font_name="Aptos", font_size=10, bold=True, font_color=NIVEL_COLOR[lvl], bg_color=TINTE[lvl], align="center", valign="vcenter", border=1, border_color=LINE)
        for lvl in ("good", "ok", "bad")
    }
    fmt["chip"] = {
        lvl: base(font_name="Aptos", font_size=8, bold=True, font_color=NIVEL_COLOR[lvl], bg_color=TINTE[lvl], align="center", valign="vcenter", border=1, border_color=LINE)
        for lvl in ("good", "ok", "bad", "info")
    }
    # Pares (impar, par) para celdas de la tabla, por tipo de formato numerico.
    fmt["_wb"] = workbook
    return fmt


def _fmt_celda(fmt: dict, name: str, numfmt: str | None, band: bool):
    wb = fmt["_wb"]
    d = dict(font_name="Aptos", font_size=10, align="center", valign="vcenter", bottom=1, bottom_color=LINE)
    if band:
        d["bg_color"] = BAND
    if numfmt:
        d["num_format"] = numfmt
    if name in ("pnl", "roi"):
        d["bold"] = True  # color por valor lo pone el formato condicional / inline
    return wb.add_format(d)


def _panel_val(fmt: dict, color: str, band: bool):
    wb = fmt["_wb"]
    d = dict(font_name="Aptos", font_size=10, bold=True, font_color=color, align="right", valign="vcenter", bottom=1, bottom_color=LINE)
    if band:
        d["bg_color"] = BAND
    return wb.add_format(d)


def _write_cell(ws, row: int, col: int, value: Any, formato, name: str) -> None:
    if value is None:
        ws.write_blank(row, col, None, formato)
        return
    if name in _DATE_COLS:
        dt = _datetime_from_us(value)
        if dt is None:
            ws.write(row, col, value, formato)
        else:
            ws.write_datetime(row, col, dt, formato)
        return
    if name == "direccion_txt":
        ws.write_string(row, col, str(value), formato)
        return
    if isinstance(value, bool):
        ws.write_string(row, col, "SÍ" if value else "NO", formato)
    elif isinstance(value, (int, float)):
        ws.write_number(row, col, value, formato)
    else:
        ws.write_string(row, col, str(value), formato)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _nivel_label(nivel: str) -> str:
    return {"good": "BUENO", "ok": "REGULAR", "bad": "MALO", "info": "INFO"}.get(nivel, "")


def _resolver_fechas(trial, fecha_inicio, fecha_fin) -> tuple[date, date]:
    if fecha_inicio is not None and fecha_fin is not None:
        return fecha_inicio, fecha_fin
    cols = trial.replay.trades
    try:
        ini = datetime.fromtimestamp(int(cols["ts_entrada"][0]) / 1_000_000, tz=timezone.utc).date()
        fin = datetime.fromtimestamp(int(cols["ts_salida"][-1]) / 1_000_000, tz=timezone.utc).date()
        return ini, fin
    except Exception:
        hoy = datetime.now(timezone.utc).date()
        return hoy, hoy


def _datetime_from_us(value: Any) -> datetime | None:
    try:
        if value is None:
            return None
        return datetime.fromtimestamp(int(value) / 1_000_000, tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _base_resultados(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    if run_dir.parent.name.upper() == "DATOS":
        return run_dir.parent.parent
    return run_dir


def _nombre_detalle(trial) -> str:
    return f"TRIAL {int(trial.numero)} - {_score_nombre(trial.score)}.xlsx"


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
    raise RuntimeError(f"[EXCEL] No se pudo crear nombre unico para {path}.")


def _slug_excel(value: Any) -> str:
    normalizado = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalizado.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").upper()
    return slug or "SIN_NOMBRE"


def _nombre_visible(value: Any) -> str:
    return _slug_excel(value).replace("_", " ")

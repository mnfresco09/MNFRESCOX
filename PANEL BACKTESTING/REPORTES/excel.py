from __future__ import annotations

import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import xlsxwriter

from REPORTES.persistencia import equity_dataframe, resumen_trials_dataframe, trades_dataframe


# ── Orden de columnas ─────────────────────────────────────────────────────────

ID_ORDER = ["trial", "activo", "timeframe", "estrategia", "salida", "score"]
METRIC_ORDER = [
    "total_trades",
    "trades_long",
    "trades_short",
    "trades_ganadores",
    "trades_perdedores",
    "win_rate",
    "roi_total",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "pnl_bruto_total",
    "pnl_total",
    "pnl_promedio",
    "saldo_inicial",
    "saldo_final",
    "duracion_media_seg",
    "parado_por_saldo",
]

# ── Nombres de cabecera ───────────────────────────────────────────────────────

SUMMARY_NAMES = {
    "trial":             "TRIAL",
    "activo":            "ACTIVO",
    "timeframe":         "TF",
    "estrategia":        "ESTRATEGIA",
    "salida":            "EXIT",
    "score":             "SCORE",
    "total_trades":      "TRADES",
    "trades_long":       "LONG",
    "trades_short":      "SHORT",
    "trades_ganadores":  "WIN",
    "trades_perdedores": "LOSS",
    "win_rate":          "WIN RATE",
    "roi_total":         "ROI",
    "max_drawdown":      "MAX DD",
    "profit_factor":     "PF",
    "sharpe_ratio":      "SHARPE",
    "pnl_bruto_total":   "PNL BRUTO",
    "pnl_total":         "PNL NETO",
    "pnl_promedio":      "PNL PROM",
    "saldo_inicial":     "SALDO INI",
    "saldo_final":       "SALDO FIN",
    "duracion_media_seg": "DUR MEDIA",
    "parado_por_saldo":  "STOP SALDO",
    "param_exit_type":   "EXIT TYPE",
    "param_exit_sl_pct": "SL %",
    "param_exit_tp_pct": "TP %",
    "param_exit_velas":  "VELAS",
}

_TRADE_COLS_OCULTAS  = {"idx_senal", "idx_entrada", "idx_salida", "direccion", "duracion_velas"}
_EQUITY_COLS_OCULTAS = {"idx_salida"}

TRADE_NAMES = {
    "ts_senal":        "SEÑAL",
    "ts_entrada":      "ENTRADA",
    "ts_salida":       "SALIDA",
    "direccion_txt":   "DIR",
    "precio_entrada":  "P. ENTRADA",
    "precio_salida":   "P. SALIDA",
    "saldo_apertura":  "SALDO APERT.",
    "tamano_posicion": "TAMAÑO POS.",
    "comision_total":  "COMISIÓN",
    "pnl_bruto":       "PNL BRUTO",
    "pnl":             "PNL NETO",
    "roi":             "ROI",
    "saldo_post":      "BALANCE",
    "motivo_salida":   "MOTIVO",
    "duracion_seg":    "DURACIÓN",
}

EQUITY_NAMES = {
    "punto":     "Nº",
    "trade_num": "TRADE",
    "ts_salida": "FECHA CIERRE",
    "saldo":     "BALANCE",
}

# ── Clasificación de columnas por tipo de formato ────────────────────────────

MONEY_COLS = {
    "saldo_inicial", "saldo_final",
    "pnl_bruto_total", "pnl_total", "pnl_promedio",
    "saldo_apertura", "tamano_posicion", "comision_total",
    "pnl_bruto", "pnl", "saldo_post", "saldo",
}
PCT_COLS      = {"win_rate", "roi_total", "max_drawdown", "roi"}
PCT_POINT_COLS = {"param_exit_sl_pct", "param_exit_tp_pct"}
INT_COLS      = {
    "trial", "total_trades", "trades_long", "trades_short",
    "trades_ganadores", "trades_perdedores",
    "punto", "trade_num", "param_exit_velas",
}
PRICE_COLS    = {"precio_entrada", "precio_salida"}
DATE_COLS     = {"ts_senal", "ts_entrada", "ts_salida"}
DUR_SEG_COLS  = {"duracion_seg", "duracion_media_seg"}

MAX_DETALLES_EXCEL = 5

# ── Paleta profesional ────────────────────────────────────────────────────────
#   Fondo oscuro tipo terminal de trading
_BG_BASE      = "#0D1117"   # fondo principal
_BG_HEADER    = "#161B22"   # cabecera de hoja
_BG_SECTION   = "#1C2128"   # sección agrupadora
_BG_ROW_EVEN  = "#161B22"   # fila par
_BG_ROW_ODD   = "#0D1117"   # fila impar
_FG_HEADER    = "#8B949E"   # texto cabecera
_FG_SECTION   = "#6E7681"   # texto sección
_FG_MAIN      = "#E6EDF3"   # texto principal
_FG_DIM       = "#8B949E"   # texto secundario
_BORDER       = "#30363D"   # borde sutil
_GREEN        = "#3FB950"   # positivo
_RED          = "#F85149"   # negativo
_BLUE_ACCENT  = "#388BFD"   # acento azul (score, títulos)
_GOLD         = "#D29922"   # alerta / stop saldo
_CHART_LINE   = "#388BFD"


# ═══════════════════════════════════════════════════════════════════════════════
# API pública
# ═══════════════════════════════════════════════════════════════════════════════

def generar_excel(run_dir: Path, trials: list, mejor) -> Path:
    excel_dir = _base_resultados(run_dir) / "EXCEL"
    excel_dir.mkdir(parents=True, exist_ok=True)

    path = _unique_path(excel_dir / _nombre_resumen(mejor))
    resumen = resumen_trials_dataframe(trials)

    workbook = xlsxwriter.Workbook(
        str(path),
        {"nan_inf_to_errors": True, "default_date_format": "dd/mm/yy hh:mm"},
    )
    try:
        formatos = _crear_formatos(workbook)
        _write_resumen(workbook, resumen, formatos)
    finally:
        workbook.close()

    verificar_resumen_excel(path, resumen.height)
    for trial in sorted(trials, key=lambda t: t.score, reverse=True)[:MAX_DETALLES_EXCEL]:
        _generar_excel_detalle(excel_dir, trial)

    return path


def verificar_resumen_excel(path: Path, filas_resumen: int) -> None:
    if not path.exists():
        raise ValueError(f"[EXCEL] No se genero {path}.")
    with zipfile.ZipFile(path) as zf:
        presentes = set(zf.namelist())
        sheet = "xl/worksheets/sheet1.xml"
        if sheet not in presentes:
            raise ValueError(f"[EXCEL] Falta hoja interna {sheet}.")
        contenido = zf.read(sheet).decode("utf-8")
        filas_xml = contenido.count("<row ")
        esperadas = filas_resumen + 2
        if filas_xml != esperadas:
            raise ValueError(
                f"[EXCEL] {sheet} no conserva filas: {filas_xml} != {esperadas}."
            )


def verificar_detalle_excel(path: Path, filas_trades: int, filas_equity: int) -> None:
    if not path.exists():
        raise ValueError(f"[EXCEL] No se genero {path}.")
    esperadas = {
        "xl/worksheets/sheet1.xml": filas_trades + 1,
        "xl/worksheets/sheet2.xml": filas_equity + 1,
    }
    with zipfile.ZipFile(path) as zf:
        presentes = set(zf.namelist())
        for sheet, filas in esperadas.items():
            if sheet not in presentes:
                raise ValueError(f"[EXCEL] Falta hoja interna {sheet}.")
            contenido = zf.read(sheet).decode("utf-8")
            filas_xml = contenido.count("<row ")
            if filas_xml != filas:
                raise ValueError(
                    f"[EXCEL] {sheet} no conserva filas: {filas_xml} != {filas}."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Generación de hojas
# ═══════════════════════════════════════════════════════════════════════════════

def _generar_excel_detalle(excel_dir: Path, trial) -> Path:
    path = _unique_path(excel_dir / _nombre_detalle(trial))
    trades = trades_dataframe(trial.resultado)
    equity = equity_dataframe(trial.resultado)

    workbook = xlsxwriter.Workbook(
        str(path),
        {"nan_inf_to_errors": True, "default_date_format": "dd/mm/yy hh:mm"},
    )
    try:
        formatos = _crear_formatos(workbook)
        _write_trades(workbook, trades, equity.height, formatos)
        _write_equity(workbook, equity, formatos)
    finally:
        workbook.close()

    verificar_detalle_excel(path, trades.height, equity.height)
    return path


def _write_resumen(workbook, df: pl.DataFrame, formatos: dict) -> None:
    ws = workbook.add_worksheet("RESUMEN")
    ws.hide_gridlines(2)
    ws.set_tab_color(_BLUE_ACCENT)

    columnas = _columnas_resumen(df)
    secciones = [_seccion_resumen(col) for col in columnas]

    _write_section_headers(ws, secciones, formatos)
    _write_headers(ws, 1, columnas, SUMMARY_NAMES, formatos["header"])

    for row_idx, row in enumerate(df.select(columnas).rows(named=True), start=2):
        alt = row_idx % 2 == 0
        ws.set_row(row_idx, 20)
        for col_idx, name in enumerate(columnas):
            fmt = _format_for(name, formatos, alt)
            _write_cell(ws, row_idx, col_idx, row[name], fmt, name)

    ws.freeze_panes(2, 0)
    ws.autofilter(1, 0, max(df.height + 1, 1), max(len(columnas) - 1, 0))
    ws.set_row(0, 18)
    ws.set_row(1, 28)
    _set_widths(ws, columnas, df, SUMMARY_NAMES)
    _add_summary_conditionals(ws, columnas, df.height, formatos)


def _write_trades(workbook, df: pl.DataFrame, filas_equity: int, formatos: dict) -> None:
    ws = workbook.add_worksheet("TRADES")
    ws.hide_gridlines(2)
    ws.set_tab_color(_GREEN)

    columnas = [c for c in df.columns if c not in _TRADE_COLS_OCULTAS]

    _write_headers(ws, 0, columnas, TRADE_NAMES, formatos["header"])
    for row_idx, row in enumerate(df.rows(named=True), start=1):
        alt = row_idx % 2 == 0
        ws.set_row(row_idx, 20)
        for col_idx, name in enumerate(columnas):
            fmt = _format_for(name, formatos, alt)
            _write_cell(ws, row_idx, col_idx, row[name], fmt, name)

    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, max(df.height, 1), max(len(columnas) - 1, 0))
    ws.set_row(0, 28)
    _set_widths(ws, columnas, df, TRADE_NAMES)
    _add_trade_conditionals(ws, columnas, df.height, formatos)

    if df.height > 0:
        _insert_balance_chart(workbook, ws, start_row=df.height + 3, filas_equity=filas_equity)


def _write_equity(workbook, df: pl.DataFrame, formatos: dict) -> None:
    ws = workbook.add_worksheet("EQUITY")
    ws.hide_gridlines(2)
    ws.set_tab_color(_CHART_LINE)

    columnas = [c for c in df.columns if c not in _EQUITY_COLS_OCULTAS]

    _write_headers(ws, 0, columnas, EQUITY_NAMES, formatos["header"])
    for row_idx, row in enumerate(df.rows(named=True), start=1):
        alt = row_idx % 2 == 0
        ws.set_row(row_idx, 20)
        for col_idx, name in enumerate(columnas):
            fmt = _format_for(name, formatos, alt)
            _write_cell(ws, row_idx, col_idx, row[name], fmt, name)

    ws.freeze_panes(1, 0)
    ws.autofilter(0, 0, max(df.height, 1), max(len(columnas) - 1, 0))
    ws.set_row(0, 28)
    _set_widths(ws, columnas, df, EQUITY_NAMES)


def _insert_balance_chart(workbook, worksheet, *, start_row: int, filas_equity: int) -> None:
    last_row = max(filas_equity + 1, 2)
    chart = workbook.add_chart({"type": "area"})
    chart.add_series(
        {
            "name":       "Balance",
            "categories": f"=EQUITY!$B$2:$B${last_row}",
            "values":     f"=EQUITY!$D$2:$D${last_row}",
            "line":  {"color": _CHART_LINE, "width": 2.0},
            "fill":  {"color": _CHART_LINE, "transparency": 80},
        }
    )
    chart.set_title({"name": "Evolución del balance", "name_font": {"color": _FG_MAIN, "size": 11}})
    chart.set_legend({"none": True})
    chart.set_y_axis(
        {
            "num_format": "$#,##0",
            "major_gridlines": {"visible": True, "line": {"color": _BORDER, "width": 0.5}},
            "line": {"none": True},
            "num_font": {"color": _FG_DIM},
        }
    )
    chart.set_x_axis(
        {
            "name": "Trade",
            "major_gridlines": {"visible": False},
            "num_font": {"color": _FG_DIM},
            "name_font": {"color": _FG_DIM},
        }
    )
    chart.set_plotarea({"border": {"none": True}, "fill": {"color": _BG_BASE}})
    chart.set_chartarea({"border": {"color": _BORDER}, "fill": {"color": _BG_HEADER}})
    chart.set_size({"width": 1100, "height": 400})
    worksheet.insert_chart(start_row, 0, chart, {"x_offset": 2, "y_offset": 5})


# ═══════════════════════════════════════════════════════════════════════════════
# Formatos
# ═══════════════════════════════════════════════════════════════════════════════

def _crear_formatos(workbook) -> dict:
    def _base(bg: str, fg: str = _FG_MAIN, bold: bool = False) -> dict:
        return {
            "font_name":  "Calibri",
            "font_size":  10,
            "font_color": fg,
            "bold":       bold,
            "bg_color":   bg,
            "align":      "center",
            "valign":     "vcenter",
            "bottom":     1,
            "bottom_color": _BORDER,
        }

    def fmt(bg: str, num_format: str | None = None, fg: str = _FG_MAIN, bold: bool = False, **extra):
        params = _base(bg, fg, bold)
        if num_format is not None:
            params["num_format"] = num_format
        params.update(extra)
        return workbook.add_format(params)

    # Pares (impar, par)
    def pair(num_fmt=None, fg=_FG_MAIN, bold=False, **extra):
        return (
            fmt(_BG_ROW_ODD,  num_fmt, fg, bold, **extra),
            fmt(_BG_ROW_EVEN, num_fmt, fg, bold, **extra),
        )

    return {
        "section": workbook.add_format({
            "font_name":    "Calibri",
            "font_size":    8,
            "bold":         True,
            "font_color":   _FG_SECTION,
            "bg_color":     _BG_SECTION,
            "align":        "center",
            "valign":       "vcenter",
            "bottom":       1,
            "bottom_color": _BORDER,
            "text_wrap":    False,
        }),
        "header": workbook.add_format({
            "font_name":    "Calibri",
            "font_size":    9,
            "bold":         True,
            "font_color":   _FG_HEADER,
            "bg_color":     _BG_HEADER,
            "align":        "center",
            "valign":       "vcenter",
            "text_wrap":    True,
            "bottom":       2,
            "bottom_color": _BLUE_ACCENT,
        }),
        # Datos normales
        "text":      pair(),
        "int":       pair("0"),
        "num":       pair("0.000000"),
        "ratio":     pair("0.00"),
        "money":     pair('$#,##0.00'),
        "money3":    pair('$#,##0.000'),
        "price":     pair('#,##0.00'),
        "pct":       pair("0.00%"),
        "pct_point": pair('0.0"%"'),
        "date":      pair("dd/mm/yy hh:mm"),
        "dur":       pair(),         # duración D/H/M → se escribe como string
        # Condicionales de color (sin fondo, solo color texto)
        "positive": workbook.add_format({
            "font_color": _GREEN,
            "bold":       True,
            "font_name":  "Calibri",
            "font_size":  10,
        }),
        "negative": workbook.add_format({
            "font_color": _RED,
            "bold":       True,
            "font_name":  "Calibri",
            "font_size":  10,
        }),
        "warning": workbook.add_format({
            "font_color": _GOLD,
            "bold":       True,
            "font_name":  "Calibri",
            "font_size":  10,
        }),
    }


def _format_for(name: str, formatos: dict, alt: bool):
    idx = 1 if alt else 0
    if name in DATE_COLS:
        return formatos["date"][idx]
    if name in DUR_SEG_COLS:
        return formatos["dur"][idx]
    if name in MONEY_COLS:
        if name == "tamano_posicion":
            return formatos["money3"][idx]
        return formatos["money"][idx]
    if name in PRICE_COLS:
        return formatos["price"][idx]
    if name in PCT_COLS:
        return formatos["pct"][idx]
    if name in PCT_POINT_COLS:
        return formatos["pct_point"][idx]
    if name in INT_COLS:
        return formatos["int"][idx]
    if name in {"profit_factor", "sharpe_ratio", "duracion_media_velas"}:
        return formatos["ratio"][idx]
    if name == "score":
        return formatos["num"][idx]
    return formatos["text"][idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Escritura de celdas
# ═══════════════════════════════════════════════════════════════════════════════

def _write_section_headers(worksheet, secciones: list[str], formatos: dict) -> None:
    if not secciones:
        return
    start = 0
    current = secciones[0]
    for idx, section in enumerate(secciones + [None]):
        if section == current:
            continue
        if idx - 1 > start:
            worksheet.merge_range(0, start, 0, idx - 1, current, formatos["section"])
        else:
            worksheet.write(0, start, current, formatos["section"])
        start = idx
        current = section


def _write_headers(worksheet, row: int, columnas: list[str], nombres: dict, formato) -> None:
    for col_idx, name in enumerate(columnas):
        worksheet.write(row, col_idx, _display_name(name, nombres), formato)


def _write_cell(worksheet, row: int, col: int, value: Any, formato, name: str) -> None:
    if value is None:
        worksheet.write_blank(row, col, None, formato)
        return

    if name in DUR_SEG_COLS:
        worksheet.write_string(row, col, _seg_a_dhm(value), formato)
        return

    if name in DATE_COLS:
        dt = _datetime_from_us(value)
        if dt is None:
            worksheet.write(row, col, value, formato)
        else:
            worksheet.write_datetime(row, col, dt, formato)
        return

    if isinstance(value, bool):
        worksheet.write_string(row, col, "SÍ" if value else "NO", formato)
    elif isinstance(value, int):
        worksheet.write_number(row, col, value, formato)
    elif isinstance(value, float):
        worksheet.write_number(row, col, value, formato)
    else:
        worksheet.write_string(row, col, str(value), formato)


# ═══════════════════════════════════════════════════════════════════════════════
# Formato condicional
# ═══════════════════════════════════════════════════════════════════════════════

def _add_summary_conditionals(worksheet, columnas: list[str], filas: int, formatos: dict) -> None:
    if filas <= 0:
        return
    for name in ("roi_total", "pnl_bruto_total", "pnl_total", "score"):
        if name not in columnas:
            continue
        col = columnas.index(name)
        worksheet.conditional_format(2, col, filas + 1, col, {
            "type": "cell", "criteria": ">", "value": 0, "format": formatos["positive"],
        })
        worksheet.conditional_format(2, col, filas + 1, col, {
            "type": "cell", "criteria": "<", "value": 0, "format": formatos["negative"],
        })
    if "parado_por_saldo" in columnas:
        col = columnas.index("parado_por_saldo")
        worksheet.conditional_format(2, col, filas + 1, col, {
            "type": "text", "criteria": "containing", "value": "SÍ",
            "format": formatos["warning"],
        })


def _add_trade_conditionals(worksheet, columnas: list[str], filas: int, formatos: dict) -> None:
    if filas <= 0:
        return
    for name in ("pnl_bruto", "pnl"):
        if name not in columnas:
            continue
        col = columnas.index(name)
        worksheet.conditional_format(1, col, filas, col, {
            "type": "cell", "criteria": ">", "value": 0, "format": formatos["positive"],
        })
        worksheet.conditional_format(1, col, filas, col, {
            "type": "cell", "criteria": "<", "value": 0, "format": formatos["negative"],
        })


# ═══════════════════════════════════════════════════════════════════════════════
# Anchos de columna
# ═══════════════════════════════════════════════════════════════════════════════

def _set_widths(worksheet, columnas: list[str], df: pl.DataFrame, nombres: dict) -> None:
    rows = df.select([c for c in columnas if c in df.columns]).head(25).rows(named=True) if columnas else []
    for col_idx, name in enumerate(columnas):
        label_w = len(_display_name(name, nombres)) + 2
        data_w = 0
        for row in rows:
            value = row.get(name)
            if value is not None:
                if name in DUR_SEG_COLS:
                    data_w = max(data_w, len(_seg_a_dhm(value)) + 2)
                else:
                    data_w = max(data_w, min(len(str(value)) + 2, 26))
        width = min(max(label_w, data_w, 9), 26)
        worksheet.set_column(col_idx, col_idx, width)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _seg_a_dhm(segundos: Any) -> str:
    """Convierte segundos a formato 'DDd HHh MMm' o '—' si es cero/None."""
    try:
        total = int(float(segundos))
    except (TypeError, ValueError):
        return "—"
    if total <= 0:
        return "—"
    dias  = total // 86400
    horas = (total % 86400) // 3600
    mins  = (total % 3600) // 60
    partes = []
    if dias:
        partes.append(f"{dias}d")
    if horas or dias:
        partes.append(f"{horas:02d}h")
    partes.append(f"{mins:02d}m")
    return " ".join(partes)


def _columnas_resumen(df: pl.DataFrame) -> list[str]:
    existentes = set(df.columns)
    ids      = [col for col in ID_ORDER     if col in existentes]
    metricas = [col for col in METRIC_ORDER if col in existentes]
    params   = sorted(col for col in df.columns if col.startswith("param_"))
    usados   = set(ids + metricas + params)
    otros    = [col for col in df.columns if col not in usados]
    return ids + metricas + otros + params


def _seccion_resumen(col: str) -> str:
    if col in ID_ORDER:
        return "IDENTIFICACIÓN"
    if col.startswith("param_"):
        return "PARÁMETROS"
    return "MÉTRICAS"


def _display_name(name: str, nombres: dict) -> str:
    if name in nombres:
        return nombres[name]
    if name.startswith("param_"):
        return name.removeprefix("param_").replace("_", " ").upper()
    return name.replace("_", " ").upper()


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


def _nombre_resumen(mejor) -> str:
    estrategia = _nombre_visible(mejor.estrategia_nombre)
    return f"RESUMEN {estrategia}.xlsx"


def _nombre_detalle(trial) -> str:
    return f"TRIAL {int(trial.numero)} - {_score_nombre(trial.score)}.xlsx"


def _score_nombre(score: float) -> str:
    valor = f"{abs(float(score)):.6f}".rstrip("0").rstrip(".")
    return f"NEG {valor}" if float(score) < 0 else valor


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem   = path.stem
    suffix = path.suffix
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{idx:02d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"[EXCEL] No se pudo crear nombre unico para {path}.")


def _slug_excel(value: Any) -> str:
    normalizado = unicodedata.normalize("NFKD", str(value))
    ascii_text  = normalizado.encode("ascii", "ignore").decode("ascii")
    slug        = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").upper()
    return slug or "SIN_NOMBRE"


def _nombre_visible(value: Any) -> str:
    return _slug_excel(value).replace("_", " ")

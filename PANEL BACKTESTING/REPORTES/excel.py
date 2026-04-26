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


ID_ORDER = ["trial", "activo", "timeframe", "estrategia", "salida", "score"]
METRIC_ORDER = [
    "total_trades",
    "trades_ganadores",
    "trades_perdedores",
    "win_rate",
    "roi_total",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "pnl_total",
    "pnl_promedio",
    "saldo_inicial",
    "saldo_final",
    "duracion_media_velas",
    "parado_por_saldo",
]

SUMMARY_NAMES = {
    "trial": "TRIAL",
    "activo": "ACTIVO",
    "timeframe": "TF",
    "estrategia": "ESTRATEGIA",
    "salida": "EXIT",
    "score": "SCORE",
    "total_trades": "TRADES",
    "trades_ganadores": "WIN TRADES",
    "trades_perdedores": "LOSS TRADES",
    "win_rate": "WIN RATE",
    "roi_total": "ROI",
    "max_drawdown": "MAX DD",
    "profit_factor": "PROFIT FACTOR",
    "sharpe_ratio": "SHARPE",
    "pnl_total": "PNL NETO",
    "pnl_promedio": "PNL PROM",
    "saldo_inicial": "SALDO INICIAL",
    "saldo_final": "SALDO FINAL",
    "duracion_media_velas": "DUR MEDIA",
    "parado_por_saldo": "STOP SALDO",
    "param_exit_type": "EXIT TYPE",
    "param_exit_sl_pct": "SL",
    "param_exit_tp_pct": "TP",
    "param_exit_velas": "VELAS EXIT",
}

TRADE_NAMES = {
    "idx_senal": "IDX SENAL",
    "idx_entrada": "IDX ENTRADA",
    "idx_salida": "IDX SALIDA",
    "ts_senal": "TS SENAL",
    "ts_entrada": "TS ENTRADA",
    "ts_salida": "TS SALIDA",
    "direccion": "DIRECCION",
    "direccion_txt": "POSICION",
    "precio_entrada": "ENTRY PRICE",
    "precio_salida": "EXIT PRICE",
    "colateral": "COLATERAL",
    "tamano_posicion": "TAMANO POSICION",
    "comision_total": "COMISIONES",
    "pnl": "PNL NETO",
    "roi": "ROI",
    "saldo_post": "BALANCE",
    "motivo_salida": "EXIT REASON",
    "duracion_velas": "DUR VELAS",
}

EQUITY_NAMES = {
    "punto": "PUNTO",
    "trade_num": "TRADE",
    "idx_salida": "IDX SALIDA",
    "ts_salida": "TS SALIDA",
    "saldo": "BALANCE",
}

MONEY_COLS = {
    "saldo_inicial",
    "saldo_final",
    "pnl_total",
    "pnl_promedio",
    "colateral",
    "tamano_posicion",
    "comision_total",
    "pnl",
    "saldo_post",
    "saldo",
}
PCT_COLS = {"win_rate", "roi_total", "max_drawdown", "roi"}
PCT_POINT_COLS = {"param_exit_sl_pct", "param_exit_tp_pct"}
INT_COLS = {
    "trial",
    "total_trades",
    "trades_ganadores",
    "trades_perdedores",
    "idx_senal",
    "idx_entrada",
    "idx_salida",
    "direccion",
    "duracion_velas",
    "punto",
    "trade_num",
    "param_exit_velas",
}
PRICE_COLS = {"precio_entrada", "precio_salida"}
DATE_COLS = {"ts_senal", "ts_entrada", "ts_salida"}
MAX_DETALLES_EXCEL = 5


def generar_excel(run_dir: Path, trials: list, mejor) -> Path:
    excel_dir = _base_resultados(run_dir) / "EXCEL"
    excel_dir.mkdir(parents=True, exist_ok=True)

    path = _unique_path(excel_dir / _nombre_resumen(mejor))
    resumen = resumen_trials_dataframe(trials)

    workbook = xlsxwriter.Workbook(
        str(path),
        {
            "nan_inf_to_errors": True,
            "default_date_format": "dd/mm/yy hh:mm",
        },
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


def _generar_excel_detalle(excel_dir: Path, trial) -> Path:
    path = _unique_path(excel_dir / _nombre_detalle(trial))
    trades = trades_dataframe(trial.resultado)
    equity = equity_dataframe(trial.resultado)

    workbook = xlsxwriter.Workbook(
        str(path),
        {
            "nan_inf_to_errors": True,
            "default_date_format": "dd/mm/yy hh:mm",
        },
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
    worksheet = workbook.add_worksheet("RESUMEN")
    worksheet.hide_gridlines(2)
    columnas = _columnas_resumen(df)
    secciones = [_seccion_resumen(col) for col in columnas]

    _write_section_headers(worksheet, secciones, formatos)
    _write_headers(worksheet, 1, columnas, SUMMARY_NAMES, formatos["header"])

    for row_idx, row in enumerate(df.select(columnas).rows(named=True), start=2):
        alt = row_idx % 2 == 0
        worksheet.set_row(row_idx, 22)
        for col_idx, name in enumerate(columnas):
            fmt = _format_for(name, formatos, alt)
            _write_cell(worksheet, row_idx, col_idx, row[name], fmt, name)

    worksheet.freeze_panes(2, 0)
    worksheet.autofilter(1, 0, max(df.height + 1, 1), max(len(columnas) - 1, 0))
    worksheet.set_row(0, 22)
    worksheet.set_row(1, 30)
    _set_widths(worksheet, columnas, df, SUMMARY_NAMES)
    _add_summary_conditionals(worksheet, columnas, df.height, formatos)


def _write_trades(workbook, df: pl.DataFrame, filas_equity: int, formatos: dict) -> None:
    worksheet = workbook.add_worksheet("TRADES")
    worksheet.hide_gridlines(2)
    columnas = list(df.columns)

    _write_headers(worksheet, 0, columnas, TRADE_NAMES, formatos["header"])
    for row_idx, row in enumerate(df.rows(named=True), start=1):
        alt = row_idx % 2 == 0
        worksheet.set_row(row_idx, 20)
        for col_idx, name in enumerate(columnas):
            fmt = _format_for(name, formatos, alt)
            _write_cell(worksheet, row_idx, col_idx, row[name], fmt, name)

    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(df.height, 1), max(len(columnas) - 1, 0))
    worksheet.set_row(0, 30)
    _set_widths(worksheet, columnas, df, TRADE_NAMES)
    _add_trade_conditionals(worksheet, columnas, df.height, formatos)

    if df.height > 0:
        _insert_balance_chart(workbook, worksheet, start_row=df.height + 3, filas_equity=filas_equity)


def _write_equity(workbook, df: pl.DataFrame, formatos: dict) -> None:
    worksheet = workbook.add_worksheet("EQUITY")
    worksheet.hide_gridlines(2)
    columnas = list(df.columns)

    _write_headers(worksheet, 0, columnas, EQUITY_NAMES, formatos["header"])
    for row_idx, row in enumerate(df.rows(named=True), start=1):
        alt = row_idx % 2 == 0
        worksheet.set_row(row_idx, 20)
        for col_idx, name in enumerate(columnas):
            fmt = _format_for(name, formatos, alt)
            _write_cell(worksheet, row_idx, col_idx, row[name], fmt, name)

    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(df.height, 1), max(len(columnas) - 1, 0))
    worksheet.set_row(0, 30)
    _set_widths(worksheet, columnas, df, EQUITY_NAMES)


def _insert_balance_chart(workbook, worksheet, *, start_row: int, filas_equity: int) -> None:
    last_row = max(filas_equity + 1, 2)
    chart = workbook.add_chart({"type": "line"})
    chart.add_series(
        {
            "name": "Balance",
            "categories": f"=EQUITY!$B$2:$B${last_row}",
            "values": f"=EQUITY!$E$2:$E${last_row}",
            "line": {"color": "#2B6CB0", "width": 2.0},
        }
    )
    chart.set_title({"name": "Evolucion del balance"})
    chart.set_legend({"none": True})
    chart.set_y_axis(
        {
            "num_format": "#,##0",
            "major_gridlines": {"visible": True, "line": {"color": "#E5E7EB"}},
            "line": {"none": True},
        }
    )
    chart.set_x_axis({"name": "Trade", "major_gridlines": {"visible": False}})
    chart.set_plotarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
    chart.set_chartarea({"border": {"none": True}, "fill": {"color": "#FCFCFD"}})
    chart.set_size({"width": 1000, "height": 420})
    worksheet.insert_chart(start_row, 1, chart, {"x_offset": 2, "y_offset": 5})


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

    if name in DATE_COLS:
        dt = _datetime_from_us(value)
        if dt is None:
            worksheet.write(row, col, value, formato)
        else:
            worksheet.write_datetime(row, col, dt, formato)
        return

    if isinstance(value, bool):
        worksheet.write_string(row, col, "SI" if value else "NO", formato)
    elif isinstance(value, int):
        worksheet.write_number(row, col, value, formato)
    elif isinstance(value, float):
        worksheet.write_number(row, col, value, formato)
    else:
        worksheet.write_string(row, col, str(value), formato)


def _crear_formatos(workbook) -> dict:
    base = {
        "font_name": "Calibri",
        "font_size": 10,
        "font_color": "#2D3436",
        "align": "center",
        "valign": "vcenter",
        "bottom": 1,
        "bottom_color": "#E2E8F0",
    }

    def fmt(bg: str, num_format: str | None = None, **extra):
        params = {**base, "bg_color": bg}
        if num_format is not None:
            params["num_format"] = num_format
        params.update(extra)
        return workbook.add_format(params)

    return {
        "section": workbook.add_format(
            {
                "font_name": "Calibri",
                "font_size": 9,
                "bold": True,
                "font_color": "#718096",
                "bg_color": "#EDF2F7",
                "align": "center",
                "valign": "vcenter",
                "bottom": 1,
                "bottom_color": "#CBD5E0",
            }
        ),
        "header": workbook.add_format(
            {
                "font_name": "Calibri",
                "font_size": 10,
                "bold": True,
                "font_color": "#718096",
                "bg_color": "#F7FAFC",
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
                "bottom": 2,
                "bottom_color": "#A0AEC0",
            }
        ),
        "text": (fmt("#FFFFFF"), fmt("#F7FAFC")),
        "int": (fmt("#FFFFFF", "0"), fmt("#F7FAFC", "0")),
        "num": (fmt("#FFFFFF", "0.000000"), fmt("#F7FAFC", "0.000000")),
        "ratio": (fmt("#FFFFFF", "0.00"), fmt("#F7FAFC", "0.00")),
        "money": (fmt("#FFFFFF", "$#,##0.00"), fmt("#F7FAFC", "$#,##0.00")),
        "price": (fmt("#FFFFFF", "#,##0.00"), fmt("#F7FAFC", "#,##0.00")),
        "pct": (fmt("#FFFFFF", "0.00%"), fmt("#F7FAFC", "0.00%")),
        "pct_point": (fmt("#FFFFFF", '0.0"%"'), fmt("#F7FAFC", '0.0"%"')),
        "date": (fmt("#FFFFFF", "dd/mm/yy hh:mm"), fmt("#F7FAFC", "dd/mm/yy hh:mm")),
        "positive": workbook.add_format({"font_color": "#276749", "bold": True}),
        "negative": workbook.add_format({"font_color": "#C53030", "bold": True}),
    }


def _format_for(name: str, formatos: dict, alt: bool):
    idx = 1 if alt else 0
    if name in DATE_COLS:
        return formatos["date"][idx]
    if name in MONEY_COLS:
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


def _add_summary_conditionals(worksheet, columnas: list[str], filas: int, formatos: dict) -> None:
    if filas <= 0:
        return
    for name in ("roi_total", "pnl_total", "score"):
        if name not in columnas:
            continue
        col = columnas.index(name)
        worksheet.conditional_format(
            2,
            col,
            filas + 1,
            col,
            {"type": "cell", "criteria": ">", "value": 0, "format": formatos["positive"]},
        )
        worksheet.conditional_format(
            2,
            col,
            filas + 1,
            col,
            {"type": "cell", "criteria": "<", "value": 0, "format": formatos["negative"]},
        )


def _add_trade_conditionals(worksheet, columnas: list[str], filas: int, formatos: dict) -> None:
    if filas <= 0 or "pnl" not in columnas:
        return
    col = columnas.index("pnl")
    worksheet.conditional_format(
        1,
        col,
        filas,
        col,
        {"type": "cell", "criteria": ">", "value": 0, "format": formatos["positive"]},
    )
    worksheet.conditional_format(
        1,
        col,
        filas,
        col,
        {"type": "cell", "criteria": "<", "value": 0, "format": formatos["negative"]},
    )


def _set_widths(worksheet, columnas: list[str], df: pl.DataFrame, nombres: dict) -> None:
    rows = df.select(columnas).head(25).rows(named=True) if columnas else []
    for col_idx, name in enumerate(columnas):
        width = len(_display_name(name, nombres)) + 2
        for row in rows:
            value = row[name]
            if value is not None:
                width = max(width, min(len(str(value)) + 2, 28))
        worksheet.set_column(col_idx, col_idx, min(max(width, 10), 28))


def _columnas_resumen(df: pl.DataFrame) -> list[str]:
    existentes = set(df.columns)
    ids = [col for col in ID_ORDER if col in existentes]
    metricas = [col for col in METRIC_ORDER if col in existentes]
    params = sorted(col for col in df.columns if col.startswith("param_"))
    usados = set(ids + metricas + params)
    otros = [col for col in df.columns if col not in usados]
    return ids + metricas + otros + params


def _seccion_resumen(col: str) -> str:
    if col in ID_ORDER:
        return "DATOS"
    if col.startswith("param_"):
        return "PARAMETROS"
    return "METRICAS"


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

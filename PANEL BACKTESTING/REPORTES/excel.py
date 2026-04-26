from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import xlsxwriter

from REPORTES.persistencia import equity_dataframe, resumen_trials_dataframe, trades_dataframe


def generar_excel(run_dir: Path, trials: list, mejor) -> Path:
    path = run_dir / "resultados.xlsx"
    resumen = resumen_trials_dataframe(trials)
    trades = trades_dataframe(mejor.resultado)
    equity = equity_dataframe(mejor.resultado)

    workbook = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
    try:
        _write_dataframe(workbook, "RESUMEN", resumen)
        _write_dataframe(workbook, "TRADES", trades)
        _write_dataframe(workbook, "EQUITY", equity)
    finally:
        workbook.close()

    verificar_excel(path, resumen.height, trades.height, equity.height)
    return path


def verificar_excel(
    path: Path,
    filas_resumen: int,
    filas_trades: int,
    filas_equity: int,
) -> None:
    if not path.exists():
        raise ValueError(f"[EXCEL] No se genero {path}.")

    esperadas = {
        "xl/worksheets/sheet1.xml": filas_resumen + 1,
        "xl/worksheets/sheet2.xml": filas_trades + 1,
        "xl/worksheets/sheet3.xml": filas_equity + 1,
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


def _write_dataframe(workbook: xlsxwriter.Workbook, sheet_name: str, df: pl.DataFrame) -> None:
    worksheet = workbook.add_worksheet(sheet_name)
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#E7EEF8", "border": 1})
    num_fmt = workbook.add_format({"num_format": "0.000000"})
    money_fmt = workbook.add_format({"num_format": "$#,##0.00"})
    pct_fmt = workbook.add_format({"num_format": "0.00%"})

    for col_idx, name in enumerate(df.columns):
        worksheet.write(0, col_idx, name, header_fmt)

    for row_idx, row in enumerate(df.rows(named=True), start=1):
        for col_idx, name in enumerate(df.columns):
            value = row[name]
            fmt = _format_for(name, num_fmt, money_fmt, pct_fmt)
            worksheet.write(row_idx, col_idx, value, fmt)

    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(df.height, 1), max(len(df.columns) - 1, 0))
    for col_idx, name in enumerate(df.columns):
        width = min(max(len(name) + 2, 12), 32)
        worksheet.set_column(col_idx, col_idx, width)


def _format_for(name: str, num_fmt, money_fmt, pct_fmt):
    if name in {"saldo_inicial", "saldo_final", "pnl_total", "pnl_promedio", "saldo"}:
        return money_fmt
    if name in {"score", "roi_total", "win_rate", "max_drawdown", "roi"}:
        return pct_fmt
    if name in {"profit_factor", "sharpe_ratio", "duracion_media_velas"}:
        return num_fmt
    return None

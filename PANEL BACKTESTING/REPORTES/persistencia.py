from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from math import isclose
from pathlib import Path

import polars as pl


TOL = 1e-7


def guardar_resultado(
    *,
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_id: int,
    estrategia_nombre: str,
    parametros: dict,
    salida,
    resultado,
    score: float,
    huella_base,
    huella_timeframe,
    conteo_senales: dict[int, int],
) -> Path:
    run_dir = _crear_run_dir(
        carpeta_resultados,
        activo,
        timeframe,
        estrategia_nombre,
        salida.tipo,
    )

    resumen = _crear_resumen(
        activo,
        timeframe,
        estrategia_id,
        estrategia_nombre,
        parametros,
        salida,
        resultado,
        score,
    )
    auditoria = _crear_auditoria(
        huella_base,
        huella_timeframe,
        conteo_senales,
        resultado,
    )

    _write_json(run_dir / "resumen.json", resumen)
    _write_json(run_dir / "auditoria.json", auditoria)
    _trades_dataframe(resultado).write_csv(run_dir / "trades.csv")
    _equity_dataframe(resultado).write_csv(run_dir / "equity.csv")

    verificar_archivos(run_dir, resultado, conteo_senales)
    return run_dir


def verificar_archivos(run_dir: Path, resultado, conteo_senales: dict[int, int]) -> None:
    resumen_path = run_dir / "resumen.json"
    auditoria_path = run_dir / "auditoria.json"
    trades_path = run_dir / "trades.csv"
    equity_path = run_dir / "equity.csv"

    for path in (resumen_path, auditoria_path, trades_path, equity_path):
        if not path.exists():
            raise ValueError(f"[REPORTES] No se genero {path}.")

    resumen = json.loads(resumen_path.read_text(encoding="utf-8"))
    auditoria = json.loads(auditoria_path.read_text(encoding="utf-8"))
    trades = pl.read_csv(trades_path)
    equity = pl.read_csv(equity_path)

    total_trades = int(resultado.total_trades)
    if int(resumen["metricas"]["total_trades"]) != total_trades:
        raise ValueError("[REPORTES] resumen.json no conserva total_trades.")
    if trades.height != total_trades:
        raise ValueError("[REPORTES] trades.csv no conserva el numero de trades.")
    if equity.height != total_trades + 1:
        raise ValueError("[REPORTES] equity.csv no conserva la curva de equity completa.")

    if not isclose(float(resumen["metricas"]["saldo_final"]), float(resultado.saldo_final), abs_tol=TOL):
        raise ValueError("[REPORTES] resumen.json no conserva saldo_final.")
    if not isclose(float(equity["saldo"][-1]), float(resultado.saldo_final), abs_tol=TOL):
        raise ValueError("[REPORTES] equity.csv no termina en saldo_final.")

    senales = auditoria["senales"]
    if int(senales["long"]) != int(conteo_senales[1]):
        raise ValueError("[REPORTES] auditoria.json no conserva conteo LONG.")
    if int(senales["short"]) != int(conteo_senales[-1]):
        raise ValueError("[REPORTES] auditoria.json no conserva conteo SHORT.")
    if int(senales["sin_senal"]) != int(conteo_senales[0]):
        raise ValueError("[REPORTES] auditoria.json no conserva conteo sin senal.")


def _crear_run_dir(
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_nombre: str,
    exit_type: str,
) -> Path:
    base = (
        carpeta_resultados
        / _slug(activo)
        / _slug(timeframe)
        / _slug(estrategia_nombre)
        / _slug(exit_type)
    )
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{timestamp}"
    contador = 1
    while run_dir.exists():
        contador += 1
        run_dir = base / f"run_{timestamp}_{contador:02d}"

    run_dir.mkdir()
    return run_dir


def _crear_resumen(
    activo: str,
    timeframe: str,
    estrategia_id: int,
    estrategia_nombre: str,
    parametros: dict,
    salida,
    resultado,
    score: float,
) -> dict:
    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "activo": activo,
        "timeframe": timeframe,
        "estrategia": {
            "id": estrategia_id,
            "nombre": estrategia_nombre,
            "parametros": parametros,
        },
        "salida": asdict(salida),
        "metricas": {
            "score": float(score),
            "saldo_inicial": float(resultado.saldo_inicial),
            "saldo_final": float(resultado.saldo_final),
            "total_trades": int(resultado.total_trades),
            "trades_ganadores": int(resultado.trades_ganadores),
            "trades_perdedores": int(resultado.trades_perdedores),
            "win_rate": float(resultado.win_rate),
            "roi_total": float(resultado.roi_total),
            "pnl_total": float(resultado.pnl_total),
            "pnl_promedio": float(resultado.pnl_promedio),
            "max_drawdown": float(resultado.max_drawdown),
            "parado_por_saldo": bool(resultado.parado_por_saldo),
        },
    }


def _crear_auditoria(
    huella_base,
    huella_timeframe,
    conteo_senales: dict[int, int],
    resultado,
) -> dict:
    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "datos_base": _huella_dict(huella_base),
        "datos_timeframe": _huella_dict(huella_timeframe),
        "senales": {
            "long": int(conteo_senales[1]),
            "short": int(conteo_senales[-1]),
            "sin_senal": int(conteo_senales[0]),
            "total": int(sum(conteo_senales.values())),
        },
        "resultado": {
            "trades_csv_filas_esperadas": int(resultado.total_trades),
            "equity_csv_filas_esperadas": int(resultado.total_trades) + 1,
            "checks": [
                "timestamps ordenados y sin duplicados",
                "OHLC coherente",
                "resampleo validado",
                "senales alineadas con datos",
                "trades alineados con senales y velas",
                "pnl/equity/saldo consistentes",
                "archivos leidos y verificados tras escritura",
            ],
        },
    }


def _trades_dataframe(resultado) -> pl.DataFrame:
    filas = []
    for trade in resultado.trades:
        filas.append(
            {
                "idx_senal": int(getattr(trade, "idx_se\u00f1al")),
                "idx_entrada": int(trade.idx_entrada),
                "idx_salida": int(trade.idx_salida),
                "ts_senal": int(getattr(trade, "ts_se\u00f1al")),
                "ts_entrada": int(trade.ts_entrada),
                "ts_salida": int(trade.ts_salida),
                "direccion": int(trade.direccion),
                "precio_entrada": float(trade.precio_entrada),
                "precio_salida": float(trade.precio_salida),
                "colateral": float(trade.colateral),
                "tamano_posicion": float(getattr(trade, "tama\u00f1o_posicion")),
                "comision_total": float(trade.comision_total),
                "pnl": float(trade.pnl),
                "roi": float(trade.roi),
                "saldo_post": float(trade.saldo_post),
                "motivo_salida": str(trade.motivo_salida),
                "duracion_velas": int(trade.duracion_velas),
            }
        )

    return pl.DataFrame(
        filas,
        schema={
            "idx_senal": pl.Int64,
            "idx_entrada": pl.Int64,
            "idx_salida": pl.Int64,
            "ts_senal": pl.Int64,
            "ts_entrada": pl.Int64,
            "ts_salida": pl.Int64,
            "direccion": pl.Int8,
            "precio_entrada": pl.Float64,
            "precio_salida": pl.Float64,
            "colateral": pl.Float64,
            "tamano_posicion": pl.Float64,
            "comision_total": pl.Float64,
            "pnl": pl.Float64,
            "roi": pl.Float64,
            "saldo_post": pl.Float64,
            "motivo_salida": pl.String,
            "duracion_velas": pl.Int64,
        },
    )


def _equity_dataframe(resultado) -> pl.DataFrame:
    trades = list(resultado.trades)
    filas = [
        {
            "punto": 0,
            "trade_num": 0,
            "idx_salida": None,
            "ts_salida": None,
            "saldo": float(resultado.equity_curve[0]),
        }
    ]

    for i, saldo in enumerate(resultado.equity_curve[1:], start=1):
        trade = trades[i - 1]
        filas.append(
            {
                "punto": i,
                "trade_num": i,
                "idx_salida": int(trade.idx_salida),
                "ts_salida": int(trade.ts_salida),
                "saldo": float(saldo),
            }
        )

    return pl.DataFrame(
        filas,
        schema={
            "punto": pl.Int64,
            "trade_num": pl.Int64,
            "idx_salida": pl.Int64,
            "ts_salida": pl.Int64,
            "saldo": pl.Float64,
        },
    )


def _huella_dict(huella) -> dict:
    return {
        "etapa": huella.etapa,
        "filas": int(huella.filas),
        "columnas": list(huella.columnas),
        "ts_inicio": str(huella.ts_inicio),
        "ts_fin": str(huella.ts_fin),
    }


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _slug(valor: str) -> str:
    normalizado = unicodedata.normalize("NFKD", str(valor))
    ascii_text = normalizado.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    return slug or "sin_nombre"

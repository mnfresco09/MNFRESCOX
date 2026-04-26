from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from math import isclose
from pathlib import Path

import polars as pl


TOL = 1e-7


def guardar_optimizacion(
    *,
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_id: int,
    estrategia_nombre: str,
    salida,
    trials: list,
    mejor,
    huella_base,
    huella_timeframe,
    conteo_senales_mejor: dict[int, int],
    conteo_salidas_mejor: dict[int, int] | None,
    max_archivos: int,
) -> Path:
    _verificar_trials_misma_combinacion(
        trials=trials,
        activo=activo,
        timeframe=timeframe,
        estrategia_id=estrategia_id,
        estrategia_nombre=estrategia_nombre,
        salida_tipo=salida.tipo,
    )

    run_dir = crear_run_dir(
        carpeta_resultados=carpeta_resultados,
        activo=activo,
        timeframe=timeframe,
        estrategia_nombre=estrategia_nombre,
        exit_type=salida.tipo,
        max_archivos=max_archivos,
    )

    _write_json(
        run_dir / "resumen.json",
        _crear_resumen_optimizacion(
            activo=activo,
            timeframe=timeframe,
            estrategia_id=estrategia_id,
            estrategia_nombre=estrategia_nombre,
            salida=salida,
            trials=trials,
            mejor=mejor,
        ),
    )
    _write_json(
        run_dir / "auditoria.json",
        _crear_auditoria(
            huella_base=huella_base,
            huella_timeframe=huella_timeframe,
            conteo_senales=conteo_senales_mejor,
            conteo_salidas=conteo_salidas_mejor,
            resultado=mejor.resultado,
            total_trials=len(trials),
        ),
    )

    resumen_trials_dataframe(trials).write_csv(run_dir / "trials.csv")
    trades_dataframe(mejor.resultado).write_csv(run_dir / "trades.csv")
    equity_dataframe(mejor.resultado).write_csv(run_dir / "equity.csv")

    verificar_optimizacion(run_dir, trials, mejor, conteo_senales_mejor, conteo_salidas_mejor)
    return run_dir


def crear_run_dir(
    *,
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_nombre: str,
    exit_type: str,
    max_archivos: int,
) -> Path:
    base = (
        carpeta_resultados
        / slug(estrategia_nombre).upper()
        / slug(exit_type).upper()
        / slug(activo).upper()
        / slug(timeframe).upper()
    )
    datos_dir = base / "DATOS"
    datos_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = datos_dir / f"RUN_{timestamp}"
    contador = 1
    while run_dir.exists():
        contador += 1
        run_dir = datos_dir / f"RUN_{timestamp}_{contador:02d}"

    run_dir.mkdir()
    _rotar_runs(datos_dir, max_archivos=max_archivos)
    return run_dir


def resumen_trials_dataframe(trials: list) -> pl.DataFrame:
    param_keys = sorted(
        {k for trial in trials for k in trial.parametros.keys()}
        | {"exit_type", "exit_sl_pct", "exit_tp_pct", "exit_velas"}
    )
    filas = []

    for trial in sorted(trials, key=lambda t: t.score, reverse=True):
        fila = {
            "trial": int(trial.numero),
            "score": float(trial.score),
            "activo": trial.activo,
            "timeframe": trial.timeframe,
            "estrategia": trial.estrategia_nombre,
            "salida": trial.salida.tipo,
        }
        fila.update(_normalizar_metricas(trial.metricas))
        parametros = dict(trial.parametros)
        parametros.update(
            {
                "exit_type": trial.salida.tipo,
                "exit_sl_pct": trial.salida.sl_pct,
                "exit_tp_pct": trial.salida.tp_pct,
                "exit_velas": trial.salida.velas,
            }
        )
        for key in param_keys:
            fila[f"param_{key}"] = parametros.get(key)
        filas.append(fila)

    return pl.DataFrame(filas)


def trades_dataframe(resultado) -> pl.DataFrame:
    filas = []
    for trade in resultado.trades:
        pnl_neto = float(trade.pnl)
        comision = float(trade.comision_total)
        dur_seg = max(0, int(trade.ts_salida - trade.ts_entrada)) / 1_000_000
        filas.append(
            {
                "idx_senal": int(getattr(trade, "idx_señal")),
                "idx_entrada": int(trade.idx_entrada),
                "idx_salida": int(trade.idx_salida),
                "ts_senal": int(getattr(trade, "ts_señal")),
                "ts_entrada": int(trade.ts_entrada),
                "ts_salida": int(trade.ts_salida),
                "direccion": int(trade.direccion),
                "direccion_txt": "LONG" if int(trade.direccion) == 1 else "SHORT",
                "precio_entrada": float(trade.precio_entrada),
                "precio_salida": float(trade.precio_salida),
                "saldo_apertura": float(trade.colateral),
                "tamano_posicion": float(getattr(trade, "tamaño_posicion")),
                "comision_total": float(comision),
                "pnl_bruto": float(pnl_neto + comision),
                "pnl": float(pnl_neto),
                "roi": float(trade.roi),
                "saldo_post": float(trade.saldo_post),
                "motivo_salida": str(trade.motivo_salida),
                "duracion_velas": int(trade.duracion_velas),
                "duracion_seg": float(dur_seg),
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
            "direccion_txt": pl.String,
            "precio_entrada": pl.Float64,
            "precio_salida": pl.Float64,
            "saldo_apertura": pl.Float64,
            "tamano_posicion": pl.Float64,
            "comision_total": pl.Float64,
            "pnl_bruto": pl.Float64,
            "pnl": pl.Float64,
            "roi": pl.Float64,
            "saldo_post": pl.Float64,
            "motivo_salida": pl.String,
            "duracion_velas": pl.Int64,
            "duracion_seg": pl.Float64,
        },
    )


def equity_dataframe(resultado) -> pl.DataFrame:
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


def verificar_optimizacion(
    run_dir: Path,
    trials: list,
    mejor,
    conteo_senales: dict[int, int],
    conteo_salidas: dict[int, int] | None,
) -> None:
    resumen_path = run_dir / "resumen.json"
    auditoria_path = run_dir / "auditoria.json"
    trials_path = run_dir / "trials.csv"
    trades_path = run_dir / "trades.csv"
    equity_path = run_dir / "equity.csv"

    for path in (resumen_path, auditoria_path, trials_path, trades_path, equity_path):
        if not path.exists():
            raise ValueError(f"[REPORTES] No se genero {path}.")

    resumen = json.loads(resumen_path.read_text(encoding="utf-8"))
    auditoria = json.loads(auditoria_path.read_text(encoding="utf-8"))
    trials_df = pl.read_csv(trials_path)
    trades_df = pl.read_csv(trades_path)
    equity_df = pl.read_csv(equity_path)

    total_trials = len(trials)
    total_trades = len(list(mejor.resultado.trades))
    if int(resumen["optimizacion"]["total_trials"]) != total_trials:
        raise ValueError("[REPORTES] resumen.json no conserva total_trials.")
    if trials_df.height != total_trials:
        raise ValueError("[REPORTES] trials.csv no conserva todos los trials.")
    if trades_df.height != total_trades:
        raise ValueError("[REPORTES] trades.csv no conserva trades del mejor trial.")
    if equity_df.height != total_trades + 1:
        raise ValueError("[REPORTES] equity.csv no conserva la curva del mejor trial.")
    if not isclose(float(equity_df["saldo"][-1]), float(mejor.resultado.saldo_final), abs_tol=TOL):
        raise ValueError("[REPORTES] equity.csv no termina en saldo_final.")

    senales = auditoria["senales_mejor_trial"]
    if int(senales["long"]) != int(conteo_senales[1]):
        raise ValueError("[REPORTES] auditoria.json no conserva conteo LONG.")
    if int(senales["short"]) != int(conteo_senales[-1]):
        raise ValueError("[REPORTES] auditoria.json no conserva conteo SHORT.")
    if int(senales["sin_senal"]) != int(conteo_senales[0]):
        raise ValueError("[REPORTES] auditoria.json no conserva conteo sin senal.")
    if conteo_salidas is not None:
        salidas = auditoria.get("salidas_custom_mejor_trial")
        if not isinstance(salidas, dict):
            raise ValueError("[REPORTES] auditoria.json no conserva salidas custom.")
        if int(salidas["cerrar_long"]) != int(conteo_salidas[1]):
            raise ValueError("[REPORTES] auditoria.json no conserva cierres LONG custom.")
        if int(salidas["cerrar_short"]) != int(conteo_salidas[-1]):
            raise ValueError("[REPORTES] auditoria.json no conserva cierres SHORT custom.")
        if int(salidas["sin_salida"]) != int(conteo_salidas[0]):
            raise ValueError("[REPORTES] auditoria.json no conserva conteo sin salida custom.")


def _verificar_trials_misma_combinacion(
    *,
    trials: list,
    activo: str,
    timeframe: str,
    estrategia_id: int,
    estrategia_nombre: str,
    salida_tipo: str,
) -> None:
    if not trials:
        raise ValueError("[REPORTES] No hay trials que guardar.")

    for trial in trials:
        if trial.activo != activo:
            raise ValueError("[REPORTES] Se intento mezclar activos en el mismo run.")
        if trial.timeframe != timeframe:
            raise ValueError("[REPORTES] Se intento mezclar timeframes en el mismo run.")
        if int(trial.estrategia_id) != int(estrategia_id):
            raise ValueError("[REPORTES] Se intento mezclar estrategias en el mismo run.")
        if trial.estrategia_nombre != estrategia_nombre:
            raise ValueError("[REPORTES] Nombre de estrategia inconsistente en trials.")
        if trial.salida.tipo != salida_tipo:
            raise ValueError("[REPORTES] Se intento mezclar tipos de salida en el mismo run.")


def slug(valor: str) -> str:
    normalizado = unicodedata.normalize("NFKD", str(valor))
    ascii_text = normalizado.encode("ascii", "ignore").decode("ascii")
    slug_text = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_text).strip("_").lower()
    return slug_text or "sin_nombre"


def _crear_resumen_optimizacion(
    *,
    activo: str,
    timeframe: str,
    estrategia_id: int,
    estrategia_nombre: str,
    salida,
    trials: list,
    mejor,
) -> dict:
    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "activo": activo,
        "timeframe": timeframe,
        "estrategia": {
            "id": int(estrategia_id),
            "nombre": estrategia_nombre,
        },
        "salida_base": asdict(salida),
        "optimizacion": {
            "total_trials": int(len(trials)),
            "mejor_trial": int(mejor.numero),
            "mejor_score": float(mejor.score),
            "mejores_parametros": dict(mejor.parametros),
        },
        "metricas_mejor_trial": _normalizar_metricas(mejor.metricas),
    }


def _crear_auditoria(
    *,
    huella_base,
    huella_timeframe,
    conteo_senales: dict[int, int],
    conteo_salidas: dict[int, int] | None,
    resultado,
    total_trials: int,
) -> dict:
    total_trades = len(list(resultado.trades))
    auditoria = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "datos_base": _huella_dict(huella_base),
        "datos_timeframe": _huella_dict(huella_timeframe),
        "senales_mejor_trial": {
            "long": int(conteo_senales[1]),
            "short": int(conteo_senales[-1]),
            "sin_senal": int(conteo_senales[0]),
            "total": int(sum(conteo_senales.values())),
        },
        "resultado_mejor_trial": {
            "trades_csv_filas_esperadas": int(total_trades),
            "equity_csv_filas_esperadas": int(total_trades) + 1,
            "total_trials_esperados": int(total_trials),
            "checks": [
                "timestamps ordenados y sin duplicados",
                "OHLC coherente",
                "resampleo validado",
                "senales alineadas con datos en cada trial",
                "salidas custom alineadas con datos cuando aplica",
                "trades alineados con senales y velas en cada trial",
                "pnl/equity/saldo consistentes en cada trial",
                "archivos core leidos y verificados tras escritura",
            ],
        },
    }
    if conteo_salidas is not None:
        auditoria["salidas_custom_mejor_trial"] = {
            "cerrar_long": int(conteo_salidas[1]),
            "cerrar_short": int(conteo_salidas[-1]),
            "sin_salida": int(conteo_salidas[0]),
            "total": int(sum(conteo_salidas.values())),
        }
    return auditoria


def _huella_dict(huella) -> dict:
    return {
        "etapa": huella.etapa,
        "filas": int(huella.filas),
        "columnas": list(huella.columnas),
        "ts_inicio": str(huella.ts_inicio),
        "ts_fin": str(huella.ts_fin),
    }


def _normalizar_metricas(metricas: dict) -> dict:
    normalizadas = {}
    for key, value in metricas.items():
        if isinstance(value, bool):
            normalizadas[key] = bool(value)
        elif isinstance(value, int):
            normalizadas[key] = int(value)
        elif isinstance(value, float):
            normalizadas[key] = float(value)
        else:
            normalizadas[key] = value
    return normalizadas


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _rotar_runs(base: Path, *, max_archivos: int) -> None:
    runs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("RUN_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in runs[int(max_archivos):]:
        shutil.rmtree(path)
    _verificar_rotacion(base, max_archivos=max_archivos)


def _verificar_rotacion(base: Path, *, max_archivos: int) -> None:
    runs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("RUN_")]
    if len(runs) > int(max_archivos):
        raise ValueError(
            "[REPORTES] Rotacion incompleta: "
            f"{len(runs)} runs conservados para max_archivos={max_archivos}."
        )

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from math import isclose
from pathlib import Path

import numpy as np
import polars as pl

from MOTOR.wrapper import MOTIVOS

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
            replay=mejor.replay,
            total_trials=len(trials),
        ),
    )

    resumen_trials_dataframe(trials).write_csv(run_dir / "trials.csv")
    trades_dataframe(mejor.replay).write_csv(run_dir / "trades.csv")
    equity_dataframe(mejor.replay).write_csv(run_dir / "equity.csv")

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
    base = _base_combinacion(
        carpeta_resultados=carpeta_resultados,
        activo=activo,
        timeframe=timeframe,
        estrategia_nombre=estrategia_nombre,
        exit_type=exit_type,
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


def preparar_resultados_combinacion(
    *,
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_nombre: str,
    exit_type: str,
) -> Path:
    base = _base_combinacion(
        carpeta_resultados=carpeta_resultados,
        activo=activo,
        timeframe=timeframe,
        estrategia_nombre=estrategia_nombre,
        exit_type=exit_type,
    )
    if base.exists():
        shutil.rmtree(base)
    return base


def _base_combinacion(
    *,
    carpeta_resultados: Path,
    activo: str,
    timeframe: str,
    estrategia_nombre: str,
    exit_type: str,
) -> Path:
    return (
        carpeta_resultados
        / slug(estrategia_nombre).upper()
        / slug(exit_type).upper()
        / slug(activo).upper()
        / slug(timeframe).upper()
    )


def resumen_trials_dataframe(trials: list) -> pl.DataFrame:
    param_keys = sorted(
        {k for trial in trials for k in trial.parametros.keys()}
        | {
            "exit_type",
            "exit_sl_pct",
            "exit_tp_pct",
            "exit_velas",
            "exit_trail_act_pct",
            "exit_trail_dist_pct",
        }
    )
    filas = []
    for trial in sorted(trials, key=lambda t: t.score, reverse=True):
        fila = {
            "trial": int(trial.numero),
            "score": float(trial.score),
            "activo": trial.activo,
            "timeframe": trial.timeframe,
            "timeframe_ejecucion": getattr(trial, "timeframe_ejecucion", trial.timeframe),
            "perturbacion_seed": getattr(trial, "perturbacion_seed", None),
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
                "exit_trail_act_pct": getattr(trial.salida, "trail_act_pct", 0.0),
                "exit_trail_dist_pct": getattr(trial.salida, "trail_dist_pct", 0.0),
            }
        )
        for key in param_keys:
            fila[f"param_{key}"] = parametros.get(key)
        filas.append(fila)
    return pl.DataFrame(filas)


def trades_dataframe(replay) -> pl.DataFrame:
    """Construye el DataFrame de trades directamente desde columnas numpy.
    Cero iteración Python: una sola asignación por columna.
    """
    if replay is None:
        raise ValueError("[REPORTES] No hay replay disponible para trades_dataframe.")
    t = replay.trades
    direccion = t["direccion"].astype(np.int8)
    pnl_neto = t["pnl"].astype(np.float64)
    comision = t["comision_total"].astype(np.float64)
    dur_seg = (t["ts_salida"] - t["ts_entrada"]).clip(min=0).astype(np.float64) / 1_000_000.0
    motivo = np.array(MOTIVOS, dtype=object)[t["motivo_salida"].astype(np.int64)]
    direccion_txt = np.where(direccion == 1, "LONG", "SHORT")

    return pl.DataFrame(
        {
            "idx_senal":      t["idx_senal"].astype(np.int64),
            "idx_entrada":    t["idx_entrada"].astype(np.int64),
            "idx_salida":     t["idx_salida"].astype(np.int64),
            "ts_senal":       t["ts_senal"].astype(np.int64),
            "ts_entrada":     t["ts_entrada"].astype(np.int64),
            "ts_salida":      t["ts_salida"].astype(np.int64),
            "direccion":      direccion,
            "direccion_txt":  direccion_txt,
            "precio_entrada": t["precio_entrada"].astype(np.float64),
            "precio_salida":  t["precio_salida"].astype(np.float64),
            "saldo_apertura": t["colateral"].astype(np.float64),
            "apalancamiento": t["apalancamiento"].astype(np.float64),
            "tamano_posicion": t["tamano_posicion"].astype(np.float64),
            "risk_vol_ewma":  t["risk_vol_ewma"].astype(np.float64),
            "risk_sl_dist_pct": t["risk_sl_dist_pct"].astype(np.float64),
            "comision_total": comision,
            "pnl_bruto":      pnl_neto + comision,
            "pnl":            pnl_neto,
            "roi":            t["roi"].astype(np.float64),
            "saldo_post":     t["saldo_post"].astype(np.float64),
            "motivo_salida":  motivo,
            "duracion_velas": t["duracion_velas"].astype(np.int64),
            "duracion_seg":   dur_seg,
        },
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
            "apalancamiento": pl.Float64,
            "tamano_posicion": pl.Float64,
            "risk_vol_ewma": pl.Float64,
            "risk_sl_dist_pct": pl.Float64,
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


def equity_dataframe(replay) -> pl.DataFrame:
    if replay is None:
        raise ValueError("[REPORTES] No hay replay disponible para equity_dataframe.")
    eq = replay.equity_curve.astype(np.float64)
    n_trades = int(replay.trades["idx_salida"].shape[0])
    if eq.shape[0] != n_trades + 1:
        raise ValueError("[REPORTES] equity_curve no coincide con número de trades.")

    punto = np.arange(eq.shape[0], dtype=np.int64)
    trade_num = np.arange(eq.shape[0], dtype=np.int64)
    if n_trades > 0:
        idx_salida_full = np.concatenate(([np.iinfo(np.int64).min], replay.trades["idx_salida"].astype(np.int64)))
        ts_salida_full = np.concatenate(([np.iinfo(np.int64).min], replay.trades["ts_salida"].astype(np.int64)))
        # marcar fila inicial como nulos en idx_salida y ts_salida
        df = pl.DataFrame(
            {
                "punto": punto,
                "trade_num": trade_num,
                "idx_salida": idx_salida_full,
                "ts_salida": ts_salida_full,
                "saldo": eq,
            }
        )
        # nulos sólo en la primera fila para idx_salida y ts_salida
        df = df.with_columns([
            pl.when(pl.col("punto") == 0).then(None).otherwise(pl.col("idx_salida")).alias("idx_salida"),
            pl.when(pl.col("punto") == 0).then(None).otherwise(pl.col("ts_salida")).alias("ts_salida"),
        ])
    else:
        df = pl.DataFrame(
            {
                "punto": punto,
                "trade_num": trade_num,
                "idx_salida": pl.Series("idx_salida", [None], dtype=pl.Int64),
                "ts_salida": pl.Series("ts_salida", [None], dtype=pl.Int64),
                "saldo": eq,
            }
        )
    return df.cast({
        "punto": pl.Int64,
        "trade_num": pl.Int64,
        "idx_salida": pl.Int64,
        "ts_salida": pl.Int64,
        "saldo": pl.Float64,
    })


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
    total_trades = int(mejor.replay.trades["idx_salida"].shape[0]) if mejor.replay else 0
    if int(resumen["optimizacion"]["total_trials"]) != total_trials:
        raise ValueError("[REPORTES] resumen.json no conserva total_trials.")
    if trials_df.height != total_trials:
        raise ValueError("[REPORTES] trials.csv no conserva todos los trials.")
    if trades_df.height != total_trades:
        raise ValueError("[REPORTES] trades.csv no conserva trades del mejor trial.")
    if equity_df.height != total_trades + 1:
        raise ValueError("[REPORTES] equity.csv no conserva la curva del mejor trial.")
    if not isclose(float(equity_df["saldo"][-1]), float(mejor.replay.metricas_obj.saldo_final), abs_tol=TOL):
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
        "timeframe_ejecucion_mejor_trial": getattr(mejor, "timeframe_ejecucion", timeframe),
        "estrategia": {
            "id": int(estrategia_id),
            "nombre": estrategia_nombre,
        },
        "salida_base": asdict(salida),
        "optimizacion": {
            "total_trials": int(len(trials)),
            "mejor_trial": int(mejor.numero),
            "mejor_score": float(mejor.score),
            "mejor_perturbacion_seed": getattr(mejor, "perturbacion_seed", None),
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
    replay,
    total_trials: int,
) -> dict:
    total_trades = int(replay.trades["idx_salida"].shape[0]) if replay else 0
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
                "trades alineados con senales y velas en el replay",
                "pnl/equity/saldo consistentes en el replay",
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

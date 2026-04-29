"""Puente Python ↔ Rust del motor de backtesting.

Expone dos modos:
  - `simular_metricas`  → dict de métricas (uso de Optuna en cada trial).
  - `simular_full`      → SimResultFull con columnas numpy (replay para reportes).

Ambas funciones reciben buffers numpy ya preparados por `NUCLEO.contexto` y
no convierten a list en ningún momento. Los arrays viajan zero-copy a Rust.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from types import ModuleType

import numpy as np


MOTOR_DIR = Path(__file__).resolve().parent
EXTENSION_NAME = "motor_backtesting"

# Códigos compactos del motor → strings (mismo orden que `tipos::motivo` en Rust).
MOTIVOS = ("SL", "TP", "BARS", "CUSTOM", "TRAILING", "END")


def simular_metricas(arrays, senales, *, sim_cfg, salidas_custom=None):
    """Devuelve el struct `Metricas` (escalares) para la simulación.

    Esta es la ruta caliente de Optuna: no genera trades en Python, no
    construye listas intermedias. El motor Rust libera el GIL durante el
    cómputo, lo que permite escalar con `n_jobs > 1`.
    """
    motor = cargar_motor()
    salidas = arrays.salidas_neutras if salidas_custom is None else _ensure_int8(salidas_custom)
    senales_arr = _ensure_int8(senales)
    risk_vol = _risk_vol_array(arrays, sim_cfg)
    _validar_longitud(arrays, senales_arr, salidas, risk_vol)
    return motor.simulate_metrics(
        arrays.timestamps,
        arrays.opens,
        arrays.highs,
        arrays.lows,
        arrays.closes,
        risk_vol,
        senales_arr,
        salidas,
        float(sim_cfg.saldo_inicial),
        float(sim_cfg.saldo_por_trade),
        float(sim_cfg.apalancamiento),
        float(sim_cfg.saldo_minimo),
        float(sim_cfg.comision_pct),
        int(sim_cfg.comision_lados),
        str(sim_cfg.exit_type),
        float(sim_cfg.exit_sl_pct),
        float(sim_cfg.exit_tp_pct),
        int(sim_cfg.exit_velas),
        float(getattr(sim_cfg, "exit_trail_act_pct", 0.0)),
        float(getattr(sim_cfg, "exit_trail_dist_pct", 0.0)),
        bool(getattr(sim_cfg, "paridad_riesgo", False)),
        float(getattr(sim_cfg, "paridad_riesgo_max_pct", 0.0)),
        float(getattr(sim_cfg, "paridad_apalancamiento_min", 1.0)),
        float(getattr(sim_cfg, "paridad_apalancamiento_max", 1.0)),
        float(getattr(sim_cfg, "exit_sl_ewma_mult", 0.0)),
        float(getattr(sim_cfg, "exit_tp_ewma_mult", 0.0)),
        float(getattr(sim_cfg, "exit_trail_act_ewma_mult", 0.0)),
        float(getattr(sim_cfg, "exit_trail_dist_ewma_mult", 0.0)),
        bool(getattr(sim_cfg, "paridad_skip_bajo_min", True)),
    )


def simular_full(arrays, senales, *, sim_cfg, salidas_custom=None):
    """Devuelve un `SimResult` con métricas + columnas numpy de los trades.

    Sólo se usa para los top-N trials que alimentan reportes (Excel / HTML /
    CSV). El objeto devuelto tiene `take_trades()` que entrega un dict de
    arrays numpy y libera la memoria interna del motor.
    """
    motor = cargar_motor()
    salidas = arrays.salidas_neutras if salidas_custom is None else _ensure_int8(salidas_custom)
    senales_arr = _ensure_int8(senales)
    risk_vol = _risk_vol_array(arrays, sim_cfg)
    _validar_longitud(arrays, senales_arr, salidas, risk_vol)
    return motor.simulate_full(
        arrays.timestamps,
        arrays.opens,
        arrays.highs,
        arrays.lows,
        arrays.closes,
        risk_vol,
        senales_arr,
        salidas,
        float(sim_cfg.saldo_inicial),
        float(sim_cfg.saldo_por_trade),
        float(sim_cfg.apalancamiento),
        float(sim_cfg.saldo_minimo),
        float(sim_cfg.comision_pct),
        int(sim_cfg.comision_lados),
        str(sim_cfg.exit_type),
        float(sim_cfg.exit_sl_pct),
        float(sim_cfg.exit_tp_pct),
        int(sim_cfg.exit_velas),
        float(getattr(sim_cfg, "exit_trail_act_pct", 0.0)),
        float(getattr(sim_cfg, "exit_trail_dist_pct", 0.0)),
        bool(getattr(sim_cfg, "paridad_riesgo", False)),
        float(getattr(sim_cfg, "paridad_riesgo_max_pct", 0.0)),
        float(getattr(sim_cfg, "paridad_apalancamiento_min", 1.0)),
        float(getattr(sim_cfg, "paridad_apalancamiento_max", 1.0)),
        float(getattr(sim_cfg, "exit_sl_ewma_mult", 0.0)),
        float(getattr(sim_cfg, "exit_tp_ewma_mult", 0.0)),
        float(getattr(sim_cfg, "exit_trail_act_ewma_mult", 0.0)),
        float(getattr(sim_cfg, "exit_trail_dist_ewma_mult", 0.0)),
        bool(getattr(sim_cfg, "paridad_skip_bajo_min", True)),
    )


def cargar_motor() -> ModuleType:
    ruta = _ruta_extension()
    if not ruta.exists() or _extension_obsoleta(ruta):
        _compilar_motor()
    try:
        return _importar_extension(ruta)
    except ImportError:
        _compilar_motor()
        return _importar_extension(ruta)


def _ensure_int8(serie_o_array) -> np.ndarray:
    """Devuelve un ndarray contiguo int8. Acepta pl.Series o np.ndarray."""
    if isinstance(serie_o_array, np.ndarray):
        if serie_o_array.dtype == np.int8 and serie_o_array.flags["C_CONTIGUOUS"]:
            return serie_o_array
        return np.ascontiguousarray(serie_o_array, dtype=np.int8)
    # pl.Series: vamos por to_numpy(). Si la serie ya es Int8, polars devuelve
    # la vista subyacente sin copia.
    arr = serie_o_array.to_numpy()
    if arr.dtype != np.int8 or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.int8)
    return arr


def _risk_vol_array(arrays, sim_cfg) -> np.ndarray:
    if bool(getattr(sim_cfg, "paridad_riesgo", False)):
        risk_vol = getattr(sim_cfg, "risk_vol_ewma", None)
        if risk_vol is None:
            raise ValueError("[PARIDAD] Falta risk_vol_ewma en SimConfigMotor.")
        return _ensure_float64(risk_vol)
    # Buffer f64 ya existente, sin asignar un array nuevo por trial.
    volumes = getattr(arrays, "volumes", None)
    if volumes is not None:
        return _ensure_float64(volumes)
    return np.zeros(arrays.timestamps.shape[0], dtype=np.float64)


def _ensure_float64(serie_o_array) -> np.ndarray:
    if isinstance(serie_o_array, np.ndarray):
        if serie_o_array.dtype == np.float64 and serie_o_array.flags["C_CONTIGUOUS"]:
            return serie_o_array
        return np.ascontiguousarray(serie_o_array, dtype=np.float64)
    arr = serie_o_array.to_numpy()
    if arr.dtype != np.float64 or not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def _validar_longitud(
    arrays,
    senales: np.ndarray,
    salidas: np.ndarray,
    risk_vol: np.ndarray,
) -> None:
    n = arrays.timestamps.shape[0]
    if senales.shape[0] != n:
        raise ValueError(
            f"Arrays y senales no coinciden: arrays={n:,}, senales={senales.shape[0]:,}."
        )
    if salidas.shape[0] != n:
        raise ValueError(
            f"Arrays y salidas no coinciden: arrays={n:,}, salidas={salidas.shape[0]:,}."
        )
    if risk_vol.shape[0] != n:
        raise ValueError(
            f"Arrays y risk_vol_ewma no coinciden: arrays={n:,}, risk_vol={risk_vol.shape[0]:,}."
        )


def _ruta_extension() -> Path:
    if sys.platform == "darwin":
        nombre = f"lib{EXTENSION_NAME}.dylib"
    elif sys.platform.startswith("linux"):
        nombre = f"lib{EXTENSION_NAME}.so"
    elif sys.platform == "win32":
        nombre = f"{EXTENSION_NAME}.dll"
    else:
        raise RuntimeError(f"Plataforma no soportada para el motor Rust: {sys.platform}")
    return MOTOR_DIR / "target" / "release" / nombre


def _compilar_motor() -> None:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=MOTOR_DIR,
        env=env,
        check=True,
    )


def _extension_obsoleta(ruta: Path) -> bool:
    if not ruta.exists():
        return True
    compilado = ruta.stat().st_mtime
    fuentes = [MOTOR_DIR / "Cargo.toml", MOTOR_DIR / "Cargo.lock", MOTOR_DIR / "build.rs"]
    fuentes.extend((MOTOR_DIR / "src").glob("*.rs"))
    return any(p.exists() and p.stat().st_mtime > compilado for p in fuentes)


def _importar_extension(ruta: Path) -> ModuleType:
    existente = sys.modules.get(EXTENSION_NAME)
    if existente is not None and Path(getattr(existente, "__file__", "")) == ruta:
        return existente
    sys.modules.pop(EXTENSION_NAME, None)
    loader = ExtensionFileLoader(EXTENSION_NAME, str(ruta))
    spec = spec_from_loader(EXTENSION_NAME, loader)
    if spec is None:
        raise ImportError(f"No se pudo crear spec para {ruta}")
    modulo = module_from_spec(spec)
    loader.exec_module(modulo)
    sys.modules[EXTENSION_NAME] = modulo
    return modulo

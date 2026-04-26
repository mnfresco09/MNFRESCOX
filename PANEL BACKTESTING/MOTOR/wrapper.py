from __future__ import annotations

import os
import subprocess
import sys
from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from types import ModuleType

import polars as pl


MOTOR_DIR = Path(__file__).resolve().parent
EXTENSION_NAME = "motor_backtesting"


def simular_dataframe(
    df: pl.DataFrame,
    senales: pl.Series,
    *,
    saldo_inicial: float,
    saldo_por_trade: float,
    apalancamiento: float,
    saldo_minimo: float,
    comision_pct: float,
    comision_lados: int,
    exit_type: str,
    exit_sl_pct: float,
    exit_tp_pct: float,
    exit_velas: int,
):
    """
    Convierte un DataFrame Polars validado al contrato plano que espera Rust.
    La funcion no filtra ni reordena filas: si algo no encaja, falla.
    """
    if df.height != len(senales):
        raise ValueError(
            f"Filas y senales no coinciden: df={df.height:,}, senales={len(senales):,}."
        )

    motor = cargar_motor()
    timestamps = _timestamps_us(df)

    return motor.simulate_trades(
        timestamps,
        df["open"].cast(pl.Float64).to_list(),
        df["high"].cast(pl.Float64).to_list(),
        df["low"].cast(pl.Float64).to_list(),
        df["close"].cast(pl.Float64).to_list(),
        _volume_list(df),
        senales.cast(pl.Int8).to_list(),
        float(saldo_inicial),
        float(saldo_por_trade),
        float(apalancamiento),
        float(saldo_minimo),
        float(comision_pct),
        int(comision_lados),
        str(exit_type),
        float(exit_sl_pct),
        float(exit_tp_pct),
        int(exit_velas),
    )


def cargar_motor() -> ModuleType:
    ruta = _ruta_extension()
    if not ruta.exists():
        _compilar_motor()

    try:
        return _importar_extension(ruta)
    except ImportError:
        _compilar_motor()
        return _importar_extension(ruta)


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


def _timestamps_us(df: pl.DataFrame) -> list[int]:
    dtype = df.schema.get("timestamp")
    if dtype is None:
        raise ValueError("El DataFrame no contiene columna 'timestamp'.")

    if isinstance(dtype, pl.Datetime):
        return df.select(pl.col("timestamp").dt.epoch("us")).to_series().to_list()

    if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        return df["timestamp"].cast(pl.Int64).to_list()

    raise ValueError(f"timestamp debe ser Datetime o entero en microsegundos, no {dtype}.")


def _volume_list(df: pl.DataFrame) -> list[float]:
    if "volume" not in df.columns:
        return [0.0] * df.height
    return df["volume"].cast(pl.Float64).fill_null(0.0).to_list()

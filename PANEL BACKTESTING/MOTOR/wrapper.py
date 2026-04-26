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


def simular(
    arrays,
    senales: pl.Series,
    *,
    sim_cfg,
    salidas_custom: pl.Series | None = None,
):
    """
    Ejecuta el motor Rust con arrays OHLCV ya precalculados.
    La funcion no filtra ni reordena filas: si algo no encaja, falla.
    """
    total = len(arrays)
    if total != len(senales):
        raise ValueError(
            f"Arrays y senales no coinciden: arrays={total:,}, senales={len(senales):,}."
        )
    if salidas_custom is None:
        salidas_lista = arrays.salidas_neutras
    else:
        if total != len(salidas_custom):
            raise ValueError(
                "Arrays y salidas_custom no coinciden: "
                f"arrays={total:,}, salidas_custom={len(salidas_custom):,}."
            )
        salidas_lista = salidas_custom.cast(pl.Int8).to_list()

    motor = cargar_motor()

    return motor.simulate_trades(
        arrays.timestamps,
        arrays.opens,
        arrays.highs,
        arrays.lows,
        arrays.closes,
        arrays.volumes,
        senales.cast(pl.Int8).to_list(),
        salidas_lista,
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
    return any(path.exists() and path.stat().st_mtime > compilado for path in fuentes)


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

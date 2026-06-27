"""Puente Python ↔ Rust del micro-motor de riesgo (MOTOR_RIESGO).

Expone dos cálculos iterativos pesados, idénticos en firma tengan o no Rust:

  • `fhs`         → Filtered Historical Simulation: VaR/CVaR 95/99 a T+1.
  • `montecarlo` → trayectorias por bootstrapping: percentiles del fan chart,
                    probabilidad de pérdida y CDaR a horizonte.

El motor Rust se compila bajo demanda (`cargo build --release`) como en
PANEL BACKTESTING/MOTOR, pero este crate es INDEPENDIENTE (MOTOR_RIESGO/). Si
Rust no está disponible (sin `cargo`, sin compilar, error de import), se cae a
una implementación NumPy equivalente y se marca la fuente como
"python_fallback". El motor NUNCA devuelve la matriz completa de trayectorias:
solo percentiles y agregados.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from importlib.machinery import ExtensionFileLoader
from importlib.util import spec_from_loader
from pathlib import Path
from types import ModuleType

import numpy as np

EXTENSION_NAME = "motor_riesgo"
MOTOR_DIR = Path(__file__).resolve().parents[1] / "MOTOR_RIESGO"

_motor_cache: ModuleType | None = None
_motor_intentado = False
_motor_lock = threading.Lock()


# ===========================================================================
#  Carga / compilación del crate Rust (best-effort)
# ===========================================================================
def _ruta_extension() -> Path:
    if sys.platform == "darwin":
        nombre = f"lib{EXTENSION_NAME}.dylib"
    elif sys.platform.startswith("linux"):
        nombre = f"lib{EXTENSION_NAME}.so"
    elif sys.platform == "win32":
        nombre = f"{EXTENSION_NAME}.dll"
    else:
        raise RuntimeError(f"Plataforma no soportada: {sys.platform}")
    return MOTOR_DIR / "target" / "release" / nombre


def _fuentes_mas_nuevas(destino: Path) -> bool:
    if not destino.exists():
        return True
    t = destino.stat().st_mtime
    fuentes = [MOTOR_DIR / "Cargo.toml", MOTOR_DIR / "build.rs"]
    fuentes += list((MOTOR_DIR / "src").glob("*.rs"))
    return any(p.exists() and p.stat().st_mtime > t for p in fuentes)


def _compilar() -> None:
    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    subprocess.run(["cargo", "build", "--release"], cwd=MOTOR_DIR, env=env, check=True)


def _importar(ruta: Path) -> ModuleType:
    existente = sys.modules.get(EXTENSION_NAME)
    if existente is not None and getattr(existente, "__file__", "") == str(ruta):
        return existente
    sys.modules.pop(EXTENSION_NAME, None)
    loader = ExtensionFileLoader(EXTENSION_NAME, str(ruta))
    spec = spec_from_loader(EXTENSION_NAME, loader)
    modulo = loader.create_module(spec)
    if modulo is None:
        import importlib.util as _u
        modulo = _u.module_from_spec(spec)
    loader.exec_module(modulo)
    sys.modules[EXTENSION_NAME] = modulo
    return modulo


_MARCADOR_FALLO = MOTOR_DIR / "target" / ".compilacion_fallida"


def _compilacion_fallida_vigente() -> bool:
    """True si una compilación previa falló y el código Rust no ha cambiado desde
    entonces. Evita reintentar (y esperar ~minutos) una build rota en cada
    ejecución; un cambio en las fuentes invalida el marcador automáticamente."""
    if not _MARCADOR_FALLO.exists():
        return False
    return not _fuentes_mas_nuevas(_MARCADOR_FALLO)


def cargar_motor() -> ModuleType | None:
    """Devuelve el módulo Rust o None si no es posible (→ fallback Python)."""
    global _motor_cache, _motor_intentado
    with _motor_lock:
        if _motor_cache is not None:
            return _motor_cache
        if _motor_intentado:
            return None
        _motor_intentado = True
        if not (MOTOR_DIR / "Cargo.toml").exists():
            return None
        ruta = _ruta_extension()
        try:
            if _fuentes_mas_nuevas(ruta):
                if _compilacion_fallida_vigente():
                    return None  # build rota conocida: directo al fallback, sin reintentar
                _compilar()
            _motor_cache = _importar(ruta)
            try:
                _MARCADOR_FALLO.unlink(missing_ok=True)
            except OSError:
                pass
            return _motor_cache
        except (subprocess.CalledProcessError, FileNotFoundError, ImportError, OSError, RuntimeError):
            try:
                _MARCADOR_FALLO.parent.mkdir(parents=True, exist_ok=True)
                _MARCADOR_FALLO.write_text("compilacion fallida", encoding="utf-8")
            except OSError:
                pass
            return None


# ===========================================================================
#  Implementaciones NumPy de respaldo (fallback)
# ===========================================================================
def _fhs_python(residuos: np.ndarray, sigma_next: float, niveles: np.ndarray) -> dict:
    """FHS: escala residuos estandarizados por la vol T+1 y lee cuantiles."""
    escalados = np.sort(residuos * float(sigma_next))
    salida = {}
    for nivel in niveles:
        alpha = 1.0 - float(nivel)
        var = float(np.quantile(escalados, alpha))
        cola = escalados[escalados <= var]
        cvar = float(cola.mean()) if cola.size else var
        salida[float(nivel)] = (var, cvar)
    return salida


def _montecarlo_python(
    retornos: np.ndarray,
    horizonte: int,
    n_traj: int,
    percentiles: np.ndarray,
    seed: int,
) -> dict:
    """Bootstrapping con reemplazo: percentiles de la senda de capital, prob. de
    pérdida y CDaR a horizonte. Solo agregados (no devuelve trayectorias)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, retornos.shape[0], size=(n_traj, horizonte))
    muestras = retornos[idx]                              # (n_traj, horizonte)
    log_acum = np.cumsum(np.log1p(muestras), axis=1)
    capital = np.exp(log_acum)                            # base 1, (n_traj, horizonte)

    sendas = np.percentile(capital, percentiles, axis=0)  # (n_perc, horizonte)
    ret_final = capital[:, -1] - 1.0
    prob_perdida = float(np.mean(ret_final < 0.0))

    # CDaR: media del peor (1-alpha) de los drawdowns máximos por trayectoria.
    picos = np.maximum.accumulate(capital, axis=1)
    drawdowns = capital / picos - 1.0
    max_dd = drawdowns.min(axis=1)                        # más negativo = peor
    umbral = np.quantile(max_dd, 0.05)                    # CDaR 95%
    peores = max_dd[max_dd <= umbral]
    cdar = float(peores.mean()) if peores.size else float(umbral)

    return {
        "sendas": sendas,
        "prob_perdida": prob_perdida,
        "cdar": cdar,
        "retorno_mediano": float(np.median(ret_final)),
        "perdida_p5": float(np.percentile(ret_final, 5)),
    }


# ===========================================================================
#  Interfaz pública (valida tipos y delega en Rust o fallback)
# ===========================================================================
def fhs(residuos_estandarizados, sigma_next: float, niveles=(0.95, 0.99)) -> tuple[dict, str]:
    """Devuelve ({nivel: (var, cvar)}, fuente)."""
    residuos = np.ascontiguousarray(np.asarray(residuos_estandarizados, dtype=np.float64))
    niveles_arr = np.asarray(niveles, dtype=np.float64)
    if residuos.ndim != 1 or residuos.size < 10:
        raise ValueError("FHS requiere un vector de al menos 10 residuos.")
    motor = cargar_motor()
    if motor is not None and hasattr(motor, "fhs"):
        var95, var99, cvar95, cvar99 = motor.fhs(residuos, float(sigma_next), niveles_arr)
        resultado = {0.95: (var95, cvar95), 0.99: (var99, cvar99)}
        return resultado, "rust"
    return _fhs_python(residuos, sigma_next, niveles_arr), "python_fallback"


def montecarlo(
    retornos_historicos,
    horizonte: int,
    n_trayectorias: int,
    percentiles=(5, 25, 50, 75, 95),
    seed: int = 42,
) -> tuple[dict, str]:
    """Devuelve (resumen_agregado, fuente). Nunca trayectorias completas."""
    retornos = np.ascontiguousarray(np.asarray(retornos_historicos, dtype=np.float64))
    perc = np.asarray(percentiles, dtype=np.float64)
    if retornos.ndim != 1 or retornos.size < 30:
        raise ValueError("Monte Carlo requiere al menos 30 retornos históricos.")
    motor = cargar_motor()
    if motor is not None and hasattr(motor, "montecarlo"):
        sendas, prob, cdar, ret_med, p5 = motor.montecarlo(
            retornos, int(horizonte), int(n_trayectorias), perc, int(seed)
        )
        return {
            "sendas": np.asarray(sendas), "prob_perdida": float(prob), "cdar": float(cdar),
            "retorno_mediano": float(ret_med), "perdida_p5": float(p5),
        }, "rust"
    return _montecarlo_python(retornos, int(horizonte), int(n_trayectorias), perc, int(seed)), "python_fallback"

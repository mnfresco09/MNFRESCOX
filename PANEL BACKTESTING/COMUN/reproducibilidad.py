"""Huella de reproducibilidad (Fase 0).

Regla de oro institucional: **un backtest debe ser reconstruible bit a bit desde
sus metadatos.** Esta huella combina, de forma DETERMINISTA, todo lo que define
el resultado de un run:

  - la configuración serializada (todos los parámetros que afectan al backtest),
  - el hash SHA-256 del fichero de datos que alimenta el run,
  - el commit de git del árbol de código (si es un repositorio; opcional),
  - las versiones de Python y de las dependencias relevantes.

`NUCLEO/integridad.py` ya verifica determinismo DENTRO de una sesión (replay de
los top-N). Esto lo extiende a determinismo ENTRE sesiones y máquinas: dos runs
con la misma huella deben producir exactamente el mismo resultado.

Diseño
------
- El `digest` final NO incluye la marca de tiempo de creación: dos runs idénticos
  lanzados en momentos distintos comparten huella. La marca de tiempo se guarda
  como metadato informativo, fuera del hash.
- git es opcional: si el árbol no es un repositorio o git no está instalado, los
  campos de git quedan a None y la huella sigue siendo válida (se apoya en el
  hash de datos + config + versiones).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Dependencias cuya versión registramos. El resultado de un backtest puede
# depender de la versión de cualquiera de estas (motor numérico, optimizador,
# E/S de datos). Si falta alguna, se registra como None sin romper la huella.
_DEPENDENCIAS = ("polars", "numpy", "optuna", "numba", "pyarrow")

_BLOQUE_LECTURA = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class HuellaReproducibilidad:
    """Metadatos deterministas de un run. `digest` es la huella canónica."""

    digest: str
    config_hash: str
    datos_hash: str
    datos_ruta: str
    datos_bytes: int
    git_commit: str | None
    git_sucio: bool | None
    python: str
    versiones: dict[str, str | None]
    creado_utc: str  # informativo; NO entra en el digest

    def como_dict(self) -> dict:
        return asdict(self)

    def resumen(self) -> str:
        git = self.git_commit[:10] if self.git_commit else "sin-git"
        sucio = " (árbol sucio)" if self.git_sucio else ""
        return f"huella={self.digest[:12]} datos={self.datos_hash[:10]} git={git}{sucio}"


def calcular_huella(cfg, ruta_datos, *, raiz: Path | None = None) -> HuellaReproducibilidad:
    """Construye la huella de reproducibilidad de un run.

    Parameters
    ----------
    cfg:
        Módulo/objeto de configuración (se serializan sus atributos en MAYÚSCULAS).
    ruta_datos:
        Ruta al fichero de datos que alimenta el run (se hashea su contenido).
    raiz:
        Raíz del repositorio para consultar git. Por defecto `cfg.RAIZ` si existe.
    """
    ruta = Path(ruta_datos)
    config_serializable = config_a_dict(cfg)
    config_hash = _sha256_texto(_json_canonico(config_serializable))
    datos_hash = hash_fichero(ruta)
    datos_bytes = ruta.stat().st_size if ruta.exists() else 0

    if raiz is None:
        raiz = getattr(cfg, "RAIZ", None)
    git_commit, git_sucio = info_git(raiz)

    versiones = versiones_dependencias()
    python = sys.version.split()[0]

    digest = _sha256_texto(
        _json_canonico(
            {
                "config_hash": config_hash,
                "datos_hash": datos_hash,
                "git_commit": git_commit,
                "python": python,
                "versiones": versiones,
            }
        )
    )

    return HuellaReproducibilidad(
        digest=digest,
        config_hash=config_hash,
        datos_hash=datos_hash,
        datos_ruta=str(ruta),
        datos_bytes=int(datos_bytes),
        git_commit=git_commit,
        git_sucio=git_sucio,
        python=python,
        versiones=versiones,
        creado_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def hash_fichero(ruta: Path, *, bloque: int = _BLOQUE_LECTURA) -> str:
    """SHA-256 hex del contenido de un fichero, leído por bloques (memoria O(1))."""
    ruta = Path(ruta)
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def config_a_dict(cfg) -> dict:
    """Serializa los atributos en MAYÚSCULAS de la config a tipos JSON-ables.

    Solo se toman atributos cuyo nombre está en MAYÚSCULAS (la convención del
    proyecto para parámetros), excluyendo módulos y callables. `Path` y otros
    tipos no nativos se convierten a `str` de forma estable.
    """
    salida: dict = {}
    for nombre in dir(cfg):
        if not nombre.isupper():
            continue
        valor = getattr(cfg, nombre)
        if callable(valor) or _es_modulo(valor):
            continue
        salida[nombre] = _normalizar_valor(valor)
    return salida


def info_git(raiz) -> tuple[str | None, bool | None]:
    """Devuelve `(commit, sucio)`; `(None, None)` si no hay repo o git no existe.

    `sucio` indica si hay cambios sin commitear (working tree dirty), señal de
    que el run podría no ser reconstruible desde un commit limpio.
    """
    if raiz is None:
        return None, None
    raiz = Path(raiz)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=raiz,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode != 0:
            return None, None
        estado = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=raiz,
            capture_output=True,
            text=True,
            timeout=5,
        )
        sucio = bool(estado.stdout.strip()) if estado.returncode == 0 else None
        return commit.stdout.strip(), sucio
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None, None


def versiones_dependencias() -> dict[str, str | None]:
    """Versiones instaladas de las dependencias relevantes (None si ausente)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return {dep: None for dep in _DEPENDENCIAS}

    salida: dict[str, str | None] = {}
    for dep in _DEPENDENCIAS:
        try:
            salida[dep] = version(dep)
        except PackageNotFoundError:
            salida[dep] = None
    return salida


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _normalizar_valor(valor):
    if isinstance(valor, Path):
        return str(valor)
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    if isinstance(valor, dict):
        return {str(k): _normalizar_valor(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        items = [_normalizar_valor(v) for v in valor]
        # `set` no tiene orden estable: se ordena para que el hash sea determinista.
        return sorted(items, key=repr) if isinstance(valor, set) else items
    return str(valor)


def _es_modulo(valor) -> bool:
    return type(valor).__name__ == "module"


def _json_canonico(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


def _sha256_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()

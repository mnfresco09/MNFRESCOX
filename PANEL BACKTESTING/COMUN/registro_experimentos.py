"""Registro de experimentos (Fase 0): base de datos consultable de todo lo que
se ha probado.

¿Por qué existe? El `N` correcto para corregir por testing múltiple (Deflated
Sharpe, Fase 3) **no son los 300 trials de un run** — son TODAS las
configuraciones que se han probado en ese activo a lo largo de la investigación.
Sin un registro persistente de cada trial de cada run, ese `N` es incalculable y
el DSR queda subestimado (demasiado optimista).

Cada trial de cada run queda guardado con su activo, estrategia, parámetros,
score, métricas, fecha y la huella de reproducibilidad del run. La base de datos
es SQLite (un solo fichero local, sin servidor), consultable con SQL plano.

El módulo es Python puro (solo `sqlite3` + `json` de la stdlib): no depende de
Polars, NumPy ni del motor, por lo que es verificable de forma aislada.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    creado_utc        TEXT    NOT NULL,
    activo            TEXT    NOT NULL,
    timeframe         TEXT    NOT NULL,
    estrategia_id     INTEGER NOT NULL,
    estrategia_nombre TEXT    NOT NULL,
    salida_tipo       TEXT    NOT NULL,
    modo              TEXT    NOT NULL,
    n_trials          INTEGER NOT NULL,
    sampler           TEXT,
    funcion_score     TEXT,
    huella_digest     TEXT,
    config_hash       TEXT,
    datos_hash        TEXT,
    git_commit        TEXT,
    git_sucio         INTEGER,
    ts_datos_inicio   TEXT,
    ts_datos_fin      TEXT,
    filas_datos       INTEGER
);

CREATE TABLE IF NOT EXISTS trials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    trial_numero    INTEGER NOT NULL,
    score           REAL,
    sharpe          REAL,
    total_trades    INTEGER,
    parametros_json TEXT,
    metricas_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_activo ON runs(activo);
CREATE INDEX IF NOT EXISTS idx_runs_activo_estrategia ON runs(activo, estrategia_id);
CREATE INDEX IF NOT EXISTS idx_trials_run ON trials(run_id);
"""

# Claves candidatas para extraer el Sharpe de un dict de métricas, en orden de
# preferencia. Se usa la primera presente. El DSR necesita el Sharpe en la misma
# escala que el SR observado del candidato.
# "sharpe_ratio" es el Sharpe POR OPERACIÓN que produce metricas.py (la misma
# escala que consume el PSR/DSR); las demás son alias defensivos.
_CLAVES_SHARPE = ("sharpe_ratio", "sharpe_por_trade", "sharpe")
_CLAVES_TRADES = ("total_trades", "n_trades", "num_trades")


@dataclass(frozen=True)
class ResumenActivo:
    activo: str
    n_runs: int
    n_configuraciones: int


class RegistroExperimentos:
    """Acceso a la base de datos de experimentos.

    Uso típico:

        with RegistroExperimentos(cfg.BD_EXPERIMENTOS) as registro:
            run_id = registro.registrar_run(...)
            registro.registrar_trials(run_id, trials)
            n = registro.contar_configuraciones("BTC")
    """

    def __init__(self, db_path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path))
        self._con.row_factory = sqlite3.Row
        # WAL mejora la concurrencia, pero algunos sistemas de ficheros (montajes
        # de red, FUSE) no soportan su memoria compartida. Si falla, se cae con
        # elegancia al journal por defecto en vez de romper el run.
        try:
            self._con.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            pass
        self._con.execute("PRAGMA foreign_keys=ON;")
        self._con.executescript(_ESQUEMA)
        self._con.commit()

    # -- ciclo de vida ------------------------------------------------------
    def __enter__(self) -> "RegistroExperimentos":
        return self

    def __exit__(self, *exc) -> None:
        self.cerrar()

    def cerrar(self) -> None:
        if self._con is not None:
            self._con.commit()
            self._con.close()
            self._con = None  # type: ignore[assignment]

    # -- escritura ----------------------------------------------------------
    def registrar_run(
        self,
        *,
        activo: str,
        timeframe: str,
        estrategia_id: int,
        estrategia_nombre: str,
        salida_tipo: str,
        modo: str,
        n_trials: int,
        sampler: str | None = None,
        funcion_score: str | None = None,
        huella: Mapping | None = None,
        ts_datos_inicio: str | None = None,
        ts_datos_fin: str | None = None,
        filas_datos: int | None = None,
        creado_utc: str | None = None,
    ) -> int:
        """Inserta un run y devuelve su `run_id`."""
        from datetime import datetime, timezone

        huella = dict(huella or {})
        creado = creado_utc or huella.get("creado_utc") or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        git_sucio = huella.get("git_sucio")
        cur = self._con.execute(
            """
            INSERT INTO runs (
                creado_utc, activo, timeframe, estrategia_id, estrategia_nombre,
                salida_tipo, modo, n_trials, sampler, funcion_score,
                huella_digest, config_hash, datos_hash, git_commit, git_sucio,
                ts_datos_inicio, ts_datos_fin, filas_datos
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                creado,
                str(activo),
                str(timeframe),
                int(estrategia_id),
                str(estrategia_nombre),
                str(salida_tipo),
                str(modo),
                int(n_trials),
                None if sampler is None else str(sampler),
                None if funcion_score is None else str(funcion_score),
                huella.get("digest"),
                huella.get("config_hash"),
                huella.get("datos_hash"),
                huella.get("git_commit"),
                None if git_sucio is None else int(bool(git_sucio)),
                ts_datos_inicio,
                ts_datos_fin,
                None if filas_datos is None else int(filas_datos),
            ),
        )
        self._con.commit()
        return int(cur.lastrowid)

    def registrar_trials(self, run_id: int, trials: Iterable) -> int:
        """Inserta todos los trials de un run. Devuelve cuántos insertó.

        Acepta objetos con atributos (`numero`, `score`, `parametros`,
        `metricas`) o dicts con esas claves. El Sharpe y el nº de trades se
        extraen del dict de métricas de forma defensiva.
        """
        filas = [self._fila_trial(int(run_id), t) for t in trials]
        if not filas:
            return 0
        self._con.executemany(
            """
            INSERT INTO trials (
                run_id, trial_numero, score, sharpe, total_trades,
                parametros_json, metricas_json
            ) VALUES (?,?,?,?,?,?,?)
            """,
            filas,
        )
        self._con.commit()
        return len(filas)

    # -- consultas (alimentan la Fase 3) ------------------------------------
    def contar_configuraciones(self, activo: str, *, estrategia_id: int | None = None) -> int:
        """Nº TOTAL de configuraciones probadas en un activo (el `N` real del DSR)."""
        sql = (
            "SELECT COUNT(*) AS n FROM trials t "
            "JOIN runs r ON r.run_id = t.run_id WHERE r.activo = ?"
        )
        params: list = [str(activo)]
        if estrategia_id is not None:
            sql += " AND r.estrategia_id = ?"
            params.append(int(estrategia_id))
        fila = self._con.execute(sql, params).fetchone()
        return int(fila["n"]) if fila else 0

    def sharpes_configuraciones(
        self, activo: str, *, estrategia_id: int | None = None
    ) -> list[float]:
        """Sharpes de todas las configuraciones probadas (para Var(SR_trials))."""
        sql = (
            "SELECT t.sharpe AS sharpe FROM trials t "
            "JOIN runs r ON r.run_id = t.run_id "
            "WHERE r.activo = ? AND t.sharpe IS NOT NULL"
        )
        params: list = [str(activo)]
        if estrategia_id is not None:
            sql += " AND r.estrategia_id = ?"
            params.append(int(estrategia_id))
        return [float(f["sharpe"]) for f in self._con.execute(sql, params).fetchall()]

    def resumen_por_activo(self) -> list[ResumenActivo]:
        sql = (
            "SELECT r.activo AS activo, COUNT(DISTINCT r.run_id) AS n_runs, "
            "COUNT(t.id) AS n_cfg FROM runs r "
            "LEFT JOIN trials t ON t.run_id = r.run_id GROUP BY r.activo ORDER BY r.activo"
        )
        return [
            ResumenActivo(activo=f["activo"], n_runs=int(f["n_runs"]), n_configuraciones=int(f["n_cfg"]))
            for f in self._con.execute(sql).fetchall()
        ]

    def listar_runs(self, *, activo: str | None = None, limite: int = 50) -> list[sqlite3.Row]:
        sql = "SELECT * FROM runs"
        params: list = []
        if activo is not None:
            sql += " WHERE activo = ?"
            params.append(str(activo))
        sql += " ORDER BY run_id DESC LIMIT ?"
        params.append(int(limite))
        return list(self._con.execute(sql, params).fetchall())

    # -- helpers privados ---------------------------------------------------
    def _fila_trial(self, run_id: int, trial) -> tuple:
        numero = _attr(trial, "numero", _attr(trial, "trial_numero", None))
        score = _attr(trial, "score", None)
        parametros = _attr(trial, "parametros", _attr(trial, "params", {})) or {}
        metricas = _attr(trial, "metricas", {}) or {}
        sharpe = _primero_presente(metricas, _CLAVES_SHARPE)
        total_trades = _primero_presente(metricas, _CLAVES_TRADES)
        return (
            run_id,
            None if numero is None else int(numero),
            None if score is None else float(score),
            None if sharpe is None else float(sharpe),
            None if total_trades is None else int(total_trades),
            _json(parametros),
            _json(metricas),
        )


@contextmanager
def abrir(db_path):
    """Context manager de conveniencia: `with abrir(cfg.BD_EXPERIMENTOS) as r: ...`."""
    registro = RegistroExperimentos(db_path)
    try:
        yield registro
    finally:
        registro.cerrar()


# ---------------------------------------------------------------------------
# Helpers de extracción tolerante (objeto o dict)
# ---------------------------------------------------------------------------

def _attr(obj, nombre: str, defecto):
    if isinstance(obj, Mapping):
        return obj.get(nombre, defecto)
    return getattr(obj, nombre, defecto)


def _primero_presente(metricas: Mapping, claves: tuple[str, ...]):
    if not isinstance(metricas, Mapping):
        return None
    for clave in claves:
        if clave in metricas and metricas[clave] is not None:
            return metricas[clave]
    return None


def _json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)

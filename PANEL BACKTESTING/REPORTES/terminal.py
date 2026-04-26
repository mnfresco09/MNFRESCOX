from __future__ import annotations

from threading import Lock

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class MonitorOptimizacion:
    def __init__(
        self,
        *,
        activo: str,
        timeframe: str,
        estrategia: str,
        salida: str,
        total_trials: int,
        sampler: str,
        n_jobs: int,
    ) -> None:
        self.activo = activo
        self.timeframe = timeframe
        self.estrategia = estrategia
        self.salida = salida
        self.total_trials = int(total_trials)
        self.sampler = sampler
        self.n_jobs = int(n_jobs)
        self.completados = 0
        self.mejor_score: float | None = None
        self.mejor_trial: int | None = None
        self.ultimos: list[dict[str, object]] = []
        self._lock = Lock()
        self._live: Live | None = None
        self._console = Console()

    def __enter__(self) -> "MonitorOptimizacion":
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=6,
            transient=False,
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()

    def registrar(self, *, trial_number: int, score: float, metricas: dict, params: dict) -> None:
        with self._lock:
            self.completados += 1
            if self.mejor_score is None or score > self.mejor_score:
                self.mejor_score = float(score)
                self.mejor_trial = int(trial_number)

            self.ultimos.append(
                {
                    "trial": int(trial_number),
                    "score": float(score),
                    "roi": float(metricas["roi_total"]),
                    "win_rate": float(metricas["win_rate"]),
                    "max_dd": float(metricas["max_drawdown"]),
                    "trades": int(metricas["total_trades"]),
                    "params": params,
                }
            )
            self.ultimos = self.ultimos[-8:]

            if self._live is not None:
                self._live.update(self._render())

    def _render(self) -> Panel:
        tabla = Table(expand=True)
        tabla.add_column("Trial", justify="right")
        tabla.add_column("Score", justify="right")
        tabla.add_column("ROI", justify="right")
        tabla.add_column("Win", justify="right")
        tabla.add_column("DD", justify="right")
        tabla.add_column("Trades", justify="right")
        tabla.add_column("Parametros")

        for item in reversed(self.ultimos):
            tabla.add_row(
                str(item["trial"]),
                f"{item['score']:.6f}",
                f"{item['roi']:.2%}",
                f"{item['win_rate']:.2%}",
                f"{item['max_dd']:.2%}",
                f"{item['trades']}",
                _params_cortos(item["params"]),
            )

        mejor = "sin completar"
        if self.mejor_score is not None:
            mejor = f"trial {self.mejor_trial} | score {self.mejor_score:.6f}"

        titulo = (
            f"{self.activo} {self.timeframe} | {self.estrategia} | {self.salida} | "
            f"{self.sampler} | jobs={self.n_jobs}"
        )
        subtitulo = f"{self.completados}/{self.total_trials} completados | mejor: {mejor}"
        return Panel(tabla, title=titulo, subtitle=subtitulo)


def _params_cortos(params: object) -> str:
    if not isinstance(params, dict):
        return str(params)
    partes = [f"{k}={v}" for k, v in sorted(params.items())]
    texto = ", ".join(partes)
    return texto if len(texto) <= 90 else texto[:87] + "..."

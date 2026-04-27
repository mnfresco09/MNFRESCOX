from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import Lock
from typing import Any

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from REPORTES.formatos import formatear_duracion


@dataclass(frozen=True)
class Theme:
    BLUE: str = "steel_blue3"
    CYAN: str = "cyan"
    GREEN: str = "green3"
    GREEN_SOFT: str = "pale_green3"
    ORANGE: str = "orange3"
    RED: str = "red3"
    GOLD: str = "gold3"
    BEST_BOX: str = "dodger_blue2"
    TEXT: str = "grey78"
    MUTED: str = "grey50"
    DIM: str = "grey35"
    BORDER: str = "grey42"
    WHITE: str = "white"


THEME = Theme()


class Metricas:
    KEYS = {
        "saldo_inicial": ("saldo_inicial", "initial_balance"),
        "saldo_final": ("saldo_final", "balance"),
        "total_trades": ("total_trades", "trades", "n_trades"),
        "win_rate": ("win_rate", "winrate", "wr"),
        "roi_total": ("roi_total", "roi", "return_pct"),
        "expectancy": ("expectancy", "expectancy_per_trade"),
        "trades_por_dia": ("trades_por_dia", "trades_per_day", "tpd"),
        "pnl_total": ("pnl_total", "pnl", "pnl_neto"),
        "max_drawdown": ("max_drawdown", "drawdown", "dd", "mdd"),
        "profit_factor": ("profit_factor", "pf"),
        "sharpe_ratio": ("sharpe_ratio", "sharpe", "sr"),
        "duracion_media_seg": ("duracion_media_seg", "avg_duration_sec"),
        "duracion_media_velas": ("duracion_media_velas", "avg_duration"),
    }

    @classmethod
    def get(cls, data: dict[str, Any], key: str, default: float = 0.0) -> float:
        if not data:
            return default
        for candidate in cls.KEYS.get(key, (key,)):
            value = data.get(candidate)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        return default

    @classmethod
    def get_int(cls, data: dict[str, Any], key: str, default: int = 0) -> int:
        return int(cls.get(data, key, float(default)))


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
        self._pendientes: dict[int, dict[str, Any]] = {}
        self._siguiente_trial = 0
        self._impresos = 0
        self._mejor_impreso_score: float | None = None
        self._mejor_impreso_trial: int | None = None
        self._lock = Lock()
        self._console = Console()

    def __enter__(self) -> "MonitorOptimizacion":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with self._lock:
            for trial_number in sorted(self._pendientes):
                self._imprimir_item(self._pendientes[trial_number])
            self._pendientes.clear()

    def registrar(self, *, trial_number: int, score: float, metricas: dict, params: dict) -> None:
        with self._lock:
            trial_number = int(trial_number)
            self.completados += 1
            self._pendientes[trial_number] = {
                "trial": trial_number,
                "score": float(score),
                "metricas": dict(metricas),
                "params": dict(params),
            }
            self._imprimir_pendientes_en_orden()

    def _imprimir_pendientes_en_orden(self) -> None:
        while self._siguiente_trial in self._pendientes:
            item = self._pendientes.pop(self._siguiente_trial)
            self._imprimir_item(item)
            self._siguiente_trial += 1

    def _imprimir_item(self, item: dict[str, Any]) -> None:
        score = float(item["score"])
        trial = int(item["trial"])
        es_mejor = self._mejor_impreso_score is None or score > self._mejor_impreso_score
        if es_mejor:
            self._mejor_impreso_score = score
            self._mejor_impreso_trial = trial
        self._impresos += 1
        item = {
            **item,
            "is_best": es_mejor,
            "best_trial": self._mejor_impreso_trial,
            "best_score": self._mejor_impreso_score,
            "completados": self._impresos,
        }
        self._console.print(self._render_trial(item))

    def _render_trial(self, item: dict[str, Any]) -> Panel:
        titulo = _titulo(
            activo=self.activo,
            timeframe=self.timeframe,
            estrategia=self.estrategia,
            salida=self.salida,
            sampler=self.sampler,
            n_jobs=self.n_jobs,
            completados=int(item["completados"]),
            total=self.total_trials,
        )
        metricas = item["metricas"]
        params = item["params"]
        score_line = _score_line(
            score=float(item["score"]),
            trial=int(item["trial"]),
            is_best=bool(item["is_best"]),
            best_trial=item["best_trial"],
            best_score=item["best_score"],
        )
        paneles = Columns(
            [
                _panel_performance(metricas),
                _panel_finanzas(metricas),
                _panel_parametros(params, self.salida),
            ],
            equal=False,
            expand=True,
        )
        contenido = Group(
            Align.center(titulo),
            "",
            Align.center(score_line),
            "",
            paneles,
        )
        border = THEME.BEST_BOX if item["is_best"] else THEME.BORDER
        title = "[bold gold3]NEW BEST[/]" if item["is_best"] else f"[bold {THEME.BLUE}]RICH MONITOR[/]"
        return Panel(
            contenido,
            title=title,
            border_style=border,
            box=box.DOUBLE if item["is_best"] else box.ROUNDED,
            padding=(1, 2),
        )


def _titulo(
    *,
    activo: str,
    timeframe: str,
    estrategia: str,
    salida: str,
    sampler: str,
    n_jobs: int,
    completados: int,
    total: int,
) -> Text:
    text = Text()
    text.append(str(activo).upper(), style=f"bold {THEME.BLUE}")
    text.append(f" {str(timeframe).upper()}  |  ", style=THEME.MUTED)
    text.append(str(estrategia).upper(), style=THEME.WHITE)
    text.append("  |  ", style=THEME.MUTED)
    text.append(str(salida).upper(), style=THEME.CYAN)
    text.append("  |  ", style=THEME.MUTED)
    text.append(str(sampler).upper(), style=THEME.TEXT)
    text.append(f"  |  JOBS {n_jobs}  |  ", style=THEME.MUTED)
    text.append(f"{completados}/{total}", style=f"bold {THEME.WHITE}")
    return text


def _score_line(
    *,
    score: float,
    trial: int,
    is_best: bool,
    best_trial: int | None,
    best_score: float | None,
) -> Text:
    color = THEME.GOLD if is_best else THEME.BLUE
    text = Text()
    text.append("TRIAL ", style=THEME.MUTED)
    text.append(str(trial), style=f"bold {THEME.WHITE}")
    text.append("  |  SCORE ", style=THEME.MUTED)
    text.append(_format_score(score), style=f"bold {color}")
    if best_trial is not None and best_score is not None:
        text.append("  |  BEST ", style=THEME.MUTED)
        text.append(f"{_format_score(best_score)} T{best_trial}", style=f"bold {THEME.GOLD}")
    return text


def _panel_performance(metricas: dict[str, Any]) -> Panel:
    grid = _grid(14, 12)
    grid.add_row("WIN RATE", _fmt_pct(Metricas.get(metricas, "win_rate")))
    grid.add_row("EXPECT.", _fmt_pct(Metricas.get(metricas, "expectancy")))
    grid.add_row("PROFIT F", _fmt_num(Metricas.get(metricas, "profit_factor"), 2))
    grid.add_row("SHARPE", _fmt_num(Metricas.get(metricas, "sharpe_ratio"), 2))
    grid.add_row("MAX DD", _fmt_pct(Metricas.get(metricas, "max_drawdown"), invert=True))
    grid.add_row("TRADES/DIA", _fmt_num(Metricas.get(metricas, "trades_por_dia"), 3))
    grid.add_row("TRADES", str(Metricas.get_int(metricas, "total_trades")))
    grid.add_row("DUR MEDIA", _fmt_duracion_media(metricas))
    return Panel(
        grid,
        title=f"[{THEME.MUTED}]PERFORMANCE[/]",
        border_style=THEME.BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _panel_finanzas(metricas: dict[str, Any]) -> Panel:
    roi = Metricas.get(metricas, "roi_total")
    pnl = Metricas.get(metricas, "pnl_total")
    saldo_inicial = Metricas.get(metricas, "saldo_inicial")
    saldo_final = Metricas.get(metricas, "saldo_final")
    color = _roi_color(_pct(roi))
    grid = _grid(12, 15)
    grid.add_row("PNL", f"[{color}]{_money(pnl, signed=True)}[/]")
    grid.add_row("ROI", f"[{color}]{_pct(roi):+.2f}%[/]")
    grid.add_row("", f"[{THEME.DIM}]{'-' * 13}[/]")
    grid.add_row("INICIAL", _money(saldo_inicial))
    grid.add_row("FINAL", _money(saldo_final))
    grid.add_row("DELTA", f"[{color}]{_money(saldo_final - saldo_inicial, signed=True)}[/]")
    return Panel(
        grid,
        title=f"[{THEME.MUTED}]FINANZAS[/]",
        border_style=THEME.BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _panel_parametros(params: dict[str, Any], salida_base: str) -> Panel:
    exit_type = str(params.get("__exit_type", salida_base)).upper()
    grid = _grid(16, 14)
    grid.add_row(f"[bold {THEME.TEXT}]EXIT[/]", f"[bold {THEME.TEXT}]{exit_type}[/]")
    if exit_type == "BARS":
        grid.add_row("SL", f"{_float_param(params, '__exit_sl_pct'):.1f}%")
        grid.add_row("VELAS", str(int(_float_param(params, "__exit_velas"))))
    elif exit_type == "CUSTOM":
        grid.add_row("SL", f"{_float_param(params, '__exit_sl_pct'):.1f}%")
        grid.add_row("CIERRE", "ESTRATEGIA")
    else:
        grid.add_row("SL", f"{_float_param(params, '__exit_sl_pct'):.1f}%")
        grid.add_row("TP", f"{_float_param(params, '__exit_tp_pct'):.1f}%")

    strategy_params = {
        k: v
        for k, v in params.items()
        if not str(k).startswith("__")
        and k
        not in {
            "exit_sl_pct",
            "exit_tp_pct",
            "exit_velas",
        }
    }
    if strategy_params:
        grid.add_row("", f"[{THEME.DIM}]{'-' * 12}[/]")
        for key, value in list(sorted(strategy_params.items()))[:10]:
            grid.add_row(str(key).replace("_", " ").upper()[:14], _fmt_param(value))
    return Panel(
        grid,
        title=f"[{THEME.MUTED}]PARAMETROS[/]",
        border_style=THEME.BORDER,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _grid(label_width: int, value_width: int) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=THEME.MUTED, width=label_width)
    grid.add_column(style=THEME.TEXT, justify="right", width=value_width)
    return grid


def _pct(value: float) -> float:
    return float(value) * 100.0


def _fmt_pct(value: float, *, invert: bool = False) -> str:
    pct = _pct(value)
    color = _dd_color(pct) if invert else _roi_color(pct)
    return f"[{color}]{pct:.2f}%[/]"


def _fmt_num(value: float, decimals: int) -> str:
    if not isfinite(value):
        return "inf"
    return f"{value:.{decimals}f}"


def _fmt_duracion_media(metricas: dict[str, Any]) -> str:
    segundos = Metricas.get(metricas, "duracion_media_seg")
    velas = Metricas.get(metricas, "duracion_media_velas")
    return formatear_duracion(segundos, velas)


def _fmt_param(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def _format_score(score: float) -> str:
    return f"{score:.6f}"


def _money(value: float, *, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}${value:,.2f}"


def _roi_color(roi_pct: float) -> str:
    if roi_pct > 100:
        return THEME.GREEN
    if roi_pct > 0:
        return THEME.GREEN_SOFT
    if roi_pct >= -50:
        return THEME.ORANGE
    return THEME.RED


def _dd_color(dd_pct: float) -> str:
    if dd_pct <= 10:
        return THEME.GREEN_SOFT
    if dd_pct <= 30:
        return THEME.ORANGE
    return THEME.RED


def _float_param(params: dict[str, Any], key: str) -> float:
    try:
        return float(params.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0

"""Analitica avanzada de un trial a partir del replay (trades + equity).

El motor Rust solo calcula las metricas basicas (sharpe, max_dd, win_rate,
profit_factor, etc.). Aqui se derivan las metricas avanzadas que necesitan la
serie de trades y la curva de equity completas.

Metricas calculadas:
  Riesgo/Rentabilidad: Sortino, Calmar, CAGR, Recovery, Exposicion, VaR/CVaR
  Distribucion: Skewness, curtosis, percentiles, payoff
  Rachas: max/avg racha ganadora y perdedora
  Estadisticas de sistema: Z-Score, SQN, R-Expectancy, AHPR, GHPR, Stagnation
  Agrupacion temporal: tabla de retornos mensuales (funcion separada)

Es una capa de solo-lectura sobre ReplayTrial: no toca el motor ni los datos.
Tanto el PDF como el Excel y el HTML la consumen, asi que los numeros son
identicos entre todos los reportes.

Referencias:
  Z-Score:      Wald & Wolfowitz (1940), Annals of Mathematical Statistics.
  SQN:          Van Tharp (2006), Definitive Guide to Position Sizing.
  R-Expectancy: Van Tharp (1998), Trade Your Way to Financial Freedom, Cap 8.
  AHPR/GHPR:    Vince (1992), Mathematics of Money Management.
  PSR:          Bailey & Lopez de Prado (2012), Sharpe Ratio Efficient Frontier.
"""

from __future__ import annotations

import math
from calendar import month_abbr
from datetime import date, datetime, timezone

import numpy as np

from COMUN import estadistica as est


# ---------------------------------------------------------------------------
# API publica principal
# ---------------------------------------------------------------------------

def analitica_avanzada(
    *,
    metricas: dict,
    trades: dict[str, np.ndarray],
    equity_curve: np.ndarray,
    fecha_inicio: date,
    fecha_fin: date,
) -> dict[str, float | int]:
    """Devuelve el dict completo de metricas avanzadas. Tolera 0 trades."""
    pnl    = np.asarray(trades.get("pnl",  np.empty(0)), dtype=np.float64)
    roi    = np.asarray(trades.get("roi",  np.empty(0)), dtype=np.float64)
    dur    = np.asarray(trades.get("duracion_velas", np.empty(0)), dtype=np.int64)
    n      = int(pnl.shape[0])
    eq     = np.asarray(equity_curve, dtype=np.float64)

    dias  = max(1, (fecha_fin - fecha_inicio).days + 1)
    anios = dias / 365.25

    if n == 0 or eq.shape[0] < 2:
        return _vacia()

    saldo_inicial = float(eq[0])
    saldo_final   = float(eq[-1])

    # ── Drawdown ─────────────────────────────────────────────────────────
    picos      = np.maximum.accumulate(eq)
    dd_money   = float(np.max(picos - eq))
    max_dd_frac = float(metricas.get("max_drawdown", 0.0))
    pnl_total  = float(metricas.get("pnl_total", float(np.sum(pnl))))

    # ── Rentabilidad temporal ─────────────────────────────────────────────
    cagr              = est.cagr(saldo_inicial, saldo_final, anios)
    yearly_avg_profit = pnl_total / anios  if anios > 0 else 0.0
    monthly_avg_profit= pnl_total / (dias / 30.4375)
    daily_avg_profit  = pnl_total / dias

    # ── Ratios ajustados por riesgo ───────────────────────────────────────
    calmar           = est.calmar(cagr, max_dd_frac)
    recovery         = pnl_total / dd_money if dd_money > 1e-9 else 0.0
    return_dd_ratio  = pnl_total / dd_money if dd_money > 1e-9 else 0.0
    annual_pct_dd    = (cagr * 100.0) / (max_dd_frac * 100.0) if max_dd_frac > 1e-9 else 0.0

    # ── Sortino (misma escala por-trade que el Sharpe del motor) ─────────
    media_roi   = float(np.mean(roi))
    bajistas    = np.minimum(roi, 0.0)
    desv_bajista= float(math.sqrt(float(np.mean(bajistas * bajistas))))
    sortino     = est.sortino_por_trade(media_roi, desv_bajista)

    # ── Exposicion ────────────────────────────────────────────────────────
    if "ts_entrada" in trades and "ts_salida" in trades:
        dur_us       = (np.asarray(trades["ts_salida"], dtype=np.int64)
                        - np.asarray(trades["ts_entrada"], dtype=np.int64))
        en_mercado_s = float(np.sum(np.clip(dur_us, 0, None))) / 1_000_000.0
    else:
        en_mercado_s = 0.0
    exposure = min(1.0, max(0.0, en_mercado_s / (dias * 86_400.0)))

    # ── Trades por segmento ───────────────────────────────────────────────
    ganadores = pnl[pnl > 0]
    perdedores= pnl[pnl < 0]
    n_wins    = int(ganadores.size)
    n_losses  = int(perdedores.size)
    avg_win   = float(np.mean(ganadores)) if n_wins  else 0.0
    avg_loss  = float(np.mean(perdedores)) if n_losses else 0.0
    gross_profit = float(np.sum(ganadores)) if n_wins  else 0.0
    gross_loss   = float(np.sum(perdedores)) if n_losses else 0.0
    payoff    = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    wins_losses_ratio = n_wins / n_losses if n_losses > 0 else float("inf")

    # ── Duracion media por resultado ──────────────────────────────────────
    avg_bars_wins  = float(np.mean(dur[pnl > 0])) if n_wins   else 0.0
    avg_bars_losses= float(np.mean(dur[pnl < 0])) if n_losses else 0.0

    # ── VaR / CVaR ───────────────────────────────────────────────────────
    var95  = float(np.percentile(pnl, 5))
    cola   = pnl[pnl <= var95]
    cvar95 = float(np.mean(cola)) if cola.size else var95

    # ── Distribucion de retornos ──────────────────────────────────────────
    skew, kurt = _skew_kurtosis(pnl)
    perc = {int(q): float(np.percentile(pnl, q)) for q in (5, 25, 50, 75, 95)}

    # ── Versiones anualizadas y PSR fino ─────────────────────────────────
    sharpe         = float(metricas.get("sharpe_ratio", 0.0))
    tpa            = est.trades_por_anio(n / dias if dias > 0 else 0.0)
    sharpe_anual   = est.sharpe_anualizado(sharpe, tpa)
    sortino_anual  = est.sharpe_anualizado(sortino, tpa)
    psr            = est.probabilistic_sharpe_ratio(sharpe, n,
                         asimetria=skew, curtosis=kurt + 3.0)

    # ── Rachas: max y MEDIA ───────────────────────────────────────────────
    max_win_s,  avg_win_s  = _rachas_stats(pnl, ganadora=True)
    max_loss_s, avg_loss_s = _rachas_stats(pnl, ganadora=False)

    # ── AHPR / GHPR (Vince 1992) ─────────────────────────────────────────
    # AHPR = media aritmetica de (1 + HPR_i) - 1, con HPR = roi por trade.
    # GHPR = media geometrica de HPR: (sf/si)^(1/N) - 1 (exacto si colateral fijo).
    ahpr = float(np.mean(roi)) * 100.0
    ghpr = ((saldo_final / saldo_inicial) ** (1.0 / n) - 1.0) * 100.0 \
           if saldo_inicial > 0 else 0.0

    # ── Z-Score — Wald–Wolfowitz runs test ────────────────────────────────
    # Solo wins/losses binarios (excluye neutros).
    # R  = numero de rachas; E[R] y Var[R] bajo H0 de independencia.
    # Z < 0 → clustering; Z > 0 → alternancia.
    # Z_prob = Phi(|Z|)*100 → confianza estadistica del patron observado.
    z_score, z_prob = _z_score(pnl)

    # ── SQN — Van Tharp ───────────────────────────────────────────────────
    # SQN = sqrt(min(N,100)) * mean(PnL) / std(PnL,ddof=1)
    # El tope en 100 es la convencion original de Van Tharp para comparabilidad.
    # Escala: <1.6 pobre | 1.6-1.9 medio | 2.0-2.4 bueno | 2.5+ excelente | 3+ santo grial
    sqn = _sqn(pnl, n)

    # ── R-Expectancy — Van Tharp ──────────────────────────────────────────
    # R_unit      = |avg_loss|  (riesgo medio como referencia)
    # R_exp       = win_rate * payoff - loss_rate
    # R_exp_score = R_exp * sqrt(N)  (normalizado por frecuencia)
    win_rate  = float(metricas.get("win_rate",
                      n_wins / n if n > 0 else 0.0))
    r_exp, r_score = _r_expectancy(win_rate=win_rate, payoff_ratio=payoff, n=n)

    # ── Stagnation ────────────────────────────────────────────────────────
    # stagnation_days = periodo mas largo (en dias) sin nuevo maximo de equity.
    # stagnation_pct  = fraccion del periodo total pasada por debajo del pico.
    ts_salida = trades.get("ts_salida")
    stag_days, stag_pct = _stagnation(
        eq=eq,
        ts_salida=ts_salida,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        dias=dias,
    )

    return {
        # ── Ya existentes ─────────────────────────────────────────────────
        "sortino_ratio":     _fin(sortino),
        "sharpe_anualizado": _fin(sharpe_anual),
        "sortino_anualizado":_fin(sortino_anual),
        "psr":               _fin(psr),
        "calmar_ratio":      _fin(calmar),
        "cagr":              _fin(cagr),
        "recovery_factor":   _fin(recovery),
        "max_drawdown_money":_fin(dd_money),
        "exposure":          _fin(exposure),
        "payoff_ratio":      _fin(payoff),
        "avg_win":           _fin(avg_win),
        "avg_loss":          _fin(avg_loss),
        "best_trade":        _fin(float(np.max(pnl))),
        "worst_trade":       _fin(float(np.min(pnl))),
        "var95":             _fin(var95),
        "cvar95":            _fin(cvar95),
        "skew":              _fin(skew),
        "kurtosis":          _fin(kurt),
        "max_win_streak":    max_win_s,
        "max_loss_streak":   max_loss_s,
        "percentil_5":       _fin(perc[5]),
        "percentil_25":      _fin(perc[25]),
        "percentil_50":      _fin(perc[50]),
        "percentil_75":      _fin(perc[75]),
        "percentil_95":      _fin(perc[95]),
        # ── Nuevas ────────────────────────────────────────────────────────
        "yearly_avg_profit":  _fin(yearly_avg_profit),
        "monthly_avg_profit": _fin(monthly_avg_profit),
        "daily_avg_profit":   _fin(daily_avg_profit),
        "return_dd_ratio":    _fin(return_dd_ratio),
        "annual_pct_over_maxdd": _fin(annual_pct_dd),
        "gross_profit":       _fin(gross_profit),
        "gross_loss":         _fin(gross_loss),
        "wins_losses_ratio":  _fin(wins_losses_ratio) if math.isfinite(wins_losses_ratio) else 0.0,
        "avg_bars_wins":      _fin(avg_bars_wins),
        "avg_bars_losses":    _fin(avg_bars_losses),
        "avg_consec_wins":    _fin(avg_win_s),
        "avg_consec_losses":  _fin(avg_loss_s),
        "z_score":            _fin(z_score),
        "z_probability":      _fin(z_prob),
        "sqn":                _fin(sqn),
        "r_expectancy":       _fin(r_exp),
        "r_expectancy_score": _fin(r_score),
        "ahpr":               _fin(ahpr),
        "ghpr":               _fin(ghpr),
        "stagnation_days":    _fin(stag_days),
        "stagnation_pct":     _fin(stag_pct),
    }


def tabla_monthly_returns(
    trades: dict[str, np.ndarray],
    *,
    fecha_inicio: date,
    fecha_fin: date,
) -> dict[int, dict[int, float]]:
    """Tabla de retornos mensuales agrupados por fecha de cierre.

    Devuelve {year: {1..12: pnl_sum}}.
    Todos los meses del rango aparecen aunque no haya trades (valor 0.0).
    Se usa la fecha de CIERRE (ts_salida) porque es cuando el PnL es realizado.
    """
    pnl_arr = np.asarray(trades.get("pnl", np.empty(0)), dtype=np.float64)
    ts_arr  = trades.get("ts_salida")

    # Relleno del rango completo con ceros.
    result: dict[int, dict[int, float]] = {}
    y, m = fecha_inicio.year, fecha_inicio.month
    while (y, m) <= (fecha_fin.year, fecha_fin.month):
        result.setdefault(y, {})[m] = 0.0
        m += 1
        if m > 12:
            m, y = 1, y + 1

    if ts_arr is None or pnl_arr.size == 0:
        return result

    ts_us = np.asarray(ts_arr, dtype=np.int64)
    for ts, p in zip(ts_us.tolist(), pnl_arr.tolist()):
        try:
            dt = datetime.fromtimestamp(ts / 1_000_000.0, tz=timezone.utc)
            result.setdefault(dt.year, {})[dt.month] = \
                result.get(dt.year, {}).get(dt.month, 0.0) + p
        except (OSError, ValueError, OverflowError):
            pass

    return result


def distribuciones(
    trades: dict[str, np.ndarray],
    *,
    bins_pnl: int = 16,
    bins_dur: int = 14,
) -> dict:
    """Histogramas y conteos para los graficos de distribucion."""
    pnl     = np.asarray(trades.get("pnl", np.empty(0)), dtype=np.float64)
    dur     = np.asarray(trades.get("duracion_velas", np.empty(0)), dtype=np.float64)
    motivos = np.asarray(trades.get("motivo_salida", np.empty(0)), dtype=np.int64)
    from MOTOR.wrapper import MOTIVOS

    conteo_motivo: dict[str, int] = {}
    for codigo in motivos.tolist():
        nombre = MOTIVOS[int(codigo)] if 0 <= int(codigo) < len(MOTIVOS) else str(codigo)
        conteo_motivo[nombre] = conteo_motivo.get(nombre, 0) + 1

    return {
        "pnl":      _histograma(pnl, bins_pnl),
        "duracion": _histograma(dur, bins_dur),
        "motivo":   conteo_motivo,
    }


# ---------------------------------------------------------------------------
# Formulas privadas
# ---------------------------------------------------------------------------

def _z_score(pnl: np.ndarray) -> tuple[float, float]:
    """Wald–Wolfowitz runs test de independencia serial.

    Excluye trades neutros (pnl == 0).
    Devuelve (Z, Z_prob) donde Z_prob = Phi(|Z|)*100.
    """
    from COMUN.estadistica import phi
    binary = pnl[pnl != 0]
    n = int(binary.size)
    W = int(np.sum(binary > 0))
    L = int(np.sum(binary < 0))
    if W < 2 or L < 2 or n < 4:
        return 0.0, 0.0

    outcomes = (binary > 0).astype(np.int8)
    R = int(1 + np.sum(outcomes[1:] != outcomes[:-1]))

    E_R   = 1.0 + (2.0 * W * L) / n
    num_v = 2.0 * W * L * (2.0 * W * L - n)
    den_v = (n ** 2) * (n - 1)
    if den_v <= 0 or num_v < 0:
        return 0.0, 0.0
    Var_R = num_v / den_v
    if Var_R <= 0:
        return 0.0, 0.0

    z      = (R - E_R) / math.sqrt(Var_R)
    z_prob = phi(abs(z)) * 100.0
    return _fin(z), _fin(z_prob)


def _sqn(pnl: np.ndarray, n: int) -> float:
    """System Quality Number (Van Tharp).

    SQN = sqrt(min(N, 100)) * mean(PnL) / std(PnL, ddof=1)
    """
    if n < 2:
        return 0.0
    std = float(np.std(pnl, ddof=1))
    if std <= 0:
        return 0.0
    return _fin(math.sqrt(min(n, 100)) * float(np.mean(pnl)) / std)


def _r_expectancy(
    *,
    win_rate: float,
    payoff_ratio: float,
    n: int,
) -> tuple[float, float]:
    """R-Expectancy y R-Expectancy Score.

    R_exp       = win_rate * payoff - (1 - win_rate)
    R_exp_score = R_exp * sqrt(N)
    """
    r_exp   = win_rate * payoff_ratio - (1.0 - win_rate)
    r_score = r_exp * math.sqrt(max(1, n))
    return _fin(r_exp), _fin(r_score)


def _stagnation(
    *,
    eq: np.ndarray,
    ts_salida,
    fecha_inicio: date,
    fecha_fin: date,
    dias: int,
) -> tuple[float, float]:
    """Stagnation: periodo mas largo bajo el pico (dias) y % tiempo en drawdown."""
    if eq.shape[0] < 2 or ts_salida is None:
        return 0.0, 0.0

    ts = np.asarray(ts_salida, dtype=np.int64)
    n_eq, n_ts = eq.shape[0], ts.shape[0]

    fi_us = int(datetime(fecha_inicio.year, fecha_inicio.month,
                         fecha_inicio.day, tzinfo=timezone.utc).timestamp() * 1_000_000)
    if n_eq == n_ts + 1:
        ts_eq = np.concatenate([[fi_us], ts])
    elif n_eq == n_ts:
        ts_eq = ts
    else:
        return 0.0, 0.0

    picos  = np.maximum.accumulate(eq)
    en_dd  = eq < picos

    max_dd_days = 0.0
    total_dd_us = 0
    ini_dd: int | None = None

    for i in range(len(en_dd)):
        if en_dd[i]:
            if ini_dd is None:
                ini_dd = int(ts_eq[i - 1]) if i > 0 else int(ts_eq[0])
            if i > 0:
                total_dd_us += int(ts_eq[i]) - int(ts_eq[i - 1])
        else:
            if ini_dd is not None:
                dur = (int(ts_eq[i]) - ini_dd) / (1_000_000.0 * 86_400.0)
                max_dd_days = max(max_dd_days, dur)
                ini_dd = None

    if ini_dd is not None:
        ff_us = int(datetime(fecha_fin.year, fecha_fin.month,
                             fecha_fin.day, tzinfo=timezone.utc).timestamp() * 1_000_000)
        dur = (ff_us - ini_dd) / (1_000_000.0 * 86_400.0)
        max_dd_days = max(max_dd_days, dur)

    stag_pct = total_dd_us / (dias * 86_400.0 * 1_000_000.0)
    return _fin(max_dd_days), _fin(min(1.0, stag_pct) * 100.0)


def _rachas_stats(pnl: np.ndarray, *, ganadora: bool) -> tuple[int, float]:
    """Racha maxima y racha media de wins (ganadora=True) o losses (False)."""
    rachas: list[int] = []
    actual = 0
    for v in pnl.tolist():
        es = (v > 0) if ganadora else (v < 0)
        if es:
            actual += 1
        else:
            if actual > 0:
                rachas.append(actual)
            actual = 0
    if actual > 0:
        rachas.append(actual)
    if not rachas:
        return 0, 0.0
    return int(max(rachas)), float(np.mean(rachas))


def _histograma(data: np.ndarray, bins: int) -> dict:
    if data.size == 0:
        return {"edges": [], "counts": []}
    lo = float(np.min(data))
    hi = float(np.max(data))
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(data, bins=bins, range=(lo, hi))
    return {
        "edges":  [round(float(e), 2) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
    }


def _skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    n = x.size
    if n < 3:
        return 0.0, 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 0:
        return 0.0, 0.0
    z    = (x - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    return skew, kurt


def _racha(pnl: np.ndarray, *, ganadora: bool) -> int:
    """Compatibilidad: devuelve solo el maximo (usada por codigo legacy)."""
    return _rachas_stats(pnl, ganadora=ganadora)[0]


def _fin(v: float) -> float:
    f = float(v)
    return f if math.isfinite(f) else 0.0


def _vacia() -> dict[str, float | int]:
    claves_float = [
        "sortino_ratio", "sharpe_anualizado", "sortino_anualizado", "psr",
        "calmar_ratio", "cagr", "recovery_factor", "max_drawdown_money",
        "exposure", "payoff_ratio", "avg_win", "avg_loss",
        "best_trade", "worst_trade", "var95", "cvar95", "skew", "kurtosis",
        "percentil_5", "percentil_25", "percentil_50", "percentil_75", "percentil_95",
        "yearly_avg_profit", "monthly_avg_profit", "daily_avg_profit",
        "return_dd_ratio", "annual_pct_over_maxdd",
        "gross_profit", "gross_loss", "wins_losses_ratio",
        "avg_bars_wins", "avg_bars_losses",
        "avg_consec_wins", "avg_consec_losses",
        "z_score", "z_probability", "sqn",
        "r_expectancy", "r_expectancy_score",
        "ahpr", "ghpr", "stagnation_days", "stagnation_pct",
    ]
    salida: dict[str, float | int] = {k: 0.0 for k in claves_float}
    salida["max_win_streak"]  = 0
    salida["max_loss_streak"] = 0
    return salida

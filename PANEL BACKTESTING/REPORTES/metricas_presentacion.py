"""Fuente unica de presentacion de metricas: etiqueta, formato y valoracion.

Tanto el report HTML como el Excel construyen su panel de "estadistica
avanzada" desde aqui, asi que las etiquetas, los formatos y el juicio
(BUENO / REGULAR / MALO / INFO) son siempre identicos entre ambos. Si se
quiere afinar un umbral, se cambia en un unico sitio.

`construir_analitica(metricas, avanzada)` recibe el dict de metricas del motor
y el de `REPORTES.analitica.analitica_avanzada`, y devuelve las secciones ya
formateadas mas un veredicto global legible de un vistazo.
"""

from __future__ import annotations


# ── formato ────────────────────────────────────────────────────────────────

def fmt_pct(v: float, dec: int = 2) -> str:
    v = float(v)
    return f"{'+' if v >= 0 else ''}{v * 100:.{dec}f}%"


def fmt_pct_u(v: float, dec: int = 1) -> str:
    return f"{float(v) * 100:.{dec}f}%"


def fmt_money(v: float, dec: int = 2) -> str:
    v = float(v)
    return f"{'-' if v < 0 else ''}${abs(v):,.{dec}f}"


def fmt_money0(v: float) -> str:
    v = float(v)
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def fmt_ratio(v: float, dec: int = 2) -> str:
    return f"{float(v):.{dec}f}"


# ── valoradores (umbrales) ───────────────────────────────────────────────────

def _mayor(v, bueno, regular):  # mas alto = mejor
    v = float(v)
    return "good" if v >= bueno else ("ok" if v >= regular else "bad")


def _menor(v, bueno, regular):  # mas bajo = mejor
    v = float(v)
    return "good" if v <= bueno else ("ok" if v <= regular else "bad")


def _signo(v, *, bueno_si_positivo=True):
    v = float(v)
    pos = v > 0
    return "good" if (pos == bueno_si_positivo) else "bad"


# ── construccion de secciones + veredicto ────────────────────────────────────

def construir_analitica(metricas: dict, avanzada: dict) -> dict:
    """Devuelve {'secciones': [...], 'veredicto': {...}} listo para render."""
    m = {**metricas, **avanzada}

    def fila(clave, label, valor, nivel):
        return {"clave": clave, "label": label, "valor": valor, "nivel": nivel}

    psr = float(m.get("psr", 0.0))
    riesgo = [
        fila("psr", "PSR (P[Sharpe>0])", fmt_pct_u(psr), _mayor(psr, 0.95, 0.80)),
        fila("sharpe_anualizado", "Sharpe anual.", fmt_ratio(m.get("sharpe_anualizado", 0.0)), _mayor(m.get("sharpe_anualizado", 0.0), 2, 1)),
        fila("sortino_anualizado", "Sortino anual.", fmt_ratio(m.get("sortino_anualizado", 0.0)), _mayor(m.get("sortino_anualizado", 0.0), 2.5, 1.5)),
        fila("calmar_ratio", "Calmar", fmt_ratio(m["calmar_ratio"]), _mayor(m["calmar_ratio"], 3, 1)),
        fila("cagr", "CAGR", fmt_pct(m["cagr"]), "good" if m["cagr"] >= 0.15 else ("ok" if m["cagr"] > 0 else "bad")),
        fila("max_drawdown", "Max drawdown", "-" + fmt_pct_u(m["max_drawdown"], 2), _menor(m["max_drawdown"], 0.10, 0.20)),
        fila("recovery_factor", "Recovery factor", fmt_ratio(m["recovery_factor"]), _mayor(m["recovery_factor"], 3, 1)),
        fila("exposure", "Exposicion", fmt_pct_u(m["exposure"]), "info"),
    ]
    robustez = [
        fila("var95", "VaR 95%", fmt_money(m["var95"]), "info"),
        fila("cvar95", "CVaR 95%", fmt_money(m["cvar95"]), "info"),
        fila("skew", "Skew", fmt_ratio(m["skew"]), "good" if m["skew"] > 0.1 else ("ok" if m["skew"] > -0.3 else "bad")),
        fila("kurtosis", "Kurtosis", fmt_ratio(m["kurtosis"]), _menor(abs(float(m["kurtosis"])), 1, 3)),
        fila("max_win_streak", "Racha max. ganadora", str(int(m["max_win_streak"])), "info"),
        fila("max_loss_streak", "Racha max. perdedora", str(int(m["max_loss_streak"])), _menor(m["max_loss_streak"], 5, 9)),
        fila("percentil_50", "Mediana PnL", fmt_money(m["percentil_50"]), _signo(m["percentil_50"])),
    ]
    rendimiento = [
        fila("profit_factor", "Profit factor", fmt_ratio(m["profit_factor"]), _mayor(m["profit_factor"], 1.5, 1.1)),
        fila("payoff_ratio", "Payoff ratio", fmt_ratio(m["payoff_ratio"]), _mayor(m["payoff_ratio"], 2, 1)),
        fila("win_rate", "Win rate", fmt_pct_u(m["win_rate"]), _mayor(m["win_rate"], 0.5, 0.4)),
        fila("expectancy", "Expectancy", fmt_pct(m["expectancy"]), _signo(m["expectancy"])),
        fila("avg_winloss", "Avg win / loss", f'{fmt_money(m["avg_win"])} / {fmt_money(m["avg_loss"])}', "info"),
        fila("mejorpeor", "Mejor / peor", f'{fmt_money0(m["best_trade"])} / {fmt_money0(m["worst_trade"])}', "info"),
        fila("p5p95", "P5 / P95", f'{fmt_money0(m["percentil_5"])} / {fmt_money0(m["percentil_95"])}', "info"),
    ]
    secciones = [
        {"titulo": "RIESGO", "filas": riesgo},
        {"titulo": "ROBUSTEZ", "filas": robustez},
        {"titulo": "RENDIMIENTO", "filas": rendimiento},
    ]
    return {"secciones": secciones, "veredicto": _veredicto(m, secciones)}


# Suelo de operaciones por debajo del cual el resultado no es fiable, aunque el
# PSR salga alto por casualidad. El PSR ya penaliza la muestra pequeña; este es
# un guardarraíl mínimo adicional.
_MIN_TRADES_FIABLE = 10
# Umbral de PSR para considerar el edge estadísticamente creíble.
_PSR_FIABLE = 0.80


def _clip01(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _veredicto(m: dict, secciones: list[dict]) -> dict:
    """Veredicto profesional: fiabilidad (PSR) × calidad (riesgo/retorno).

    No es un recuento de métricas (muchas están correlacionadas y eso infla el
    juicio). Se separan dos ejes:

      - Fiabilidad = PSR: ¿el edge es real o ruido? Es la puerta de entrada.
      - Calidad    = combinación ponderada de las métricas decisivas y menos
                     redundantes: Sharpe anualizado, Calmar, control de drawdown
                     y profit factor.

    Nota global = Fiabilidad × Calidad. Una estrategia debe ser a la vez creíble
    y rentable: retornos altos con PSR bajo (sobreajuste) → nota baja.
    """
    import math

    psr = _clip01(m.get("psr", 0.0))
    n = int(m.get("total_trades", 0))
    sharpe_anual = float(m.get("sharpe_anualizado", 0.0) or 0.0)
    calmar = float(m.get("calmar_ratio", m.get("calmar", 0.0)) or 0.0)
    max_dd = float(m.get("max_drawdown", 1.0) or 0.0)
    pf = m.get("profit_factor", 0.0)
    pf = float(pf) if (pf is not None and math.isfinite(float(pf))) else 99.0

    # Calidad ajustada al riesgo (cada término normalizado a 0..1).
    q_sharpe = _clip01(sharpe_anual / 2.0)        # Sharpe anual 2.0 = excelente
    q_calmar = _clip01(calmar / 3.0)              # Calmar 3.0 = excelente
    q_dd = _clip01(1.0 - max_dd / 0.20)           # DD 0%→1, 20%+→0
    q_pf = _clip01((pf - 1.0) / 0.5)              # PF 1.0→0, 1.5+→1
    calidad = 0.40 * q_sharpe + 0.25 * q_calmar + 0.20 * q_dd + 0.15 * q_pf

    confianza = psr
    nota = confianza * calidad

    fiable = (psr >= _PSR_FIABLE) and (n >= _MIN_TRADES_FIABLE)
    if not fiable:
        nivel, badge = "bad", "NO FIABLE"
    elif nota >= 0.55:
        nivel, badge = "good", "SOLIDA"
    elif nota >= 0.30:
        nivel, badge = "ok", "ACEPTABLE"
    else:
        nivel, badge = "bad", "DEBIL"

    # Puntos fuertes / a vigilar (a partir de la valoración por métrica).
    valoradas = [f for s in secciones for f in s["filas"] if f["nivel"] != "info"]
    good = sum(1 for f in valoradas if f["nivel"] == "good")
    total = len(valoradas) or 1
    prioridad = [
        "PSR (P[Sharpe>0])", "Sharpe anual.", "Calmar", "Max drawdown",
        "Sortino anual.", "Profit factor", "CAGR", "Win rate", "Payoff ratio",
        "Recovery factor", "Skew", "Kurtosis", "Mediana PnL",
    ]

    def elegir(nv):
        etiquetas = {f["label"] for f in valoradas if f["nivel"] == nv}
        return [p for p in prioridad if p in etiquetas][:3]

    fuertes = elegir("good")
    debiles = elegir("bad")
    if not fiable:
        cabecera = (
            f"Edge poco fiable: PSR {psr * 100:.0f}% "
            f"({'muestra insuficiente' if n < _MIN_TRADES_FIABLE else 'no significativo'}). "
        )
    else:
        cabecera = f"Fiabilidad (PSR) {psr * 100:.0f}% · calidad {calidad * 100:.0f}%. "
    linea = (
        cabecera
        + f"Fuertes: {', '.join(fuertes) if fuertes else '—'}. "
        + f"A vigilar: {', '.join(debiles) if debiles else 'ninguno relevante'}."
    )
    return {
        "nivel": nivel,
        "badge": badge,
        "favorables": good,
        "total": total,
        "score": round(nota if fiable else 0.0, 3),
        "confianza": round(confianza, 3),
        "calidad": round(calidad, 3),
        "fuertes": fuertes,
        "debiles": debiles,
        "linea": linea,
    }

"""Veredicto con umbrales fijados A PRIORI (Fases 3 y 4 del protocolo).

Estos umbrales son la segunda mitad de la disciplina innegociable: se fijan
**antes** de ver el resultado de una estrategia concreta. Si decides el umbral
después de ver el número, siempre encontrarás una justificación para aprobar lo
que querías aprobar. Por eso viven aquí, en código, con los valores por defecto
del documento — se calibran en abstracto, nunca a posteriori.

El veredicto combina cada métrica en un color (🟢 verde / 🟡 ámbar / 🔴 rojo) y
el resultado global es **el peor color de todos los criterios evaluados**: una
sola luz roja mata la estrategia. Las métricas no aportadas (None) no se
evalúan, para poder emitir veredictos parciales a medida que avanza el protocolo.

Nota: el criterio de "supervivencia a 2× costes" del documento NO se incluye
aquí porque depende del modelo de costes de ejecución (funding/liquidación/
slippage), que queda fuera de este alcance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VERDE = "verde"
AMBAR = "ambar"
ROJO = "rojo"
_ORDEN = {VERDE: 0, AMBAR: 1, ROJO: 2}


@dataclass(frozen=True)
class Umbrales:
    """Umbrales por defecto del documento (Parte IV). Calíbralos en abstracto."""

    dsr_verde: float = 0.95
    dsr_ambar: float = 0.90
    pbo_verde: float = 0.20
    pbo_ambar: float = 0.50
    ratio_oos_is_verde: float = 0.70
    ratio_oos_is_ambar: float = 0.50
    n_trades_verde: int = 100
    n_trades_ambar: int = 30
    wfa_verde: float = 0.60
    wfa_ambar: float = 0.50
    # Tolerancia relativa para "≈ capital inicial" en el bootstrap p5.
    holdout_ratio_verde: float = 0.70


@dataclass(frozen=True)
class Criterio:
    nombre: str
    valor: float
    color: str
    detalle: str


@dataclass(frozen=True)
class Veredicto:
    color: str
    criterios: list[Criterio] = field(default_factory=list)

    @property
    def aprobada(self) -> bool:
        return self.color == VERDE

    def motivos(self, color: str) -> list[str]:
        return [f"{c.nombre}: {c.detalle}" for c in self.criterios if c.color == color]

    def resumen(self) -> str:
        rojos = self.motivos(ROJO)
        ambar = self.motivos(AMBAR)
        partes = [f"VEREDICTO: {self.color.upper()}"]
        if rojos:
            partes.append("rojo → " + "; ".join(rojos))
        if ambar:
            partes.append("ámbar → " + "; ".join(ambar))
        return " | ".join(partes)


def evaluar_veredicto(
    *,
    dsr: float | None = None,
    pbo: float | None = None,
    ratio_oos_is: float | None = None,
    p25_sharpe_oos: float | None = None,
    mediana_sharpe_oos: float | None = None,
    n_trades: int | None = None,
    wfa_efficiency: float | None = None,
    bootstrap_p5_equity: float | None = None,
    capital_inicial: float | None = None,
    holdout_metrica: float | None = None,
    holdout_referencia: float | None = None,
    umbrales: Umbrales | None = None,
) -> Veredicto:
    """Combina las métricas disponibles en un veredicto 🟢/🟡/🔴.

    El color global es el peor de todos los criterios evaluados. Las métricas
    None se omiten (veredicto parcial).
    """
    u = umbrales or Umbrales()
    criterios: list[Criterio] = []

    if dsr is not None:
        criterios.append(_mayor_mejor("DSR", dsr, u.dsr_verde, u.dsr_ambar))
    if pbo is not None:
        criterios.append(_menor_mejor("PBO", pbo, u.pbo_verde, u.pbo_ambar))
    if ratio_oos_is is not None:
        criterios.append(
            _mayor_mejor("Sharpe OOS/IS", ratio_oos_is, u.ratio_oos_is_verde, u.ratio_oos_is_ambar)
        )
    if n_trades is not None:
        criterios.append(
            _mayor_mejor("Nº trades", float(n_trades), float(u.n_trades_verde), float(u.n_trades_ambar))
        )
    if wfa_efficiency is not None:
        criterios.append(_mayor_mejor("WFA efficiency", wfa_efficiency, u.wfa_verde, u.wfa_ambar))
    if p25_sharpe_oos is not None or mediana_sharpe_oos is not None:
        criterios.append(_distribucion_oos(p25_sharpe_oos, mediana_sharpe_oos))
    if bootstrap_p5_equity is not None and capital_inicial is not None:
        criterios.append(_bootstrap(bootstrap_p5_equity, capital_inicial))
    if holdout_metrica is not None:
        criterios.append(_holdout(holdout_metrica, holdout_referencia, u.holdout_ratio_verde))

    color = _peor([c.color for c in criterios]) if criterios else AMBAR
    return Veredicto(color=color, criterios=criterios)


# ---------------------------------------------------------------------------
# Criterios individuales
# ---------------------------------------------------------------------------

def _mayor_mejor(nombre: str, valor: float, verde: float, ambar: float) -> Criterio:
    """Métrica donde más alto es mejor (DSR, ratio, nº trades, WFA)."""
    v = float(valor)
    if v >= verde:
        color, detalle = VERDE, f"{v:.4g} ≥ {verde:g}"
    elif v >= ambar:
        color, detalle = AMBAR, f"{ambar:g} ≤ {v:.4g} < {verde:g}"
    else:
        color, detalle = ROJO, f"{v:.4g} < {ambar:g}"
    return Criterio(nombre, v, color, detalle)


def _menor_mejor(nombre: str, valor: float, verde: float, ambar: float) -> Criterio:
    """Métrica donde más bajo es mejor (PBO)."""
    v = float(valor)
    if v < verde:
        color, detalle = VERDE, f"{v:.4g} < {verde:g}"
    elif v <= ambar:
        color, detalle = AMBAR, f"{verde:g} ≤ {v:.4g} ≤ {ambar:g}"
    else:
        color, detalle = ROJO, f"{v:.4g} > {ambar:g}"
    return Criterio(nombre, v, color, detalle)


def _distribucion_oos(p25: float | None, mediana: float | None) -> Criterio:
    """Distribución de Sharpe OOS (CPCV): p25>0 verde; mediana>0,p25<0 ámbar; mediana≤0 rojo."""
    p25_v = float(p25) if p25 is not None else float("nan")
    med_v = float(mediana) if mediana is not None else float("nan")
    if p25 is not None and p25_v > 0.0:
        return Criterio("Distribución Sharpe OOS", p25_v, VERDE, f"p25={p25_v:.4g} > 0")
    if mediana is not None and med_v > 0.0:
        return Criterio("Distribución Sharpe OOS", med_v, AMBAR, f"mediana={med_v:.4g} > 0, p25 ≤ 0")
    valor = med_v if mediana is not None else p25_v
    return Criterio("Distribución Sharpe OOS", valor, ROJO, "mediana ≤ 0")


def _bootstrap(p5_equity: float, capital: float) -> Criterio:
    p5 = float(p5_equity)
    cap = float(capital)
    if p5 > cap:
        return Criterio("Bootstrap p5 equity", p5, VERDE, f"{p5:.6g} > capital {cap:g}")
    if p5 >= cap * 0.999:  # ≈ capital inicial
        return Criterio("Bootstrap p5 equity", p5, AMBAR, f"{p5:.6g} ≈ capital {cap:g}")
    return Criterio("Bootstrap p5 equity", p5, ROJO, f"{p5:.6g} < capital {cap:g}")


def _holdout(metrica: float, referencia: float | None, ratio_verde: float) -> Criterio:
    """Holdout: coherente con OOS verde; degrada pero positivo ámbar; colapsa/negativo rojo."""
    m = float(metrica)
    if m <= 0.0:
        return Criterio("Holdout bloqueado", m, ROJO, f"{m:.4g} ≤ 0 (colapsa)")
    if referencia is not None and float(referencia) > 0.0:
        ratio = m / float(referencia)
        if ratio >= ratio_verde:
            return Criterio("Holdout bloqueado", m, VERDE, f"coherente con OOS (ratio={ratio:.2f})")
        return Criterio("Holdout bloqueado", m, AMBAR, f"degrada pero positivo (ratio={ratio:.2f})")
    return Criterio("Holdout bloqueado", m, AMBAR, f"{m:.4g} > 0 (sin referencia OOS)")


def _peor(colores: list[str]) -> str:
    return max(colores, key=lambda c: _ORDEN[c])

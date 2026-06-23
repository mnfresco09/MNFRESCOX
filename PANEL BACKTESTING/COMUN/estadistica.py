"""Fórmulas estadísticas del sistema: fuente única y documentada.

Aquí viven las definiciones "de libro" de las métricas de riesgo/rentabilidad
ajustada, para que tanto la función de score (`puntuacion.py`), como las
métricas del run (`metricas.py`) y la analítica de reportes (`analitica.py`)
usen exactamente la misma matemática.

Convención de Sharpe interna del motor
--------------------------------------
El motor calcula el Sharpe **por operación** (media / desviación típica muestral
de los retornos por trade, sin tasa libre de riesgo). Es un Sharpe por
observación; para anualizarlo se multiplica por la raíz del número de
operaciones por año (asumiendo retornos por trade aproximadamente
independientes).

Probabilistic Sharpe Ratio (PSR)
--------------------------------
Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier".
PSR estima la probabilidad de que el Sharpe real supere un Sharpe de
referencia, dado el Sharpe observado, el tamaño de muestra y la forma de la
distribución (asimetría y curtosis). Penaliza de forma estadística las muestras
pequeñas y las colas gruesas, por eso es el objetivo recomendado: premia
estrategias creíbles, no casualidades con pocas operaciones.
"""

from __future__ import annotations

from math import erf, isfinite, sqrt

DIAS_POR_ANIO = 365.25


def phi(x: float) -> float:
    """Función de distribución acumulada de la normal estándar N(0,1)."""
    if not isfinite(x):
        return 1.0 if x > 0 else 0.0
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def trades_por_anio(trades_por_dia: float, dias_anio: float = DIAS_POR_ANIO) -> float:
    return max(0.0, float(trades_por_dia)) * float(dias_anio)


def sharpe_anualizado(sharpe_por_trade: float, trades_por_anio_: float) -> float:
    """Anualiza el Sharpe por operación: SR_anual = SR_trade * sqrt(N_trades_año).

    Asume retornos por trade aproximadamente i.i.d. (supuesto estándar de la
    anualización del Sharpe). Si la frecuencia es 0, devuelve 0.
    """
    sr = float(sharpe_por_trade)
    if not isfinite(sr) or trades_por_anio_ <= 0:
        return 0.0
    return sr * sqrt(float(trades_por_anio_))


def probabilistic_sharpe_ratio(
    sharpe_por_trade: float,
    n_trades: int,
    *,
    sharpe_referencia: float = 0.0,
    asimetria: float = 0.0,
    curtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe Ratio (PSR), en [0, 1].

    Probabilidad de que el Sharpe real supere `sharpe_referencia`, dado el
    Sharpe observado por operación, el número de operaciones y la forma de la
    distribución de retornos. `curtosis` es la curtosis NO en exceso (3 = normal);
    `asimetria` es la skewness. Con los valores por defecto (skew=0, curtosis=3)
    se obtiene la aproximación normal: PSR = Phi( SR * sqrt(n-1) / sqrt(1 + SR^2/2) ).
    """
    sr = float(sharpe_por_trade)
    n = int(n_trades)
    if n < 2 or not isfinite(sr):
        return 0.0
    denom = 1.0 - float(asimetria) * sr + ((float(curtosis) - 1.0) / 4.0) * sr * sr
    if denom <= 0.0 or not isfinite(denom):
        return 0.0
    z = (sr - float(sharpe_referencia)) * sqrt(n - 1.0) / sqrt(denom)
    return phi(z)


def cagr(saldo_inicial: float, saldo_final: float, anios: float) -> float:
    """Tasa de crecimiento anual compuesto (Compound Annual Growth Rate)."""
    si, sf, a = float(saldo_inicial), float(saldo_final), float(anios)
    if si <= 0.0 or sf <= 0.0 or a <= 0.0:
        return 0.0
    valor = (sf / si) ** (1.0 / a) - 1.0
    return valor if isfinite(valor) else 0.0


def calmar(cagr_valor: float, max_drawdown: float, *, suelo_dd: float = 1e-6) -> float:
    """Ratio de Calmar = CAGR / Max Drawdown (drawdown como fracción positiva).

    Si no hubo drawdown se usa un suelo mínimo para evitar dividir por cero
    (un Calmar enorme pero finito), de modo que el optimizador pueda ordenarlo.
    """
    dd = max(float(max_drawdown), 0.0)
    denom = max(dd, float(suelo_dd))
    valor = float(cagr_valor) / denom
    return valor if isfinite(valor) else 0.0


def sortino_por_trade(media_retorno: float, desviacion_bajista: float) -> float:
    """Sortino por operación = media de retornos / desviación bajista (MAR=0)."""
    if desviacion_bajista <= 0.0 or not isfinite(desviacion_bajista):
        return 0.0
    return float(media_retorno) / float(desviacion_bajista)


def anios_desde_dias(dias: float) -> float:
    return max(0.0, float(dias)) / DIAS_POR_ANIO

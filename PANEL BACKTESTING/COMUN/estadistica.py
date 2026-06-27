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

from math import e as EULER_E
from math import erf, isfinite, log, sqrt
from statistics import NormalDist

DIAS_POR_ANIO = 365.25

# Constante de Euler-Mascheroni, usada en la corrección por testing múltiple
# (Deflated Sharpe Ratio y Minimum Backtest Length).
GAMMA_EULER = 0.5772156649015329

_NORMAL = NormalDist()


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


def phi_inv(p: float) -> float:
    """Inversa de la CDF normal estándar (función cuantil Z⁻¹)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    return _NORMAL.inv_cdf(p)


def maximo_sharpe_esperado_estandarizado(n_configuraciones: int) -> float:
    """Esperanza del MÁXIMO de N Sharpes estandarizados bajo la hipótesis nula.

    Es el término entre corchetes del Deflated Sharpe Ratio:

        E_N = (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e))

    Mide cuánto Sharpe (en unidades de su propia desviación entre trials) cabe
    esperar SOLO por haber elegido el mejor de `N` pruebas sobre ruido puro.
    Para N <= 1 no hay selección múltiple y devuelve 0.
    """
    n = int(n_configuraciones)
    if n <= 1:
        return 0.0
    termino_1 = (1.0 - GAMMA_EULER) * phi_inv(1.0 - 1.0 / n)
    termino_2 = GAMMA_EULER * phi_inv(1.0 - 1.0 / (n * EULER_E))
    return termino_1 + termino_2


def sharpe_referencia_deflactado(varianza_sharpe_trials: float, n_configuraciones: int) -> float:
    """Sharpe de referencia (SR_0) contra el que se compara el candidato en el DSR.

        SR_0 = sqrt(Var(SR_trials)) · E_N

    `varianza_sharpe_trials` es la varianza de los Sharpes POR OPERACIÓN entre
    todas las configuraciones probadas (misma escala que el Sharpe observado).
    """
    var = float(varianza_sharpe_trials)
    if var < 0.0 or not isfinite(var):
        return 0.0
    return sqrt(var) * maximo_sharpe_esperado_estandarizado(n_configuraciones)


def deflated_sharpe_ratio(
    sharpe_por_trade: float,
    n_trades: int,
    *,
    n_configuraciones: int,
    varianza_sharpe_trials: float,
    asimetria: float = 0.0,
    curtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014), en [0, 1].

    Es EXACTAMENTE el PSR de este módulo pero con el Sharpe de referencia
    deflactado por testing múltiple en vez de 0: se reaprovecha
    `probabilistic_sharpe_ratio` alimentándole `sharpe_referencia = SR_0`.

    Responde a "¿qué probabilidad hay de que este Sharpe sea real, teniendo en
    cuenta que es el mejor de `n_configuraciones` pruebas?". El `n_configuraciones`
    correcto es el TOTAL de configuraciones probadas en el activo a lo largo de la
    investigación (lo aporta el registro de experimentos de la Fase 0), no los
    trials de un único run.
    """
    sr_0 = sharpe_referencia_deflactado(varianza_sharpe_trials, n_configuraciones)
    return probabilistic_sharpe_ratio(
        sharpe_por_trade,
        n_trades,
        sharpe_referencia=sr_0,
        asimetria=asimetria,
        curtosis=curtosis,
    )


def minimum_backtest_length(n_configuraciones: int, sharpe_anual_objetivo: float) -> float:
    """Longitud mínima de backtest (en AÑOS) para que un Sharpe sea creíble.

        MinBTL ≈ ( E_N / SR_anual_objetivo )²

    Dado que se prueban `N` configuraciones, el máximo Sharpe esperable sobre
    ruido crece con `E_N` (≈ sqrt(2·ln N)). Para que un Sharpe anual objetivo
    supere ese techo de ruido se necesitan al menos MinBTL años de datos. Es un
    chequeo de cordura PREVIO: si tu backtest es más corto que esto, el resultado
    no tiene valor estadístico por bonito que sea.
    """
    sr = float(sharpe_anual_objetivo)
    if sr <= 0.0 or not isfinite(sr):
        return float("inf")
    e_n = maximo_sharpe_esperado_estandarizado(n_configuraciones)
    return (e_n / sr) ** 2


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

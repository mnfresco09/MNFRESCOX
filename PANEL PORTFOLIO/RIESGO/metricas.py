"""Métricas de cartera a partir de una serie de retornos diarios.

VaR y CVaR son HISTÓRICOS (no paramétricos): se leen directamente de la
distribución empírica de retornos, sin suponer normalidad. Drawdown, Sharpe,
Sortino y Calmar se calculan sobre la curva de equity realizada.

Convención de signo: var y cvar se devuelven como retornos NEGATIVOS (la
pérdida en la cola), p. ej. -0.03 = 3% de caída diaria en el nivel de cola.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.modelos import MetricasCartera


def equity(retornos: pd.Series) -> pd.Series:
    """Curva de capital (base 1) a partir de retornos simples diarios."""
    return (1.0 + retornos).cumprod()


def var_cvar_historico(retornos: pd.Series, nivel_confianza: float) -> tuple[float, float]:
    alpha = 1.0 - nivel_confianza
    if retornos.empty:
        return 0.0, 0.0
    var = float(np.quantile(retornos.to_numpy(), alpha))
    cola = retornos[retornos <= var]
    cvar = float(cola.mean()) if not cola.empty else var
    return var, cvar


def drawdown(curva_equity: pd.Series) -> tuple[float, int, pd.Timestamp | None]:
    """Devuelve (max_drawdown, duración en días del peor episodio, fecha de recuperación)."""
    if curva_equity.empty:
        return 0.0, 0, None
    maximo = curva_equity.cummax()
    caida = curva_equity / maximo - 1.0
    max_dd = float(caida.min())
    fecha_valle = caida.idxmin()
    # Pico previo al valle (donde el equity marcaba el máximo).
    pico = curva_equity.loc[:fecha_valle].idxmax()
    valor_pico = curva_equity.loc[pico]
    posteriores = curva_equity.loc[fecha_valle:]
    recuperadas = posteriores[posteriores >= valor_pico]
    fecha_recuperacion = recuperadas.index[0] if not recuperadas.empty else None
    fin = fecha_recuperacion if fecha_recuperacion is not None else curva_equity.index[-1]
    duracion = int((fin - pico).days)
    return max_dd, duracion, fecha_recuperacion


def metricas_cartera(
    retornos: pd.Series,
    nivel_confianza: float,
    tasa_libre_riesgo: float,
    dias_anio: int,
) -> MetricasCartera:
    retornos = retornos.dropna()
    if retornos.empty:
        return MetricasCartera(0, 0, 0, 0, 0, 0, 0, None, 0, 0)

    curva = equity(retornos)
    n = len(retornos)
    cagr = float(curva.iloc[-1] ** (dias_anio / n) - 1.0) if curva.iloc[-1] > 0 else -1.0
    media_anual = float(retornos.mean() * dias_anio)
    vol_anual = float(retornos.std(ddof=1) * np.sqrt(dias_anio)) if n > 1 else 0.0
    exceso = media_anual - tasa_libre_riesgo
    sharpe = exceso / vol_anual if vol_anual > 0 else 0.0

    bajistas = np.minimum(retornos.to_numpy(), 0.0)
    desv_bajista = float(np.sqrt(np.mean(bajistas ** 2)) * np.sqrt(dias_anio))
    sortino = exceso / desv_bajista if desv_bajista > 0 else 0.0

    max_dd, duracion, fecha_rec = drawdown(curva)
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    var, cvar = var_cvar_historico(retornos, nivel_confianza)

    return MetricasCartera(
        retorno_anual=cagr,
        volatilidad_anual=vol_anual,
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
        max_drawdown=max_dd,
        duracion_drawdown_dias=duracion,
        fecha_recuperacion=fecha_rec,
        var=var,
        cvar=cvar,
    )

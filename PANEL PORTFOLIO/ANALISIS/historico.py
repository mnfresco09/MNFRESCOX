"""Métricas REALIZADAS in-sample de la cartera seleccionada.

Reconstruye la curva de capital con pesos fijos (rebalanceo diario) sobre el
histórico alineado y deriva métricas exactas de desempeño realizado: máximo
drawdown exacto, CAGR, Sharpe histórico, Calmar y correlación media móvil.

Es desempeño realizado, NO un forecast: ninguna de estas cifras debe leerse como
promesa de retorno futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.modelos import Configuracion, MetricasHistoricas, PortfolioInput


def retornos_cartera(log_retornos: pd.DataFrame, pesos: pd.Series) -> pd.Series:
    """Retorno SIMPLE diario de la cartera con pesos fijos (rebalanceo diario)."""
    w = pesos.reindex(log_retornos.columns).fillna(0.0).to_numpy()
    simples = np.expm1(log_retornos.to_numpy())
    return pd.Series(simples @ w, index=log_retornos.index)


def correlacion_media_rolling(log_retornos: pd.DataFrame, ventana: int) -> pd.Series:
    """Media de las correlaciones par-a-par (triángulo superior) en cada ventana
    móvil de `ventana` días. Resume cómo se relacionan los activos a lo largo del
    tiempo en una sola serie."""
    n = log_retornos.shape[1]
    arr = log_retornos.to_numpy()
    filas = len(arr)
    if n < 2 or filas < ventana or ventana < 2:
        return pd.Series(dtype=float)
    iu = np.triu_indices(n, k=1)
    medias = np.full(filas, np.nan)
    for t in range(ventana - 1, filas):
        win = arr[t - ventana + 1 : t + 1]
        c = np.corrcoef(win, rowvar=False)
        medias[t] = float(np.nanmean(c[iu]))
    return pd.Series(medias, index=log_retornos.index).dropna()


def calcular_metricas_historicas(
    entrada: PortfolioInput,
    pesos: pd.Series,
    cfg: Configuracion,
) -> MetricasHistoricas:
    log_ret = entrada.log_retornos
    port = retornos_cartera(log_ret, pesos)
    n = len(port)

    equity = (1.0 + port).cumprod()
    pico = equity.cummax()
    dd = equity / pico - 1.0
    max_dd = float(dd.min()) if n else float("nan")
    fecha_valle = dd.idxmin() if n else None
    fecha_pico = equity.loc[:fecha_valle].idxmax() if fecha_valle is not None else None

    dias = cfg.dias_anio
    años = n / dias if dias else 0.0
    total = float(equity.iloc[-1]) if n else 1.0
    cagr = total ** (1.0 / años) - 1.0 if años > 0 and total > 0 else float("nan")

    r_log = np.log1p(port.to_numpy())
    mu_ann = float(np.mean(r_log)) * dias if n else float("nan")
    sigma_ann = float(np.std(r_log, ddof=1)) * np.sqrt(dias) if n > 1 else float("nan")
    rf = cfg.tasa_libre_riesgo_anual
    sharpe = (mu_ann - rf) / sigma_ann if sigma_ann and np.isfinite(sigma_ann) and sigma_ann > 0 else float("nan")

    calmar = cagr / abs(max_dd) if (max_dd < 0 and np.isfinite(cagr)) else float("nan")

    ventana = min(252, n)
    corr_roll = correlacion_media_rolling(log_ret, ventana)

    return MetricasHistoricas(
        max_drawdown=max_dd,
        fecha_pico_dd=fecha_pico,
        fecha_valle_dd=fecha_valle,
        cagr=cagr,
        sharpe_historico=sharpe,
        calmar=calmar,
        correlacion_rolling=corr_roll,
        ventana_rolling=ventana,
    )

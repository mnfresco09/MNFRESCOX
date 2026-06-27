"""Súper Score avanzado para seleccionar carteras Champion vs Challenger.

El score es ciego al motor que produjo la cartera y compara candidatos de forma
transversal con componentes explícitos de eficiencia, retorno, cola, drawdown,
concentración y fragilidad:

    score =
        2.50 * Z_robusto(sharpe)
      + 0.65 * Z_robusto(sortino)
      + 0.35 * Z_robusto(retorno_exceso)
      + 0.25 * Z_robusto(k_ratio)
      + 0.20 * Z_robusto(calmar)
      - 0.90 * Z_robusto(abs(CVaR99))
      - 1.10 * Z_robusto(abs(CDaR))
      - 0.55 * Z_robusto(abs(MaxDrawdown))
      - 0.05 * Z_robusto(HHI_pesos)
      - 0.60 * Z_robusto(HHI_riesgo)
      - 0.30 * Z_robusto(correlacion_ponderada)
      - 0.75 * Z_robusto(max(0, max_RC - 0.50))

Además aplica reglas duras institucionales. Si una cartera incumple límites de
CVaR, drawdown histórico, contribución máxima al riesgo o volatilidad, recibe
una penalización no compensable por eficiencia.

`R2` queda como diagnóstico in-sample en el contrato, no como driver de
selección. La trazabilidad queda en `PortfolioCandidate.detalle_score`.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from CONTRATOS.modelos import Configuracion, PortfolioCandidate

SCORE_CONFIG = {
    "peso_sharpe": 2.50,
    "peso_sortino": 0.65,
    "peso_retorno_exceso": 0.35,
    "peso_k_ratio": 0.25,
    "peso_calmar": 0.20,
    "peso_cvar": 0.90,
    "peso_cdar": 1.10,
    "peso_max_drawdown": 0.55,
    "peso_hhi_pesos": 0.05,
    "peso_hhi_riesgo": 0.60,
    "peso_correlacion": 0.30,
    "peso_max_contrib_riesgo": 0.75,
    "umbral_max_contrib_riesgo": 0.50,
}
REGLAS_DURAS = {
    "max_drawdown_historico_max": 0.4,
    "cvar_99_max": 0.065,
    "max_contrib_riesgo_max": 0.525,
    "volatilidad_max": 0.40,
}

PESO_SHARPE = SCORE_CONFIG["peso_sharpe"]
PESO_SORTINO = SCORE_CONFIG["peso_sortino"]
PESO_RETORNO_EXCESO = SCORE_CONFIG["peso_retorno_exceso"]
PESO_K_RATIO = SCORE_CONFIG["peso_k_ratio"]
PESO_CALMAR = SCORE_CONFIG["peso_calmar"]
PESO_CVAR = SCORE_CONFIG["peso_cvar"]
PESO_CDAR = SCORE_CONFIG["peso_cdar"]
PESO_MAX_DRAWDOWN = SCORE_CONFIG["peso_max_drawdown"]
PESO_HHI_PESOS = SCORE_CONFIG["peso_hhi_pesos"]
PESO_HHI_RIESGO = SCORE_CONFIG["peso_hhi_riesgo"]
PESO_CORRELACION = SCORE_CONFIG["peso_correlacion"]
PESO_MAX_CONTRIB_RIESGO = SCORE_CONFIG["peso_max_contrib_riesgo"]
UMBRAL_MAX_CONTRIB_RIESGO = SCORE_CONFIG["umbral_max_contrib_riesgo"]
PENALIZACION_REGLA_DURA = 1_000_000.0
PENALIZACION_INVALIDA = 1_000_000.0
TOPE_RATIO_EFICIENCIA = 10.0
EPS = 1e-12


def _z_robusto(valores: np.ndarray) -> np.ndarray:
    """Z-score robusto basado en mediana y MAD.

    Los valores no finitos se sustituyen por la mediana de los valores finitos
    para no distorsionar la escala. Si no hay variabilidad robusta, devuelve
    ceros: ese componente no afecta al ranking.
    """
    arr = np.asarray(valores, dtype=float)
    salida = np.zeros_like(arr, dtype=float)
    finitos = np.isfinite(arr)
    if not finitos.any():
        return salida

    saneados = arr.copy()
    mediana = float(np.median(saneados[finitos]))
    saneados[~finitos] = mediana
    mad = float(np.median(np.abs(saneados - mediana)))
    if mad <= EPS:
        return salida
    return (saneados - mediana) / (1.4826 * mad)


def _pesos_normalizados(candidato: PortfolioCandidate, cfg: Configuracion) -> pd.Series:
    pesos = candidato.pesos.astype(float)
    if not np.isfinite(pesos.to_numpy(dtype=float)).all():
        raise ValueError(f"Pesos no finitos en {candidato.motor_optimizacion}/{candidato.nivel}.")
    if (pesos < -1e-8).any():
        raise ValueError(f"Pesos negativos en cartera long-only {candidato.motor_optimizacion}/{candidato.nivel}.")

    suma = float(pesos.sum())
    if abs(suma - 1.0) > 1e-3:
        raise ValueError(
            f"La suma de pesos de {candidato.motor_optimizacion}/{candidato.nivel} es {suma:.6f}, no 1."
        )

    restr = cfg.restricciones
    if restr.peso_maximo is not None and (pesos > float(restr.peso_maximo) + 1e-3).any():
        raise ValueError(f"Peso por encima del máximo en {candidato.motor_optimizacion}/{candidato.nivel}.")
    if (pesos < float(restr.peso_minimo) - 1e-3).any():
        raise ValueError(f"Peso por debajo del mínimo en {candidato.motor_optimizacion}/{candidato.nivel}.")
    return pesos / max(suma, EPS)


def _valor_finito(valor: float | None) -> tuple[float, bool]:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return 0.0, False
    if not np.isfinite(v):
        return 0.0, False
    return v, True


def _cvar_abs(candidato: PortfolioCandidate) -> tuple[float, bool]:
    if candidato.forecast is None:
        return 0.0, False
    v, ok = _valor_finito(candidato.forecast.cvar_fhs_99)
    return abs(v), ok


def _cdar_abs(candidato: PortfolioCandidate) -> tuple[float, bool]:
    if candidato.simulacion is None:
        return 0.0, False
    v, ok = _valor_finito(candidato.simulacion.cdar_30d)
    return abs(v), ok


def _hhi_pesos(pesos: pd.Series) -> float:
    w = pesos.to_numpy(dtype=float)
    return float(np.square(w).sum())


def _hhi_riesgo(candidato: PortfolioCandidate, pesos: pd.Series) -> tuple[float, float, bool]:
    contrib = getattr(candidato.descomposicion, "contribucion_pct", None)
    if contrib is None:
        return _hhi_pesos(pesos), float(pesos.max()), False
    rc = contrib.reindex(pesos.index).to_numpy(dtype=float)
    if not np.isfinite(rc).all():
        return _hhi_pesos(pesos), float(pesos.max()), False
    rc_abs = np.abs(rc)
    total = float(rc_abs.sum())
    if total <= EPS:
        return _hhi_pesos(pesos), float(pesos.max()), False
    rc_norm = rc_abs / total
    return float(np.square(rc_norm).sum()), float(rc_norm.max()), True


def _correlacion_ponderada(pesos: pd.Series, correlacion: pd.DataFrame | None) -> tuple[float, bool]:
    if correlacion is None or correlacion.empty:
        return 0.0, False
    activos = list(pesos.index)
    corr = correlacion.reindex(index=activos, columns=activos).fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(corr).all():
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 0.0)
    corr_pos = np.clip(corr, 0.0, None)
    w = pesos.to_numpy(dtype=float)
    return float(w @ corr_pos @ w), True


def _ratio_acotado(valor: float) -> float:
    if not np.isfinite(valor):
        return 0.0
    return float(np.clip(valor, -TOPE_RATIO_EFICIENCIA, TOPE_RATIO_EFICIENCIA))


def _metricas_curva_historica(
    pesos: pd.Series,
    cfg: Configuracion,
    log_retornos: pd.DataFrame | None,
) -> dict[str, float]:
    """Métricas de eficiencia realizadas con pesos fijos.

    Si no hay retornos disponibles, el componente queda neutral y las reglas de
    drawdown histórico se desactivan para mantener compatibilidad con llamadas
    unitarias antiguas.
    """
    if log_retornos is None or log_retornos.empty:
        return {
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown_abs": 0.0,
            "metrica_curva_historica_activa": 0.0,
        }

    cols = list(log_retornos.columns)
    w = pesos.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    datos = log_retornos.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if datos.empty or not np.isfinite(w).all():
        return {
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown_abs": 0.0,
            "metrica_curva_historica_activa": 0.0,
        }

    port = np.expm1(datos.to_numpy(dtype=float)) @ w
    port = port[np.isfinite(port)]
    port = port[port > -1.0 + EPS]
    if port.size == 0:
        return {
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown_abs": 0.0,
            "metrica_curva_historica_activa": 0.0,
        }

    equity = np.cumprod(1.0 + port)
    picos = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(picos, EPS) - 1.0
    max_drawdown = float(np.min(drawdowns))
    max_drawdown_abs = abs(max_drawdown) if np.isfinite(max_drawdown) else 0.0

    dias = max(int(cfg.dias_anio), 1)
    años = port.size / dias
    total = float(equity[-1])
    cagr = total ** (1.0 / años) - 1.0 if años > 0 and total > 0 else 0.0

    log_port = np.log1p(port)
    mu_anual = float(np.mean(log_port)) * dias
    rf = float(cfg.tasa_libre_riesgo_anual)
    rf_diaria = (1.0 + rf) ** (1.0 / dias) - 1.0 if 1.0 + rf > 0 else rf / dias
    exceso_diario = port - rf_diaria
    downside = np.minimum(exceso_diario, 0.0)
    downside_vol = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(dias))

    if downside_vol > EPS:
        sortino = (mu_anual - rf) / downside_vol
    elif mu_anual > rf:
        sortino = TOPE_RATIO_EFICIENCIA
    else:
        sortino = 0.0

    if max_drawdown_abs > EPS:
        calmar = cagr / max_drawdown_abs
    elif cagr > 0:
        calmar = TOPE_RATIO_EFICIENCIA
    else:
        calmar = 0.0

    return {
        "sortino": _ratio_acotado(sortino),
        "calmar": _ratio_acotado(calmar),
        "max_drawdown_abs": max_drawdown_abs,
        "metrica_curva_historica_activa": 1.0,
    }


def _metricas_candidato(
    candidato: PortfolioCandidate,
    cfg: Configuracion,
    correlacion: pd.DataFrame | None,
    log_retornos: pd.DataFrame | None,
) -> dict[str, float]:
    pesos = _pesos_normalizados(candidato, cfg)
    retorno, ok_retorno = _valor_finito(candidato.retorno_esperado)
    sharpe, ok_sharpe = _valor_finito(candidato.sharpe)
    k_ratio, _ok_k_ratio = _valor_finito(candidato.k_ratio)
    cvar, ok_cvar = _cvar_abs(candidato)
    cdar, ok_cdar = _cdar_abs(candidato)
    hhi_pesos = _hhi_pesos(pesos)
    hhi_riesgo, max_contribucion_riesgo, ok_hhi_riesgo = _hhi_riesgo(candidato, pesos)
    corr_pond, ok_corr = _correlacion_ponderada(pesos, correlacion)
    penalizacion_max_contribucion = max(0.0, max_contribucion_riesgo - UMBRAL_MAX_CONTRIB_RIESGO)
    curva = _metricas_curva_historica(pesos, cfg, log_retornos)
    volatilidad, ok_volatilidad = _valor_finito(candidato.volatilidad_tactica)

    regla_maxdd = float(
        bool(curva["metrica_curva_historica_activa"])
        and curva["max_drawdown_abs"] > REGLAS_DURAS["max_drawdown_historico_max"] + EPS
    )
    regla_cvar = float(ok_cvar and cvar > REGLAS_DURAS["cvar_99_max"] + EPS)
    regla_max_contrib = float(max_contribucion_riesgo > REGLAS_DURAS["max_contrib_riesgo_max"] + EPS)
    regla_volatilidad = float(ok_volatilidad and volatilidad > REGLAS_DURAS["volatilidad_max"] + EPS)
    total_reglas_incumplidas = regla_maxdd + regla_cvar + regla_max_contrib + regla_volatilidad

    invalida = not all((ok_sharpe, ok_retorno, ok_cvar, ok_cdar, ok_volatilidad))
    retorno_exceso = retorno - float(cfg.tasa_libre_riesgo_anual)
    return {
        "sharpe": sharpe,
        "sortino": curva["sortino"],
        "retorno_exceso": retorno_exceso,
        "k_ratio": k_ratio,
        "calmar": curva["calmar"],
        "cvar_abs": cvar,
        "cdar_abs": cdar,
        "max_drawdown_abs": curva["max_drawdown_abs"],
        "hhi_pesos": hhi_pesos,
        "hhi_riesgo": hhi_riesgo,
        "max_contribucion_riesgo": max_contribucion_riesgo,
        "volatilidad_tactica": volatilidad,
        "correlacion_ponderada": corr_pond,
        "componente_hhi_riesgo_activo": float(ok_hhi_riesgo),
        "componente_correlacion_activo": float(ok_corr),
        "metrica_curva_historica_activa": curva["metrica_curva_historica_activa"],
        "penalizacion_max_contribucion_riesgo": penalizacion_max_contribucion,
        "regla_max_drawdown_incumplida": regla_maxdd,
        "regla_cvar_99_incumplida": regla_cvar,
        "regla_max_contrib_riesgo_incumplida": regla_max_contrib,
        "regla_volatilidad_incumplida": regla_volatilidad,
        "reglas_duras_incumplidas": total_reglas_incumplidas,
        "penalizacion_reglas_duras": total_reglas_incumplidas * PENALIZACION_REGLA_DURA,
        "penalizacion_invalida": PENALIZACION_INVALIDA if invalida else 0.0,
    }


def calcular_score_compuesto(
    candidatos: tuple[PortfolioCandidate, ...],
    cfg: Configuracion,
    correlacion: pd.DataFrame | None = None,
    log_retornos: pd.DataFrame | None = None,
) -> tuple[PortfolioCandidate, ...]:
    """Devuelve candidatos con `score` y `detalle_score`.

    `correlacion` y `log_retornos` son opcionales para mantener compatibilidad.
    Si no se pasan, sus componentes quedan neutrales y las reglas dependientes
    de la curva histórica no se activan.
    """
    if not candidatos:
        return ()

    metricas = [_metricas_candidato(c, cfg, correlacion, log_retornos) for c in candidatos]
    sharpe = np.array([m["sharpe"] for m in metricas], dtype=float)
    sortino = np.array([m["sortino"] for m in metricas], dtype=float)
    retorno = np.array([m["retorno_exceso"] for m in metricas], dtype=float)
    k_ratio = np.array([m["k_ratio"] for m in metricas], dtype=float)
    calmar = np.array([m["calmar"] for m in metricas], dtype=float)
    cvar = np.array([m["cvar_abs"] for m in metricas], dtype=float)
    cdar = np.array([m["cdar_abs"] for m in metricas], dtype=float)
    max_drawdown = np.array([m["max_drawdown_abs"] for m in metricas], dtype=float)
    hhi_pesos = np.array([m["hhi_pesos"] for m in metricas], dtype=float)
    hhi_riesgo = np.array([m["hhi_riesgo"] for m in metricas], dtype=float)
    corr = np.array([m["correlacion_ponderada"] for m in metricas], dtype=float)
    penalizacion_max_rc = np.array([m["penalizacion_max_contribucion_riesgo"] for m in metricas], dtype=float)

    z_sharpe = _z_robusto(sharpe)
    z_sortino = _z_robusto(sortino)
    z_retorno = _z_robusto(retorno)
    z_k_ratio = _z_robusto(k_ratio)
    z_calmar = _z_robusto(calmar)
    z_cvar = _z_robusto(cvar)
    z_cdar = _z_robusto(cdar)
    z_max_drawdown = _z_robusto(max_drawdown)
    z_hhi_pesos = _z_robusto(hhi_pesos)
    z_hhi_riesgo = _z_robusto(hhi_riesgo)
    z_corr = _z_robusto(corr)
    z_penalizacion_max_rc = _z_robusto(penalizacion_max_rc)

    salida: list[PortfolioCandidate] = []
    for i, candidato in enumerate(candidatos):
        comp_sharpe = PESO_SHARPE * float(z_sharpe[i])
        comp_sortino = PESO_SORTINO * float(z_sortino[i])
        comp_retorno = PESO_RETORNO_EXCESO * float(z_retorno[i])
        comp_k_ratio = PESO_K_RATIO * float(z_k_ratio[i])
        comp_calmar = PESO_CALMAR * float(z_calmar[i])
        comp_cvar = -PESO_CVAR * float(z_cvar[i])
        comp_cdar = -PESO_CDAR * float(z_cdar[i])
        comp_max_drawdown = -PESO_MAX_DRAWDOWN * float(z_max_drawdown[i])
        comp_hhi_pesos = -PESO_HHI_PESOS * float(z_hhi_pesos[i])
        comp_hhi_riesgo = -PESO_HHI_RIESGO * float(z_hhi_riesgo[i])
        comp_corr = -PESO_CORRELACION * float(z_corr[i])
        comp_max_contrib_riesgo = -PESO_MAX_CONTRIB_RIESGO * float(z_penalizacion_max_rc[i])
        penalizacion_reglas = -float(metricas[i]["penalizacion_reglas_duras"])
        penalizacion = -float(metricas[i]["penalizacion_invalida"])
        score = float(
            comp_sharpe
            + comp_sortino
            + comp_retorno
            + comp_k_ratio
            + comp_calmar
            + comp_cvar
            + comp_cdar
            + comp_max_drawdown
            + comp_hhi_pesos
            + comp_hhi_riesgo
            + comp_corr
            + comp_max_contrib_riesgo
            + penalizacion_reglas
            + penalizacion
        )
        if not np.isfinite(score):
            score = -PENALIZACION_INVALIDA

        detalle = (
            ("score_final", score),
            ("sharpe", float(sharpe[i])),
            ("sortino", float(sortino[i])),
            ("retorno_exceso", float(retorno[i])),
            ("k_ratio", float(k_ratio[i])),
            ("calmar", float(calmar[i])),
            ("cvar_abs", float(cvar[i])),
            ("cdar_abs", float(cdar[i])),
            ("max_drawdown_abs", float(max_drawdown[i])),
            ("hhi_pesos", float(hhi_pesos[i])),
            ("hhi_riesgo", float(hhi_riesgo[i])),
            ("max_contribucion_riesgo", float(metricas[i]["max_contribucion_riesgo"])),
            ("volatilidad_tactica", float(metricas[i]["volatilidad_tactica"])),
            ("correlacion_ponderada", float(corr[i])),
            ("penalizacion_max_contribucion_riesgo", float(penalizacion_max_rc[i])),
            ("z_sharpe", float(z_sharpe[i])),
            ("z_sharpe_robusto", float(z_sharpe[i])),
            ("z_sortino", float(z_sortino[i])),
            ("z_retorno_exceso", float(z_retorno[i])),
            ("z_k_ratio", float(z_k_ratio[i])),
            ("z_calmar", float(z_calmar[i])),
            ("z_cvar", float(z_cvar[i])),
            ("z_cdar", float(z_cdar[i])),
            ("z_max_drawdown", float(z_max_drawdown[i])),
            ("z_hhi_pesos", float(z_hhi_pesos[i])),
            ("z_hhi_riesgo", float(z_hhi_riesgo[i])),
            ("z_correlacion", float(z_corr[i])),
            ("z_penalizacion_max_contrib_riesgo", float(z_penalizacion_max_rc[i])),
            ("comp_sharpe", comp_sharpe),
            ("comp_sortino", comp_sortino),
            ("comp_retorno_exceso", comp_retorno),
            ("comp_k_ratio", comp_k_ratio),
            ("comp_calmar", comp_calmar),
            ("comp_cvar", comp_cvar),
            ("comp_cdar", comp_cdar),
            ("comp_max_drawdown", comp_max_drawdown),
            ("comp_hhi_pesos", comp_hhi_pesos),
            ("comp_hhi_riesgo", comp_hhi_riesgo),
            ("comp_correlacion", comp_corr),
            ("comp_max_contrib_riesgo", comp_max_contrib_riesgo),
            ("regla_max_drawdown_incumplida", float(metricas[i]["regla_max_drawdown_incumplida"])),
            ("regla_cvar_99_incumplida", float(metricas[i]["regla_cvar_99_incumplida"])),
            ("regla_max_contrib_riesgo_incumplida", float(metricas[i]["regla_max_contrib_riesgo_incumplida"])),
            ("regla_volatilidad_incumplida", float(metricas[i]["regla_volatilidad_incumplida"])),
            ("reglas_duras_incumplidas", float(metricas[i]["reglas_duras_incumplidas"])),
            ("penalizacion_reglas_duras", float(metricas[i]["penalizacion_reglas_duras"])),
            ("comp_reglas_duras", penalizacion_reglas),
            ("penalizacion_invalida", penalizacion),
            ("metrica_curva_historica_activa", float(metricas[i]["metrica_curva_historica_activa"])),
            ("componente_hhi_riesgo_activo", float(metricas[i]["componente_hhi_riesgo_activo"])),
            ("componente_correlacion_activo", float(metricas[i]["componente_correlacion_activo"])),
        )
        salida.append(replace(candidato, score=score, detalle_score=detalle))
    return tuple(salida)


def calcular_score_cartera(
    candidatos: tuple[PortfolioCandidate, ...],
    cfg: Configuracion,
    cartera_previa=None,
    correlacion: pd.DataFrame | None = None,
    log_retornos: pd.DataFrame | None = None,
) -> tuple[PortfolioCandidate, ...]:
    """Compatibilidad con el pipeline existente; usa el Súper Score avanzado."""
    _ = cartera_previa
    return calcular_score_compuesto(candidatos, cfg, correlacion=correlacion, log_retornos=log_retornos)

"""Exploración multi-criterio de la frontera (pregunta 3, multi-lente).

Hace dos cosas:

  1. CLASIFICACIÓN automática por ANCLAS + BANDAS. Ancla en mínima varianza,
     Máx Sharpe y máximo retorno; entre esas anclas define bandas de volatilidad
     Bajo / Medio / Alto y etiqueta cada punto de la frontera y de la nube.

  2. LEADERBOARD Top-5 por criterio. Cada cartera de la frontera se evalúa con
     métricas baratas/cerradas o paramétricas (instantáneo): Sharpe táctico,
     Score-proxy, VaR/CVaR paramétrico, STARR (retorno/CVaR), ratio de
     diversificación (Choueifaty) y concentración de riesgo ERC (HHI de MCR).
     Se criba el Top-5 por criterio y SOLO esa shortlist (más los candidatos principales) se
     confirma luego con FHS + Monte Carlo precisos (ver RIESGO.exploracion_riesgo).

Convención: VaR/CVaR negativos (pérdida en la cola); estimaciones del modelo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from CONTRATOS.modelos import (
    Configuracion,
    DescomposicionRiesgo,
    MomentsResult,
    PortfolioCandidate,
    ResultadoFrontera,
)
from RIESGO.mcr import descomponer_riesgo


# --------------------------------------------------------------------------- #
#  Clasificación por anclas + bandas                                           #
# --------------------------------------------------------------------------- #
def _bandas(puntos: pd.DataFrame, vol_maxsharpe: float) -> tuple[float, float]:
    """Bordes (b1, b2) de las bandas Bajo/Medio/Alto ancladas en la estructura."""
    vol = puntos["volatilidad"].to_numpy()
    vmin, vmax = float(vol.min()), float(vol.max())
    vmid = float(min(max(vol_maxsharpe, vmin), vmax))
    b1 = (vmin + vmid) / 2.0
    b2 = (vmid + vmax) / 2.0
    if not (vmin < b1 < b2 < vmax):  # degenerado → terciles
        b1 = vmin + (vmax - vmin) / 3.0
        b2 = vmin + 2.0 * (vmax - vmin) / 3.0
    return b1, b2


def _clase(vol: float, b1: float, b2: float) -> str:
    if vol <= b1:
        return "bajo"
    if vol <= b2:
        return "medio"
    return "alto"


# --------------------------------------------------------------------------- #
#  Métricas baratas por cartera                                                #
# --------------------------------------------------------------------------- #
def _metricas_baratas(w: np.ndarray, momentos: MomentsResult, cfg: Configuracion) -> dict:
    activos = list(momentos.cov_estructural.index)
    mu = momentos.retornos_ajustados.reindex(activos).to_numpy(dtype=float)
    cov_e = momentos.cov_estructural.to_numpy(dtype=float)
    cov_t = momentos.cov_tactica.to_numpy(dtype=float)
    sig_t = np.sqrt(np.diag(cov_t))
    rf = cfg.tasa_libre_riesgo_anual

    ret = float(w @ mu)
    vol_e = float(np.sqrt(max(w @ cov_e @ w, 0.0)))
    vol_t = float(np.sqrt(max(w @ cov_t @ w, 0.0)))
    sharpe = (ret - rf) / vol_t if vol_t > 0 else 0.0
    # Ratio de diversificación (Choueifaty): media ponderada de vols / vol cartera.
    diversificacion = float((w @ sig_t) / vol_t) if vol_t > 0 else 1.0
    # ERC: concentración (HHI) de las contribuciones al riesgo.
    contrib = w * (cov_t @ w)
    contrib_pct = contrib / contrib.sum() if contrib.sum() > 0 else contrib
    erc = float((contrib_pct ** 2).sum())
    # VaR/CVaR paramétrico diario (cribado rápido).
    sig_d = vol_t / np.sqrt(cfg.dias_anio)
    mu_d = (ret - 0) / cfg.dias_anio
    alpha = 1.0 - cfg.nivel_confianza_99
    z = norm.ppf(alpha)
    var99 = mu_d + sig_d * z
    cvar99 = mu_d - sig_d * norm.pdf(z) / alpha
    # STARR = retorno excedente anual / CVaR99 anualizado.
    cvar99_anual = abs(cvar99) * np.sqrt(cfg.dias_anio)
    starr = (ret - rf) / cvar99_anual if cvar99_anual > 0 else 0.0
    return dict(retorno=ret, vol_e=vol_e, vol_t=vol_t, sharpe=sharpe,
                diversificacion=diversificacion, erc=erc,
                var99=float(var99), cvar99=float(cvar99), starr=float(starr))


def construir_candidato(
    clave: str, pesos: pd.Series, momentos: MomentsResult, cfg: Configuracion, clase: str
) -> PortfolioCandidate:
    activos = list(momentos.cov_estructural.index)
    w = pesos.reindex(activos)
    m = _metricas_baratas(w.to_numpy(dtype=float), momentos, cfg)
    descomp: DescomposicionRiesgo = descomponer_riesgo(w, momentos.cov_tactica)
    return PortfolioCandidate(
        nivel=clave, pesos=w,
        retorno_esperado=m["retorno"], volatilidad_estructural=m["vol_e"],
        volatilidad_tactica=m["vol_t"], sharpe=m["sharpe"], descomposicion=descomp,
        diversificacion=m["diversificacion"], starr=m["starr"],
        erc_concentracion=m["erc"], clase_riesgo=clase,
    )


# --------------------------------------------------------------------------- #
#  Tabla de exploración + criterios                                            #
# --------------------------------------------------------------------------- #
# (clave, nombre, descripcion, columna, sentido)
CRITERIOS = (
    ("sharpe", "Máx Sharpe", "Mejor binomio retorno/riesgo táctico.", "sharpe", "max"),
    ("score", "Máx Score", "Score multifactor (Sharpe penalizado por riesgo).", "score_proxy", "max"),
    ("var99", "Mín VaR 99%", "Menor pérdida estimada mañana (cola 1%).", "var99", "max"),
    ("cdar", "Mín CDaR", "Menor drawdown de cola proyectado (proxy vol táctica).", "cdar_proxy", "max"),
    ("starr", "Máx STARR", "Retorno excedente por unidad de CVaR99 (tail-aware).", "starr", "max"),
    ("diversificacion", "Máx Diversificación", "Ratio de Choueifaty: explota la anticorrelación.", "diversificacion", "max"),
)


def tabla_exploracion(
    frontera: ResultadoFrontera, momentos: MomentsResult, cfg: Configuracion
) -> tuple[pd.DataFrame, tuple[float, float], float]:
    """DataFrame con métricas baratas + clase por punto de la frontera."""
    activos = list(momentos.cov_estructural.index)
    cols_peso = [f"peso·{a}" for a in activos]
    puntos = frontera.puntos

    w_ms = frontera.maximo_sharpe_pesos.reindex(activos).to_numpy(dtype=float)
    vol_ms = float(np.sqrt(max(w_ms @ momentos.cov_estructural.to_numpy() @ w_ms, 0.0)))
    b1, b2 = _bandas(puntos, vol_ms)

    filas = []
    for _, fila in puntos.iterrows():
        w = fila[cols_peso].to_numpy(dtype=float)
        m = _metricas_baratas(w, momentos, cfg)
        m["vol_struct"] = float(fila["volatilidad"])
        m["clase"] = _clase(m["vol_struct"], b1, b2)  # banda sobre vol estructural (eje frontera)
        # CDaR proxy: escala de cola a horizonte (monótona en vol táctica).
        m["cdar_proxy"] = -m["vol_t"] / np.sqrt(cfg.dias_anio) * np.sqrt(cfg.horizonte_dias) * 2.33
        for a, peso in zip(activos, w):
            m[f"peso·{a}"] = float(peso)
        filas.append(m)
    tabla = pd.DataFrame(filas)

    # Score-proxy: z(sharpe) − z(|var99|) − z(hhi-erc) (versión barata del score).
    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0
    tabla["score_proxy"] = (z(tabla["sharpe"]) - z(tabla["var99"].abs())
                            - 0.5 * z(tabla["erc"]) + 0.3 * z(tabla["starr"]))
    return tabla, (b1, b2), vol_ms


def top_por_criterio(tabla: pd.DataFrame, n: int = 5) -> dict[str, pd.DataFrame]:
    """Top-N filas de la frontera por cada criterio (cribado paramétrico)."""
    salida: dict[str, pd.DataFrame] = {}
    for clave, _nombre, _desc, columna, sentido in CRITERIOS:
        asc = sentido == "min"
        salida[clave] = tabla.sort_values(columna, ascending=asc).head(n).reset_index(drop=True)
    return salida

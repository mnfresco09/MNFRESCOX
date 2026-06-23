"""Gráficos estáticos (matplotlib) para el informe PDF.

Misma información que la versión interactiva, en imágenes de alta resolución
aptas para impresión. Guarda PNG en una carpeta y devuelve sus rutas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CONTRATOS.modelos import PaqueteReporte
from OPTIMIZACION.asignadores import metodos

from .formato import ACENTO, COLOR_METODO, SUAVE, TINTA, nombre_visible
from .i18n import perfil_visible, t

_PALETA_ACTIVOS = ["#1D4ED8", "#0E7490", "#15803D", "#B45309", "#BE123C", "#7C3AED", "#475569"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": "#D9DEE7",
    "axes.titlecolor": TINTA,
    "axes.labelcolor": TINTA,
    "text.color": TINTA,
    "xtick.color": SUAVE,
    "ytick.color": SUAVE,
    "axes.grid": True,
    "grid.color": "#E8ECF3",
    "figure.dpi": 150,
})


def _guardar(fig, carpeta: Path, nombre: str) -> Path:
    ruta = carpeta / f"{nombre}.png"
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


def _titulo_perfil(paquete: PaqueteReporte, nivel: str) -> str:
    return perfil_visible(paquete, nivel)


def generar_pngs(paquete: PaqueteReporte, carpeta: Path) -> dict[str, Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas: dict[str, Path] = {}

    # Frontera + nube + métodos
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    mc = paquete.monte_carlo.metricas
    sc = ax.scatter(mc["volatilidad"], mc["retorno"], c=mc["sharpe"], cmap="Blues", s=6, alpha=0.4)
    fr = paquete.frontera.puntos.sort_values("volatilidad")
    ax.plot(fr["volatilidad"], fr["retorno"], color=TINTA, lw=2, label=t(paquete, "frontera_eficiente_nombre"))
    for metodo in metodos(paquete.configuracion):
        m = paquete.asignaciones[metodo].metricas
        ax.scatter(m.volatilidad_anual, m.retorno_anual, s=70, marker="D",
                   color=COLOR_METODO.get(metodo, ACENTO), edgecolor="white", zorder=5, label=nombre_visible(metodo, paquete))
    ax.set_title(t(paquete, "frontera_titulo"))
    ax.set_xlabel(t(paquete, "volatilidad_anual")); ax.set_ylabel(t(paquete, "retorno_anual_esperado"))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(fontsize=7, loc="best")
    fig.colorbar(sc, ax=ax, label="Sharpe")
    rutas["frontera"] = _guardar(fig, carpeta, "frontera")

    # Pesos recomendados del perfil elegido
    recomendada = paquete.perfil_riesgo.recomendada
    pesos_rec = recomendada.pesos.sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colores = [_PALETA_ACTIVOS[i % len(_PALETA_ACTIVOS)] for i in range(len(pesos_rec))]
    ax.barh(pesos_rec.index, pesos_rec.values, color=colores)
    for i, valor in enumerate(pesos_rec.values):
        ax.text(valor + 0.005, i, f"{valor:.1%}", va="center", fontsize=8)
    mh = recomendada.metricas_historicas
    ax.set_title(
        f"{t(paquete, 'recomendacion')} · {t(paquete, 'perfil').format(perfil=_titulo_perfil(paquete, recomendada.nivel))}\n"
        f"{t(paquete, 'retorno_esperado')} {recomendada.retorno_esperado:.1%} · {t(paquete, 'volatilidad_esperada')} {recomendada.volatilidad_esperada:.1%} · "
        f"VaR {mh.var:.1%} · CVaR {mh.cvar:.1%} · maxDD {mh.max_drawdown:.1%}"
    )
    ax.set_xlabel(t(paquete, "peso_recomendado"))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, max(0.01, float(pesos_rec.max()) * 1.18))
    rutas["pesos_recomendados"] = _guardar(fig, carpeta, "pesos_recomendados")

    # Pesos por nivel de riesgo
    niveles = list(paquete.perfil_riesgo.carteras)
    etiquetas_nivel = [_titulo_perfil(paquete, c.nivel) for c in niveles]
    activos = list(paquete.datos.activos)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    base = np.zeros(len(niveles))
    for i, activo in enumerate(activos):
        valores = np.array([float(c.pesos[activo]) for c in niveles])
        ax.bar(etiquetas_nivel, valores, bottom=base, color=_PALETA_ACTIVOS[i % len(_PALETA_ACTIVOS)], label=activo)
        base += valores
    ax.set_title(t(paquete, "niveles"))
    ax.set_ylabel(t(paquete, "peso"))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=len(activos), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    rutas["pesos_niveles"] = _guardar(fig, carpeta, "pesos_niveles")

    # Composición a lo largo de la frontera
    niveles_frontera = paquete.perfil_riesgo.niveles_frontera.sort_index()
    activos_frontera = [a for a in activos if a in niveles_frontera.columns]
    x = niveles_frontera.index.to_numpy(dtype=float)
    y = [niveles_frontera[a].to_numpy(dtype=float) for a in activos_frontera]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.stackplot(
        x,
        y,
        labels=activos_frontera,
        colors=[_PALETA_ACTIVOS[i % len(_PALETA_ACTIVOS)] for i in range(len(activos_frontera))],
        alpha=0.90,
    )
    ax.axvline(recomendada.volatilidad_esperada, color=TINTA, ls=":", lw=1.4)
    ax.set_title(t(paquete, "composicion_frontera_titulo"))
    ax.set_xlabel(t(paquete, "volatilidad_anual"))
    ax.set_ylabel(t(paquete, "peso"))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=len(activos_frontera), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    rutas["composicion_frontera"] = _guardar(fig, carpeta, "composicion_frontera")

    # Equity OOS
    eq = paquete.riesgo.walk_forward.equity
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for metodo in eq.columns:
        ax.plot(eq.index, eq[metodo], lw=1.6, color=COLOR_METODO.get(metodo, SUAVE), label=nombre_visible(metodo, paquete))
    ax.set_title(t(paquete, "equity_titulo"))
    ax.set_xlabel(t(paquete, "fecha")); ax.set_ylabel(t(paquete, "capital_acumulado"))
    ax.legend(fontsize=7)
    rutas["equity"] = _guardar(fig, carpeta, "equity")

    # Drawdown
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for metodo in eq.columns:
        caida = eq[metodo] / eq[metodo].cummax() - 1.0
        ax.plot(caida.index, caida, lw=1.2, color=COLOR_METODO.get(metodo, SUAVE), label=nombre_visible(metodo, paquete))
    ax.set_title(t(paquete, "drawdown_titulo"))
    ax.set_xlabel(t(paquete, "fecha")); ax.set_ylabel(t(paquete, "drawdown"))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(fontsize=7)
    rutas["drawdown"] = _guardar(fig, carpeta, "drawdown")

    # Correlación media + exceso de cola
    for nombre, matriz, cmap, titulo in (
        ("correlacion_media", paquete.analisis.correlacion_media, "RdBu_r", t(paquete, "corr_media_titulo")),
        ("correlacion_cola", paquete.analisis.diferencia_correlacion_cola, "Reds", t(paquete, "corr_cola_titulo")),
    ):
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        im = ax.imshow(matriz.values, cmap=cmap, vmin=-1 if "media" in nombre else None,
                       vmax=1 if "media" in nombre else None)
        ax.set_xticks(range(len(matriz.columns))); ax.set_xticklabels(matriz.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(matriz.index))); ax.set_yticklabels(matriz.index, fontsize=7)
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                ax.text(j, i, f"{matriz.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(titulo); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
        rutas[nombre] = _guardar(fig, carpeta, nombre)

    # PCA
    ve = paquete.analisis.pca.varianza_explicada
    vac = paquete.analisis.pca.varianza_acumulada
    etiquetas = [f"PC{i + 1}" for i in range(len(ve))]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(etiquetas, ve.values, color=ACENTO, alpha=0.85, label=t(paquete, "varianza_explicada"))
    ax2 = ax.twinx()
    ax2.plot(etiquetas, vac.values, color=TINTA, marker="o", lw=2, label=t(paquete, "acumulada"))
    ax2.axhline(0.90, color=SUAVE, ls=":")
    ax2.set_ylim(0, 1.02)
    ax.set_title(t(paquete, "pca_titulo"))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.grid(False)
    rutas["pca"] = _guardar(fig, carpeta, "pca")

    # Pesos apilados
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    metodos_ord = list(metodos(paquete.configuracion))
    izquierda = np.zeros(len(metodos_ord))
    for i, activo in enumerate(activos):
        valores = np.array([float(paquete.asignaciones[m].pesos[activo]) for m in metodos_ord])
        etiquetas_metodos = [nombre_visible(m, paquete) for m in metodos_ord]
        ax.barh(etiquetas_metodos, valores, left=izquierda, color=_PALETA_ACTIVOS[i % len(_PALETA_ACTIVOS)], label=activo)
        izquierda += valores
    ax.set_title(t(paquete, "pesos_metodos_titulo"))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.invert_yaxis()
    ax.legend(fontsize=7, ncol=len(activos), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    rutas["pesos"] = _guardar(fig, carpeta, "pesos")

    # Diversificación global vs crisis
    div = paquete.riesgo.diversificacion_crisis
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(div.index))
    ax.bar(x - 0.2, div["enb_global"], width=0.4, color=ACENTO, label="Global")
    ax.bar(x + 0.2, div["enb_crisis"], width=0.4, color="#B45309", label="Crisis")
    ax.set_xticks(x); ax.set_xticklabels([nombre_visible(m, paquete) for m in div.index], rotation=20, ha="right", fontsize=7)
    ax.set_title(t(paquete, "diversificacion_titulo"))
    ax.set_ylabel(t(paquete, "apuestas_independientes"))
    ax.legend(fontsize=8)
    rutas["diversificacion"] = _guardar(fig, carpeta, "diversificacion")

    # Convexidad por escenario OOS
    conv = paquete.riesgo.convexidad
    escenarios = [
        ("ret_medio_todo_baja", t(paquete, "todo_baja"), "#B91C1C"),
        ("ret_medio_mixto", t(paquete, "mixto"), SUAVE),
        ("ret_medio_todo_sube", t(paquete, "todo_sube"), "#15803D"),
    ]
    columnas = [c for c, _, _ in escenarios if c in conv.columns]
    if columnas:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(conv.index))
        ancho = 0.24
        offset = -ancho * (len(columnas) - 1) / 2
        for j, clave in enumerate(columnas):
            nombre = next(nombre for c, nombre, _ in escenarios if c == clave)
            color = next(color for c, _, color in escenarios if c == clave)
            ax.bar(x + offset + j * ancho, conv[clave].to_numpy(dtype=float), width=ancho, label=nombre, color=color)
        ax.set_xticks(x); ax.set_xticklabels([nombre_visible(m, paquete) for m in conv.index], rotation=20, ha="right", fontsize=7)
        ax.set_title(t(paquete, "convexidad_titulo"))
        ax.set_ylabel(t(paquete, "retorno_medio_diario"))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
        ax.legend(fontsize=8)
        rutas["convexidad"] = _guardar(fig, carpeta, "convexidad")

    return rutas

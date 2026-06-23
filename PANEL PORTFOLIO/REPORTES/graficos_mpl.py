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
from OPTIMIZACION.asignadores import METODOS

from .formato import ACENTO, COLOR_METODO, SUAVE, TINTA

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


def generar_pngs(paquete: PaqueteReporte, carpeta: Path) -> dict[str, Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas: dict[str, Path] = {}

    # Frontera + nube + métodos
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    mc = paquete.monte_carlo.metricas
    sc = ax.scatter(mc["volatilidad"], mc["retorno"], c=mc["sharpe"], cmap="Blues", s=6, alpha=0.4)
    fr = paquete.frontera.puntos.sort_values("volatilidad")
    ax.plot(fr["volatilidad"], fr["retorno"], color=TINTA, lw=2, label="Frontera eficiente")
    for metodo in METODOS:
        m = paquete.asignaciones[metodo].metricas
        ax.scatter(m.volatilidad_anual, m.retorno_anual, s=70, marker="D",
                   color=COLOR_METODO.get(metodo, ACENTO), edgecolor="white", zorder=5, label=metodo)
    ax.set_title("Plano riesgo-retorno: frontera, nube y los 6 métodos")
    ax.set_xlabel("Volatilidad anual"); ax.set_ylabel("Retorno anual esperado")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(fontsize=7, loc="best")
    fig.colorbar(sc, ax=ax, label="Sharpe")
    rutas["frontera"] = _guardar(fig, carpeta, "frontera")

    # Equity OOS
    eq = paquete.riesgo.walk_forward.equity
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for metodo in eq.columns:
        ax.plot(eq.index, eq[metodo], lw=1.6, color=COLOR_METODO.get(metodo, SUAVE), label=metodo)
    ax.set_title("Curvas de capital out-of-sample (walk-forward, base 1.0)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Capital (×)")
    ax.legend(fontsize=7)
    rutas["equity"] = _guardar(fig, carpeta, "equity")

    # Drawdown
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for metodo in eq.columns:
        caida = eq[metodo] / eq[metodo].cummax() - 1.0
        ax.plot(caida.index, caida, lw=1.2, color=COLOR_METODO.get(metodo, SUAVE), label=metodo)
    ax.set_title("Curvas bajo el agua (drawdown OOS)")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(fontsize=7)
    rutas["drawdown"] = _guardar(fig, carpeta, "drawdown")

    # Correlación media + exceso de cola
    for nombre, matriz, cmap, titulo in (
        ("correlacion_media", paquete.analisis.correlacion_media, "RdBu_r", "Correlación media"),
        ("correlacion_cola", paquete.analisis.diferencia_correlacion_cola, "Reds", "Exceso de correlación en colas"),
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
    ax.bar(etiquetas, ve.values, color=ACENTO, alpha=0.85, label="Varianza explicada")
    ax2 = ax.twinx()
    ax2.plot(etiquetas, vac.values, color=TINTA, marker="o", lw=2, label="Acumulada")
    ax2.axhline(0.90, color=SUAVE, ls=":")
    ax2.set_ylim(0, 1.02)
    ax.set_title("PCA: factores independientes de la cesta")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.grid(False)
    rutas["pca"] = _guardar(fig, carpeta, "pca")

    # Pesos apilados
    activos = list(paquete.datos.activos)
    paleta = ["#1D4ED8", "#0E7490", "#15803D", "#B45309", "#BE123C", "#7C3AED", "#475569"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    izquierda = np.zeros(len(METODOS))
    for i, activo in enumerate(activos):
        valores = np.array([float(paquete.asignaciones[m].pesos[activo]) for m in METODOS])
        ax.barh(list(METODOS), valores, left=izquierda, color=paleta[i % len(paleta)], label=activo)
        izquierda += valores
    ax.set_title("Composición de cada cartera (pesos por activo)")
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
    ax.set_xticks(x); ax.set_xticklabels(div.index, rotation=20, ha="right", fontsize=7)
    ax.set_title("Número efectivo de apuestas: global vs. crisis")
    ax.set_ylabel("Apuestas independientes")
    ax.legend(fontsize=8)
    rutas["diversificacion"] = _guardar(fig, carpeta, "diversificacion")

    return rutas

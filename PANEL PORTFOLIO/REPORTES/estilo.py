"""Paleta, estilo de gráficos y formateadores compartidos del informe.

Estética institucional sobria: tinta azul-pizarra, acento azul, rojo para
pérdidas/cola, mucho aire en blanco. Sin chartjunk. Los mismos colores los usan
el HTML y el PDF para que ambos informes sean idénticos.
"""

from __future__ import annotations

# --- Paleta -----------------------------------------------------------------
TINTA = "#0F172A"        # azul-pizarra oscuro (texto, ejes)
SUAVE = "#475569"        # gris medio (texto secundario)
LINEA = "#D9DEE7"        # líneas/bordes finos
ACENTO = "#1D4ED8"       # azul institucional
ACENTO_CLARO = "#93B4FF"
PANEL = "#F6F8FB"        # fondo de tarjetas
NEG = "#B91C1C"          # rojo (pérdidas, VaR, cola)
VERDE = "#15803D"        # verde (positivo, recomendada)
AMBAR = "#B45309"        # avisos

# Colores cualitativos para activos (hasta 10).
SERIE = ["#1D4ED8", "#0EA5E9", "#7C3AED", "#059669", "#D97706",
         "#DB2777", "#0891B2", "#65A30D", "#9333EA", "#E11D48"]

NIVEL_COLOR = {
    "bajo": "#0EA5E9",
    "medio": "#1D4ED8",
    "alto": "#7C3AED",
    "max_sharpe": "#059669",
}


def aplicar_estilo() -> None:
    """Configura matplotlib con el estilo institucional (idempotente)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.edgecolor": LINEA,
        "axes.labelcolor": SUAVE,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": LINEA,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,
        "xtick.color": SUAVE,
        "ytick.color": SUAVE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })


# --- Formateadores ----------------------------------------------------------
def pct(x: float, dec: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.{dec}f}%"


def num(x: float, dec: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:.{dec}f}"


def dinero(x: float) -> str:
    return f"€{x:,.0f}"


NIVEL_NOMBRE = {
    "es": {"bajo": "Bajo", "medio": "Medio", "alto": "Alto", "max_sharpe": "Máx Sharpe"},
    "it": {"bajo": "Basso", "medio": "Medio", "alto": "Alto", "max_sharpe": "Max Sharpe"},
}


def nombre_nivel(nivel: str, idioma: str = "es") -> str:
    return NIVEL_NOMBRE.get(idioma, NIVEL_NOMBRE["es"]).get(nivel, nivel)

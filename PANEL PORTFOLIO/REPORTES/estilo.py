"""Paleta, estilo de gráficos y formateadores compartidos del informe.

Estética institucional minimalista: fondo blanco puro, líneas finísimas,
texto en negrita, colores grises suaves. Los gráficos mantienen color pero
el marco (tablas, KPIs, bordes) es sobrio y limpio.
"""

from __future__ import annotations

# --- Paleta minimalista -----------------------------------------------------
TINTA = "#111827"        # gris muy oscuro (texto principal)
SUAVE = "#6B7280"        # gris medio (texto secundario)
MUTED = "#9CA3AF"        # gris claro (notas al pie)
LINEA = "#E5E7EB"        # bordes finos, líneas de tabla
LINEA_FINA = "#F3F4F6"   # separadores ultra-sutiles

ACENTO = "#1D4ED8"       # azul institucional (sparingly)
ACENTO_CLARO = "#93B4FF"
PANEL = "#FFFFFF"        # fondo de tarjetas = blanco puro
FONDO_NOTA = "#F9FAFB"   # fondo sutil para notas
NEG = "#B91C1C"          # rojo (pérdidas, VaR, cola)
VERDE = "#15803D"        # verde (positivo, recomendada)
AMBAR = "#B45309"        # avisos

# Headers de tabla: gris oscuro (no navy)
HEADER_BG = "#374151"
HEADER_FG = "#FFFFFF"

# Colores cualitativos para activos (hasta 10).
SERIE = ["#1D4ED8", "#0EA5E9", "#7C3AED", "#059669", "#D97706",
         "#DB2777", "#0891B2", "#65A30D", "#9333EA", "#E11D48"]

NIVEL_COLOR = {
    "bajo": "#0EA5E9",
    "medio": "#1D4ED8",
    "alto": "#7C3AED",
    "max_sharpe": "#059669",
    "max_k_ratio": "#15803D",
    "markowitz": "#059669",
    "cvar": "#B91C1C",
    "nco": "#7C3AED",
}

MOTOR_COLOR = {
    "MARKOWITZ": "#1D4ED8",
    "CVAR": "#B91C1C",
    "NCO": "#059669",
}


def aplicar_estilo() -> None:
    """Configura matplotlib con el estilo minimalista (idempotente)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "font.weight": "normal",
        "axes.edgecolor": LINEA,
        "axes.labelcolor": SUAVE,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.linewidth": 0.5,
        "grid.color": LINEA,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        "xtick.color": SUAVE,
        "ytick.color": SUAVE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": 9,
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
    "es": {
        "bajo": "Bajo",
        "medio": "Medio",
        "alto": "Alto",
        "max_sharpe": "Máx Sharpe",
        "max_k_ratio": "Máx K-Ratio",
        "markowitz": "Markowitz",
        "cvar": "Min-CVaR",
        "nco": "NCO",
    },
    "it": {
        "bajo": "Basso",
        "medio": "Medio",
        "alto": "Alto",
        "max_sharpe": "Max Sharpe",
        "max_k_ratio": "Max K-Ratio",
        "markowitz": "Markowitz",
        "cvar": "Min-CVaR",
        "nco": "NCO",
    },
}


def nombre_nivel(nivel: str, idioma: str = "es") -> str:
    return NIVEL_NOMBRE.get(idioma, NIVEL_NOMBRE["es"]).get(nivel, nivel)

"""Formato de cifras, paleta institucional, aviso de honestidad y glosario.

Sin lógica de negocio: solo presentación. Lo usan el HTML, el PDF y el Excel
para que las tres salidas hablen el mismo idioma.
"""

from __future__ import annotations

import math

# Paleta sobria de informe institucional.
TINTA = "#0F172A"
SUAVE = "#475569"
LINEA = "#D9DEE7"
ACENTO = "#1D4ED8"
POSITIVO = "#15803D"
NEGATIVO = "#B91C1C"
FONDO = "#FFFFFF"
PANEL = "#F6F8FB"

# Color estable por método (mismo en HTML, PDF y Excel).
COLOR_METODO = {
    "Markowitz (máx Sharpe)": "#1D4ED8",
    "Mínima varianza": "#0E7490",
    "Risk parity": "#15803D",
    "HRP": "#7C3AED",
    "Min-CVaR": "#B45309",
    "Black-Litterman": "#BE123C",
    "Máxima diversificación": "#0F766E",
}

# Objetivos que el usuario elige al arrancar; el informe se centra en el elegido.
OBJETIVOS = {
    "sharpe": "Máximo ratio de Sharpe",
    "riesgo": "Mínimo riesgo",
    "objetivo": "Retorno objetivo",
    "convexidad": "Convexidad / anti-caída",
    "comparar": "Comparar los métodos",
}


def nombre_visible(metodo: str, paquete=None) -> str:
    """Nombre para mostrar. Aclara que Black-Litterman sin views es, de hecho, 1/N."""
    if metodo == "Black-Litterman" and paquete is not None:
        bl = paquete.asignaciones.get("Black-Litterman")
        if bl is not None and "sin views" in (bl.diagnostico or "").lower():
            return "Black-Litterman (sin views ⇒ 1/N)"
    return metodo


def pct(x: float, dec: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x * 100:.{dec}f}%"


def num(x: float, dec: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{dec}f}"


def fecha(ts) -> str:
    try:
        return ts.date().isoformat()
    except AttributeError:
        return str(ts)


AVISO_HONESTIDAD = (
    "Este informe es ANÁLISIS DESCRIPTIVO del comportamiento pasado de la cesta en "
    "regímenes pasados. No es una predicción ni una recomendación de inversión, y no "
    "garantiza protección alguna en la próxima crisis: las correlaciones, volatilidades "
    "y retornos cambian con el tiempo. El backtest es walk-forward (out-of-sample) y, "
    "aun así, un buen resultado histórico no asegura resultados futuros."
)

# Glosario: una frase llana por término técnico.
GLOSARIO = {
    "Retorno esperado": "Rentabilidad media anual que cabría esperar según la historia (no una promesa).",
    "Volatilidad": "Cuánto oscila la cartera; más volatilidad = más sustos.",
    "Sharpe": "Rentabilidad por unidad de riesgo total. Cuanto más alto, mejor pagado está el riesgo.",
    "Sortino": "Como el Sharpe pero penalizando solo las caídas, no las subidas.",
    "Calmar": "Rentabilidad anual dividida por el peor desplome; mide cuánto ganas por cada unidad de dolor.",
    "Max drawdown": "La mayor caída desde un máximo hasta el siguiente valle. El peor momento de la curva.",
    "VaR 95%": "Pérdida diaria que solo se supera el 5% de los días (histórico, sin suponer normalidad).",
    "CVaR 95%": "Pérdida media en ese 5% de peores días; lo que duele cuando duele.",
    "Ledoit-Wolf": "Covarianza 'encogida' que estabiliza la estimación y evita pesos extremos.",
    "Correlación de cola": "Correlación medida solo en los peores días del mercado; revela la diversificación que se evapora en las caídas.",
    "Número efectivo de apuestas": "Cuántas apuestas realmente independientes tienes (no cuántos activos).",
    "Walk-forward": "Estimar pesos con el pasado y medirlos en el futuro no visto: el único backtest honesto.",
    "Máxima diversificación": "Cartera que premia activos que se mueven en sentidos opuestos: si algo baja y algo sube, se compensan.",
    "Captura alcista": "Cuánto sigue la cartera al mercado cuando sube (1 = igual; >1 = sube más).",
    "Captura bajista": "Cuánto sigue la cartera al mercado cuando baja (cuanto menor, más protegida).",
    "Asimetría (convexidad)": "Captura alcista menos bajista; positiva = sube más de lo que baja (perfil deseable).",
    "Equiponderada (1/N)": "Repartir igual entre todos los activos; sorprendentemente difícil de batir fuera de muestra.",
}

"""Tabla maestra: 6 métodos × (pesos por activo + métricas).

Distingue claramente las métricas IN-SAMPLE (esperadas, estimadas con la ventana
actual) de las OUT-OF-SAMPLE (realizadas en el walk-forward), porque solo las
segundas son un juicio honesto.
"""

from __future__ import annotations

import pandas as pd

from CONTRATOS.modelos import PaqueteReporte
from OPTIMIZACION.asignadores import METODOS


def tabla_maestra(paquete: PaqueteReporte) -> pd.DataFrame:
    activos = list(paquete.datos.activos)
    filas: dict[str, dict] = {}
    for metodo in METODOS:
        asignacion = paquete.asignaciones[metodo]
        oos = paquete.riesgo.metricas[metodo]
        fila: dict[str, float] = {f"peso · {a}": float(asignacion.pesos[a]) for a in activos}
        fila.update({
            "Retorno esperado (in-sample)": asignacion.metricas.retorno_anual,
            "Volatilidad esperada (in-sample)": asignacion.metricas.volatilidad_anual,
            "Sharpe esperado (in-sample)": asignacion.metricas.sharpe,
            "Retorno anual (OOS)": oos.retorno_anual,
            "Volatilidad (OOS)": oos.volatilidad_anual,
            "Sharpe (OOS)": oos.sharpe,
            "Sortino (OOS)": oos.sortino,
            "Calmar (OOS)": oos.calmar,
            "Max drawdown (OOS)": oos.max_drawdown,
            "VaR 95% (OOS)": oos.var,
            "CVaR 95% (OOS)": oos.cvar,
        })
        filas[metodo] = fila
    return pd.DataFrame.from_dict(filas, orient="index")


def tabla_espejismo(paquete: PaqueteReporte) -> pd.DataFrame:
    """Lo que cada método ESPERABA (in-sample) frente a lo que fue REAL (OOS).

    La columna 'degradación' es el desplome del Sharpe al pasar al futuro no visto:
    es la medida directa del espejismo. Ordenada por la promesa in-sample.
    """
    filas: dict[str, dict] = {}
    for metodo in METODOS:
        ins = paquete.asignaciones[metodo].metricas
        oos = paquete.riesgo.metricas[metodo]
        filas[metodo] = {
            "Sharpe esperado (in-sample)": ins.sharpe,
            "Sharpe realizado (OOS)": oos.sharpe,
            "Degradación de Sharpe": oos.sharpe - ins.sharpe,
            "Retorno esperado (in-sample)": ins.retorno_anual,
            "Retorno realizado (OOS)": oos.retorno_anual,
        }
    df = pd.DataFrame.from_dict(filas, orient="index")
    return df.sort_values("Sharpe esperado (in-sample)", ascending=False)


def tabla_convexidad(paquete: PaqueteReporte) -> pd.DataFrame:
    """Comportamiento OOS por escenario y capturas alza/baja (perfil anti-caída)."""
    conv = paquete.riesgo.convexidad
    columnas = {
        "ret_medio_todo_baja": "Ret. medio · todo baja",
        "ret_medio_mixto": "Ret. medio · mixto",
        "ret_medio_todo_sube": "Ret. medio · todo sube",
        "captura_alcista": "Captura alcista",
        "captura_bajista": "Captura bajista",
        "asimetria": "Asimetría",
    }
    presentes = [c for c in columnas if c in conv.columns]
    tabla = conv[presentes].rename(columns=columnas)
    return tabla.reindex(METODOS)

"""Tablas del informe.

La tabla MAESTRA muestra SOLO pesos + métricas OUT-OF-SAMPLE (lo realizado), que
es el único juicio honesto. La promesa in-sample se separa en `tabla_espejismo`
("promesa vs realidad"), para no mezclar lo esperado con lo cumplido.
"""

from __future__ import annotations

import pandas as pd

from CONTRATOS.modelos import PaqueteReporte
from OPTIMIZACION.asignadores import metodos
from REPORTES.i18n import perfil_visible


def tabla_maestra(paquete: PaqueteReporte) -> pd.DataFrame:
    """Pesos por activo + métricas OUT-OF-SAMPLE (sin columnas in-sample)."""
    activos = list(paquete.datos.activos)
    filas: dict[str, dict] = {}
    for metodo in metodos(paquete.configuracion):
        asignacion = paquete.asignaciones[metodo]
        oos = paquete.riesgo.metricas[metodo]
        fila: dict[str, float] = {f"peso · {a}": float(asignacion.pesos[a]) for a in activos}
        fila.update({
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
    """Promesa (in-sample) frente a realidad (OOS).

    La columna 'degradación' es el desplome del Sharpe al pasar al futuro no visto:
    la medida directa del espejismo. Ordenada por la promesa in-sample.
    """
    filas: dict[str, dict] = {}
    for metodo in metodos(paquete.configuracion):
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


def tabla_pesos_niveles(paquete: PaqueteReporte) -> pd.DataFrame:
    """Pesos eficientes por nivel de riesgo, de conservador a agresivo.

    Es la tabla que acompaña al resultado principal: muestra cómo cambian los
    pesos al desplazarse por la frontera eficiente.
    """
    activos = list(paquete.datos.activos)
    filas: dict[str, dict[str, float]] = {}
    for cartera in paquete.perfil_riesgo.carteras:
        etiqueta = perfil_visible(paquete, cartera.nivel)
        filas[etiqueta] = {
            "Retorno esperado": cartera.retorno_esperado,
            "Volatilidad esperada": cartera.volatilidad_esperada,
            "VaR histórico": cartera.metricas_historicas.var,
            "CVaR histórico": cartera.metricas_historicas.cvar,
            "Max drawdown histórico": cartera.metricas_historicas.max_drawdown,
            **{f"peso · {activo}": float(cartera.pesos[activo]) for activo in activos},
        }
    return pd.DataFrame.from_dict(filas, orient="index")


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
    orden = [m for m in metodos(paquete.configuracion) if m in conv.index]
    return conv[presentes].rename(columns=columnas).reindex(orden)

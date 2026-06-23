"""Backtest WALK-FORWARD: la pieza más importante del panel.

En cada fecha de rebalanceo se estiman los pesos de cada método usando SOLO el
pasado (la ventana de días anterior a ese día) y se aplican al FUTURO inmediato
(hasta el siguiente rebalanceo), dejando que los pesos deriven con los precios.
Se desliza mes a mes.

Por qué es honesto y evita el espejismo
---------------------------------------
Si se midieran las carteras sobre la MISMA muestra con la que se estimaron,
Markowitz "ganaría" casi siempre: maximiza el Sharpe in-sample ajustándose al
ruido de ese tramo concreto. Pero ese ajuste no se repite en datos nuevos. Al
estimar en el pasado y medir en el futuro no visto, las curvas comparadas son
OUT-OF-SAMPLE: reflejan lo que un inversor habría obtenido de verdad, donde la
ventaja in-sample de Markowitz se desinfla y métodos robustos (HRP, risk parity,
mín-varianza) suelen comportarse mejor de lo que su Sharpe in-sample sugería.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CONTRATOS.errores import ErrorRiesgo
from CONTRATOS.modelos import (
    Configuracion,
    DatosAlineados,
    Rebalanceo,
    ResultadoWalkForward,
)
from OPTIMIZACION.asignadores import METODOS, calcular_pesos

_FREQ_PERIODO = {"M": "M", "W": "W", "Q": "Q"}


def ejecutar_walk_forward(datos: DatosAlineados, configuracion: Configuracion) -> ResultadoWalkForward:
    log_retornos = datos.log_retornos
    retornos_simples = np.expm1(log_retornos)          # de log a simple, aditivos por activo
    fechas = retornos_simples.index
    ventana = configuracion.ventana_estimacion
    activos = list(log_retornos.columns)

    if len(fechas) <= ventana + 21:
        raise ErrorRiesgo(
            f"Histórico insuficiente para walk-forward: {len(fechas)} ≤ ventana {ventana} + 1 mes."
        )

    fechas_oos = fechas[ventana:]                       # cada una tiene ≥ ventana de pasado
    periodo = fechas_oos.to_period(_FREQ_PERIODO[configuracion.frecuencia_rebalanceo])
    es_rebalanceo = ~pd.Series(periodo, index=fechas_oos).duplicated().to_numpy()

    pesos_actuales: dict[str, pd.Series] = {}
    retornos: dict[str, list[float]] = {m: [] for m in METODOS}
    pesos_diarios: dict[str, list[pd.Series]] = {m: [] for m in METODOS}
    rebalanceos: list[Rebalanceo] = []
    coste_unitario = configuracion.coste_transaccion_pb / 10_000.0

    for posicion, fecha in enumerate(fechas_oos):
        coste_hoy = {m: 0.0 for m in METODOS}

        if es_rebalanceo[posicion] or not pesos_actuales:
            ventana_ret = log_retornos.loc[log_retornos.index < fecha].iloc[-ventana:]
            if len(ventana_ret) < ventana:
                continue   # aún sin ventana completa (no debería ocurrir en OOS)
            objetivos = calcular_pesos(ventana_ret, configuracion)
            pesos_reb: dict[str, pd.Series] = {}
            rotacion_reb: dict[str, float] = {}
            for metodo in METODOS:
                objetivo = objetivos[metodo].reindex(activos)
                previo = pesos_actuales.get(metodo, pd.Series(0.0, index=activos))
                rotacion = 0.5 * float((objetivo - previo).abs().sum())
                coste_hoy[metodo] = rotacion * coste_unitario
                pesos_actuales[metodo] = objetivo
                pesos_reb[metodo] = objetivo
                rotacion_reb[metodo] = rotacion
            rebalanceos.append(
                Rebalanceo(
                    fecha=fecha,
                    pesos=pesos_reb,
                    rotacion=rotacion_reb,
                    coste={m: rotacion_reb[m] * coste_unitario for m in METODOS},
                )
            )

        r_activos = retornos_simples.loc[fecha]
        for metodo in METODOS:
            w = pesos_actuales[metodo]
            bruto = float((w * r_activos).sum())
            retornos[metodo].append(bruto - coste_hoy[metodo])
            pesos_diarios[metodo].append(w.copy())
            # Deriva de pesos hasta el día siguiente (buy & hold dentro del mes).
            pesos_actuales[metodo] = (w * (1.0 + r_activos)) / (1.0 + bruto)

    indice = fechas_oos[: len(retornos[METODOS[0]])]
    retornos_df = pd.DataFrame({m: retornos[m] for m in METODOS}, index=indice)
    equity_df = (1.0 + retornos_df).cumprod()
    pesos_diarios_df = {
        m: pd.DataFrame(pesos_diarios[m], index=indice) for m in METODOS
    }
    if retornos_df.empty:
        raise ErrorRiesgo("El walk-forward no produjo retornos out-of-sample.")
    return ResultadoWalkForward(
        retornos=retornos_df,
        equity=equity_df,
        pesos_diarios=pesos_diarios_df,
        rebalanceos=tuple(rebalanceos),
    )

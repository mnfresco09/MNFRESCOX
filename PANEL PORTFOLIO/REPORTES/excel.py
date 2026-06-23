"""Libro Excel del informe: tabla maestra + hojas de detalle, con formato sobrio.

Pensado para que un analista pueda reordenar, filtrar y reutilizar los números.
Marca claramente in-sample vs OOS y repite el aviso de honestidad.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from CONTRATOS.modelos import PaqueteReporte

from .formato import AVISO_HONESTIDAD
from .tablas import tabla_maestra

_PCT_COLS = {
    "Retorno esperado (in-sample)", "Volatilidad esperada (in-sample)",
    "Retorno anual (OOS)", "Volatilidad (OOS)", "Max drawdown (OOS)",
    "VaR 95% (OOS)", "CVaR 95% (OOS)",
}


def generar_excel(paquete: PaqueteReporte, ruta: Path) -> Path:
    cfg = paquete.configuracion
    maestra = tabla_maestra(paquete)

    with pd.ExcelWriter(ruta, engine="xlsxwriter") as writer:
        wb = writer.book
        f_titulo = wb.add_format({"bold": True, "font_size": 18, "font_color": "#0F172A"})
        f_sub = wb.add_format({"font_size": 11, "font_color": "#475569"})
        f_aviso = wb.add_format({"text_wrap": True, "font_size": 10, "font_color": "#7C2D12",
                                 "bg_color": "#FFF7ED", "border": 1, "valign": "top"})
        f_h = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#0F172A",
                             "border": 1, "align": "center", "valign": "vcenter", "text_wrap": True})
        f_metodo = wb.add_format({"bold": True, "bg_color": "#F6F8FB", "border": 1})
        f_pct = wb.add_format({"num_format": "0.00%", "border": 1})
        f_num = wb.add_format({"num_format": "0.00", "border": 1})

        # --- Portada --------------------------------------------------------
        ws = wb.add_worksheet("Resumen")
        ws.set_column("A:A", 100)
        ws.write("A1", "PANEL PORTFOLIO — Informe de optimización", f_titulo)
        ws.write("A2", f"Cesta: {', '.join(cfg.tickers)}", f_sub)
        ws.write("A3", f"Periodo: {cfg.fecha_inicio} a {cfg.fecha_fin} · rebalanceo {cfg.frecuencia_rebalanceo} · ventana {cfg.ventana_estimacion} días", f_sub)
        ws.merge_range("A5:A9", AVISO_HONESTIDAD, f_aviso)

        # --- Tabla maestra --------------------------------------------------
        ws = wb.add_worksheet("Tabla maestra")
        ws.freeze_panes(1, 1)
        ws.set_column(0, 0, 26)
        ws.set_column(1, len(maestra.columns), 16)
        ws.write(0, 0, "Método", f_h)
        for j, col in enumerate(maestra.columns, start=1):
            ws.write(0, j, col, f_h)
        for i, (metodo, fila) in enumerate(maestra.iterrows(), start=1):
            ws.write(i, 0, metodo, f_metodo)
            for j, col in enumerate(maestra.columns, start=1):
                fmt = f_pct if (col.startswith("peso ·") or col in _PCT_COLS) else f_num
                ws.write_number(i, j, float(fila[col]), fmt)

        # --- Hojas de detalle ----------------------------------------------
        paquete_metricas = pd.DataFrame({
            m: {
                "retorno_anual": v.retorno_anual, "volatilidad_anual": v.volatilidad_anual,
                "sharpe": v.sharpe, "sortino": v.sortino, "calmar": v.calmar,
                "max_drawdown": v.max_drawdown, "var": v.var, "cvar": v.cvar,
                "duracion_drawdown_dias": v.duracion_drawdown_dias,
            } for m, v in paquete.riesgo.metricas.items()
        }).T
        paquete_metricas.to_excel(writer, sheet_name="Riesgo OOS")

        paquete.riesgo.diversificacion_crisis.to_excel(writer, sheet_name="Diversificacion")

        # Métricas por régimen: una hoja apilada.
        bloques = []
        for metodo, df in paquete.riesgo.metricas_por_regimen.items():
            tmp = df.copy()
            tmp.insert(0, "metodo", metodo)
            tmp.index.name = "regimen"
            bloques.append(tmp.reset_index())
        if bloques:
            pd.concat(bloques, ignore_index=True).to_excel(writer, sheet_name="Por regimen", index=False)

        # Stress.
        filas_stress = []
        for nombre, res in paquete.riesgo.stress.items():
            if res.evaluable:
                for metodo, m in res.metricas.items():
                    filas_stress.append({
                        "episodio": nombre, "metodo": metodo, "dias": res.observaciones,
                        "retorno_anual": m.retorno_anual, "volatilidad_anual": m.volatilidad_anual,
                        "max_drawdown": m.max_drawdown, "cvar": m.cvar,
                    })
            else:
                filas_stress.append({"episodio": nombre, "metodo": "—",
                                     "dias": res.observaciones, "retorno_anual": None,
                                     "volatilidad_anual": None, "max_drawdown": None, "cvar": None})
        pd.DataFrame(filas_stress).to_excel(writer, sheet_name="Stress", index=False)

    return ruta

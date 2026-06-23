"""Orquestación del pipeline en pasos discretos (para progreso y reutilización).

Encadena las capas respetando las dependencias de una sola dirección y el
principio fail-fast. Se importa DESPUÉS del arranque del entorno, por eso aquí sí
se pueden importar con normalidad las capas pesadas.

Cada paso es (etiqueta, función, peso_segundos_estimado). La función recibe un
contexto mutable y un 'log' para mensajes. El peso (en segundos aproximados,
medidos sobre la cesta de referencia) permite estimar el tiempo restante.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ANALISIS.analisis import analizar
from CONTRATOS.modelos import Configuracion, PaqueteReporte
from DATOS.alineacion import alinear_y_calcular_retornos, recortar_datos
from DATOS.cargador import cargar_cierres
from DESCARGADOR.cache import asegurar_datos
from DESCARGADOR.descargador import imprimir_resumen
from OPTIMIZACION.asignadores import asignar_todos
from OPTIMIZACION.frontera import construir_frontera
from OPTIMIZACION.montecarlo import nube_montecarlo
from REPORTES.excel import generar_excel
from REPORTES.html import generar_html
from REPORTES.pdf import generar_pdf
from RIESGO.perfil import evaluar_perfil
from RIESGO.riesgo import evaluar_riesgo


def _paso_datos(ctx: dict, log) -> None:
    cfg = ctx["cfg"]
    estado, resumenes = asegurar_datos(cfg)
    if estado == "descarga":
        log("La cesta o las fechas cambiaron: histórico actualizado.")
        if resumenes is not None:
            imprimir_resumen(resumenes)
    else:
        log("Histórico vigente: se reutiliza lo ya descargado.")


def _paso_cargar(ctx: dict, log) -> None:
    cfg = ctx["cfg"]
    cierres = cargar_cierres(cfg.tickers, cfg.carpeta_historico)
    ctx["datos"] = alinear_y_calcular_retornos(cierres, cfg.min_retornos_analisis)
    log(f"{len(ctx['datos'].log_retornos):,} retornos diarios comunes tras alinear calendarios.")


def _paso_analisis(ctx: dict, log) -> None:
    cfg, datos = ctx["cfg"], ctx["datos"]
    ctx["analisis"] = analizar(datos, cfg)
    ctx["datos_actual"] = recortar_datos(datos, cfg.ventana_estimacion)
    ctx["analisis_actual"] = analizar(ctx["datos_actual"], cfg)


def _paso_asignar(ctx: dict, log) -> None:
    ctx["asignaciones"] = asignar_todos(ctx["analisis_actual"], ctx["cfg"])


def _paso_frontera(ctx: dict, log) -> None:
    aa = ctx["analisis_actual"]
    ctx["frontera"] = construir_frontera(aa.retornos_esperados, aa.covarianza, ctx["cfg"])


def _paso_montecarlo(ctx: dict, log) -> None:
    cfg, aa = ctx["cfg"], ctx["analisis_actual"]
    ctx["monte_carlo"] = nube_montecarlo(
        aa.retornos_esperados, aa.covarianza, cfg.restricciones,
        cfg.n_carteras_montecarlo, cfg.semilla, cfg.tasa_libre_riesgo_anual,
    )


def _paso_perfil(ctx: dict, log) -> None:
    ctx["perfil_riesgo"] = evaluar_perfil(ctx["datos"], ctx["frontera"], ctx["cfg"])
    carteras = ctx["perfil_riesgo"].carteras
    log("Pesos por nivel de riesgo: " + ", ".join(
        f"{c.nivel} (vol {c.volatilidad_esperada:.1%})" for c in carteras))


def _paso_riesgo(ctx: dict, log) -> None:
    ctx["riesgo"] = evaluar_riesgo(ctx["datos"], ctx["analisis"], ctx["cfg"])
    ctx["paquete"] = PaqueteReporte(
        configuracion=ctx["cfg"], datos=ctx["datos"], analisis=ctx["analisis"],
        analisis_actual=ctx["analisis_actual"], asignaciones=ctx["asignaciones"],
        frontera=ctx["frontera"], monte_carlo=ctx["monte_carlo"], riesgo=ctx["riesgo"],
        perfil_riesgo=ctx["perfil_riesgo"],
        objetivo=ctx.get("objetivo", "comparar"),
    )


def _carpeta(ctx: dict) -> Path:
    carpeta = Path(ctx["cfg"].carpeta_salidas)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def _paso_html(ctx: dict, log) -> None:
    ctx["rutas"]["html"] = generar_html(ctx["paquete"], _carpeta(ctx) / "informe.html")


def _paso_pdf(ctx: dict, log) -> None:
    ctx["rutas"]["pdf"] = generar_pdf(ctx["paquete"], _carpeta(ctx) / "informe.pdf")


def _paso_excel(ctx: dict, log) -> None:
    carpeta = _carpeta(ctx)
    ctx["rutas"]["excel"] = generar_excel(ctx["paquete"], carpeta / "informe.xlsx")
    cfg = ctx["cfg"]
    manifiesto = carpeta / "manifiesto_reporte.json"
    manifiesto.write_text(json.dumps({
        "generado_en": datetime.now().isoformat(timespec="seconds"),
        "tickers": list(cfg.tickers), "fecha_inicio": cfg.fecha_inicio, "fecha_fin": cfg.fecha_fin,
        "idioma_reporte": cfg.idioma_reporte,
        "retornos_comunes": int(len(ctx["datos"].log_retornos)),
        "archivos": {"html": "informe.html", "pdf": "informe.pdf", "excel": "informe.xlsx"},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx["rutas"]["manifiesto"] = manifiesto


# (etiqueta, función, peso ≈ segundos sobre la cesta de referencia)
PASOS = (
    ("Preparando datos (caché o descarga)", _paso_datos, 1.0),
    ("Cargando y alineando cierres", _paso_cargar, 1.0),
    ("Análisis: momentos, covarianza, PCA y regímenes", _paso_analisis, 1.0),
    ("Asignando las 6 carteras", _paso_asignar, 1.0),
    ("Construyendo la frontera eficiente", _paso_frontera, 2.0),
    ("Simulación Monte Carlo", _paso_montecarlo, 1.0),
    ("Pesos por nivel de riesgo", _paso_perfil, 1.0),
    ("Backtest walk-forward y métricas de riesgo", _paso_riesgo, 22.0),
    ("Informe HTML interactivo", _paso_html, 30.0),
    ("Informe PDF", _paso_pdf, 9.0),
    ("Libro Excel y manifiesto", _paso_excel, 1.0),
)


def construir_paquete(configuracion: Configuracion, objetivo: str = "comparar") -> PaqueteReporte:
    """Versión no interactiva: encadena los pasos de cálculo y devuelve el paquete."""
    ctx: dict = {"cfg": configuracion, "objetivo": objetivo, "rutas": {}}
    for _, funcion, _ in PASOS:
        if funcion in (_paso_html, _paso_pdf, _paso_excel):
            break
        funcion(ctx, lambda *_: None)
    return ctx["paquete"]

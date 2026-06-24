"""Orquestador del MOTOR DE RIESGO PREDICTIVO (flujo vectorial/probabilístico).

Flujo objetivo (una sola dirección, fail-fast):

  Activos → Log Returns → Estadística Individual → Doble Lente de Covarianza →
  Frontera Eficiente Restringida → Selección Automática de Carteras
  (Bajo/Medio/Alto/Máx Sharpe) → Riesgo Histórico y Forecast (VaR/CVaR/FHS) →
  Simulación Futura Agregada (Monte Carlo, fan chart, CDaR) → Score Final →
  Recomendación Ejecutiva.

Cada paso recibe un contexto mutable y un `log`. El resultado es un
`ReportPayload` que el reporting consume SIN recalcular.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from ANALISIS.analisis import calcular_momentos
from CONTRATOS.modelos import (
    Configuracion,
    PortfolioInput,
    Recomendacion,
    ReportPayload,
)
from DATOS.alineacion import alinear_y_calcular_retornos
from DATOS.cargador import cargar_cierres
from DESCARGADOR.cache import asegurar_datos
from OPTIMIZACION.frontera import construir_frontera
from OPTIMIZACION.perfiles import curva_top_sharpe, seleccionar_perfiles
from RIESGO.exploracion_riesgo import construir_exploracion
from RIESGO.forecast import calcular_forecast, calcular_simulacion
from RIESGO.regimen import detectar_regimen
from RIESGO.score import calcular_score_cartera


def _paso_datos(ctx: dict, log) -> None:
    cfg = ctx["cfg"]
    estado, _ = asegurar_datos(cfg)
    log("Histórico actualizado." if estado == "descarga" else "Histórico vigente reutilizado.")


def _paso_entrada(ctx: dict, log) -> None:
    cfg = ctx["cfg"]
    cierres = cargar_cierres(cfg.tickers, cfg.carpeta_historico)
    datos = alinear_y_calcular_retornos(cierres, cfg.min_retornos_analisis)
    ctx["entrada"] = PortfolioInput(
        activos=datos.activos,
        log_retornos=datos.log_retornos,
        cierres=datos.cierres,
        capital_base=cfg.capital_base,
        horizonte_dias=cfg.horizonte_dias,
    )
    log(f"{len(datos.log_retornos):,} retornos diarios comunes tras alinear calendarios.")


def _paso_momentos(ctx: dict, log) -> None:
    ctx["momentos"] = calcular_momentos(ctx["entrada"], ctx["cfg"])
    m = ctx["momentos"]
    log(f"Doble lente lista (Ledoit-Wolf shrink={m.shrinkage_cov:.2f}, táctica {m.fuente_tactica}).")


def _paso_frontera(ctx: dict, log) -> None:
    m = ctx["momentos"]
    ctx["frontera"] = construir_frontera(m.retornos_ajustados, m.cov_estructural, ctx["cfg"])
    log(f"Frontera restringida con {len(ctx['frontera'].puntos)} puntos eficientes.")


def _paso_perfiles(ctx: dict, log) -> None:
    ctx["candidatos"] = seleccionar_perfiles(ctx["frontera"], ctx["momentos"], ctx["cfg"])
    ctx["curva_top_sharpe"] = curva_top_sharpe(ctx["frontera"])
    log("Perfiles dinámicos: " + ", ".join(
        f"{c.nivel} (vol T+1 {c.volatilidad_tactica:.1%})" for c in ctx["candidatos"]))


def _paso_riesgo(ctx: dict, log) -> None:
    cfg = ctx["cfg"]
    log_ret = ctx["entrada"].log_retornos
    actualizados = []
    fuente = "rust"
    for c in ctx["candidatos"]:
        fc = calcular_forecast(c.pesos, log_ret, ctx["momentos"], cfg)
        sim = calcular_simulacion(c.pesos, log_ret, cfg)
        fuente = sim.fuente
        actualizados.append(replace(c, forecast=fc, simulacion=sim))
    ctx["candidatos"] = tuple(actualizados)
    log(f"Forecast VaR/CVaR + simulación a {cfg.horizonte_dias}d lista (motor: {fuente}).")


def _paso_regimen(ctx: dict, log) -> None:
    ctx["regimen"] = detectar_regimen(ctx["entrada"], ctx["momentos"], ctx["cfg"])
    log(f"Régimen detectado: {ctx['regimen'].etiqueta}.")


def _paso_score(ctx: dict, log) -> None:
    ctx["candidatos"] = calcular_score_cartera(ctx["candidatos"], ctx["cfg"])
    ganadora = max(ctx["candidatos"], key=lambda c: c.score if c.score is not None else -1e18)
    ctx["recomendada"] = ganadora
    ctx["recomendacion"] = _recomendar(ganadora, ctx["regimen"])
    log(f"Cartera recomendada: {ganadora.nivel} (score {ganadora.score:.2f}).")


def _paso_exploracion(ctx: dict, log) -> None:
    ctx["exploracion"] = construir_exploracion(
        ctx["frontera"], ctx["momentos"], ctx["entrada"], ctx["cfg"], ctx["candidatos"]
    )
    n = sum(len(c.top) for c in ctx["exploracion"]["leaderboard"])
    log(f"Exploración multi-criterio: {len(ctx['exploracion']['leaderboard'])} criterios, {n} carteras en el leaderboard.")


def _recomendar(ganadora, regimen) -> Recomendacion:
    detalle = (
        f"«{ganadora.nivel}» maximiza el score bajo el régimen actual "
        f"({regimen.etiqueta}): retorno esperado {ganadora.retorno_esperado:.1%}, "
        f"vol T+1 {ganadora.volatilidad_tactica:.1%}, "
        f"VaR 99% FHS diario {ganadora.forecast.var_fhs_99:.2%}."
    )
    return Recomendacion(
        nivel=ganadora.nivel,
        criterio="mayor score (Sharpe táctico penalizado por VaR, CDaR, concentración y turnover)",
        detalle=detalle,
    )


PASOS = (
    ("Preparando datos (caché o descarga)", _paso_datos),
    ("Activos y log-retornos alineados", _paso_entrada),
    ("Estadística individual y doble lente de covarianza", _paso_momentos),
    ("Frontera eficiente restringida", _paso_frontera),
    ("Selección automática de carteras", _paso_perfiles),
    ("Riesgo histórico, forecast y simulación", _paso_riesgo),
    ("Régimen de mercado", _paso_regimen),
    ("Score final y recomendación", _paso_score),
    ("Exploración multi-criterio (frontera + leaderboard)", _paso_exploracion),
)


def _evaluar_degeneracion(ctx: dict) -> tuple[bool, str]:
    """Detecta si la frontera colapsa (el universo no ofrece escalera de riesgo).

    Ocurre cuando hay activos dominados y/o restricciones que dejan una única
    cartera eficiente: todos los perfiles convergen. Es un resultado correcto,
    pero debe comunicarse para no mostrar filas idénticas en silencio."""
    puntos = ctx["frontera"].puntos
    vol = puntos["volatilidad"].to_numpy()
    if len(puntos) <= 1 or (vol.max() - vol.min()) <= max(vol.min() * 0.01, 1e-6):
        return True, (
            "El universo y las restricciones actuales NO ofrecen una escalera de riesgo: "
            "la frontera eficiente colapsa a una única cartera óptima, por lo que Bajo, "
            "Medio, Alto y Máx Sharpe convergen. Suele deberse a activos dominados "
            "(más volatilidad y menos retorno que otros) o a un tope por activo demasiado "
            "estricto sobre pocos activos. Para abrir el abanico: añade activos con perfil "
            "riesgo-retorno distinto, o eleva PESO_MAXIMO_POR_ACTIVO."
        )
    return False, ""


def ensamblar_payload(ctx: dict, log=lambda *_: None) -> ReportPayload:
    """ÚNICO punto de ensamblado del ReportPayload desde el contexto del pipeline.

    Lo usan tanto `construir_payload` (modo programático) como el orquestador con
    barra de progreso, para que NUNCA se desincronicen los campos del informe.
    """
    degenerada, nota = _evaluar_degeneracion(ctx)
    if degenerada:
        log("AVISO: frontera degenerada — los perfiles convergen (ver nota en el informe).")
    expl = ctx["exploracion"]
    return ReportPayload(
        configuracion=ctx["cfg"],
        entrada=ctx["entrada"],
        momentos=ctx["momentos"],
        frontera=ctx["frontera"],
        candidatos=ctx["candidatos"],
        regimen=ctx["regimen"],
        recomendada=ctx["recomendada"],
        recomendacion=ctx["recomendacion"],
        curva_top_sharpe=ctx["curva_top_sharpe"],
        frontera_degenerada=degenerada,
        nota_frontera=nota,
        clasificacion_frontera=expl["clasificacion_frontera"],
        clasificacion_nube=expl["clasificacion_nube"],
        anclas=expl["anclas"],
        leaderboard=expl["leaderboard"],
    )


def construir_payload(cfg: Configuracion, log=lambda *_: None) -> ReportPayload:
    ctx: dict = {"cfg": cfg}
    for _, funcion in PASOS:
        funcion(ctx, log)
    return ensamblar_payload(ctx, log)


def generar_informes(payload: ReportPayload, log=lambda *_: None) -> dict[str, str]:
    """Genera el dashboard HTML y PDF a partir del payload. Devuelve rutas."""
    from pathlib import Path

    from REPORTES import graficos
    from REPORTES.dashboard_html import generar_html
    from REPORTES.dashboard_pdf import generar_pdf

    carpeta = Path(payload.configuracion.carpeta_salidas)
    carpeta.mkdir(parents=True, exist_ok=True)
    figuras = graficos.generar_todos(payload, carpeta / "assets")
    ruta_html = generar_html(payload, carpeta / "informe.html", figuras=figuras)
    log(f"Dashboard HTML: {ruta_html}")
    ruta_pdf = generar_pdf(payload, carpeta / "informe.pdf", figuras=figuras)
    log(f"Dashboard PDF: {ruta_pdf}")
    return {"html": str(ruta_html), "pdf": str(ruta_pdf)}


def ejecutar_completo(cfg: Configuracion, log=lambda *_: None) -> dict[str, str]:
    """Pipeline completo: payload + informes. Punto de entrada programático."""
    payload = construir_payload(cfg, log)
    return generar_informes(payload, log)

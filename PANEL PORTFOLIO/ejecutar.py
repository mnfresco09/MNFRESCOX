"""Punto de entrada del PANEL PORTFOLIO.

Responsabilidades del entry-point: (1) dejar el entorno listo (instalar lo que
falte en vez de fallar), (2) cargar la configuración, (3) ejecutar el comando
mostrando el progreso en vivo con `rich`, con la duración estimada restante.

Las importaciones de las capas pesadas son perezosas (dentro de las funciones)
para que el arranque pueda instalar dependencias ANTES de importarlas.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="PANEL PORTFOLIO")
    parser.add_argument(
        "comando",
        choices=("descargar", "analizar"),
        help="descarga históricos propios o ejecuta el análisis completo",
    )
    return parser


def _ejecutar_descarga(configuracion) -> int:
    from DESCARGADOR.descargador import descargar_cesta, imprimir_resumen

    resumenes = descargar_cesta(
        configuracion.tickers,
        configuracion.fecha_inicio,
        configuracion.fecha_fin,
        configuracion.carpeta_historico,
    )
    imprimir_resumen(resumenes)
    return 0


_OBJETIVOS_ORDEN = ("sharpe", "riesgo", "objetivo", "convexidad", "comparar")


def _elegir_objetivo(console, configuracion) -> str:
    """Pregunta al inicio en qué centrar el análisis (con valor por defecto)."""
    from rich.prompt import Prompt

    from REPORTES.formato import OBJETIVOS

    por_defecto = getattr(configuracion, "objetivo", None) or "comparar"
    console.print("[bold]¿En qué quieres centrar el análisis?[/]")
    for i, clave in enumerate(_OBJETIVOS_ORDEN, start=1):
        marca = "  [dim](por defecto)[/]" if clave == por_defecto else ""
        console.print(f"  [cyan]{i}[/]) {OBJETIVOS[clave]}{marca}")
    idx_def = str(_OBJETIVOS_ORDEN.index(por_defecto) + 1)
    eleccion = Prompt.ask("Elige una opción", choices=[str(i) for i in range(1, 6)], default=idx_def)
    return _OBJETIVOS_ORDEN[int(eleccion) - 1]


def _ejecutar_analisis_con_progreso(configuracion) -> int:
    """Corre el pipeline mostrando cada paso y el tiempo restante estimado."""
    import orquestador

    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:
        return _ejecutar_analisis_plano(configuracion, orquestador)

    from REPORTES.formato import OBJETIVOS

    console = Console()
    console.rule("[bold cyan]PANEL PORTFOLIO · informe de optimización")
    console.print(
        f"[dim]Cesta:[/] {', '.join(configuracion.tickers)}   "
        f"[dim]Periodo:[/] {configuracion.fecha_inicio} → {configuracion.fecha_fin}\n"
    )

    objetivo = _elegir_objetivo(console, configuracion)
    console.print(f"\n[green]Objetivo:[/] {OBJETIVOS[objetivo]}\n")

    pasos = orquestador.PASOS
    total = sum(peso for _, _, peso in pasos)
    ctx: dict = {"cfg": configuracion, "objetivo": objetivo, "rutas": {}}

    columnas = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("· transcurrido"),
        TimeElapsedColumn(),
        TextColumn("· faltan ~"),
        TimeRemainingColumn(),
    )
    with Progress(*columnas, console=console) as progreso:
        tarea = progreso.add_task("Iniciando…", total=total)
        log = lambda mensaje: progreso.console.print(f"   [dim]›[/] {mensaje}")
        for i, (etiqueta, funcion, peso) in enumerate(pasos, start=1):
            progreso.update(tarea, description=f"[{i}/{len(pasos)}] {etiqueta}")
            funcion(ctx, log)
            progreso.advance(tarea, peso)
        progreso.update(tarea, description="[green]Completado")

    rutas = ctx["rutas"]
    console.print("\n[bold green]Informe generado:[/]")
    console.print(f"  HTML  : {rutas['html']}")
    console.print(f"  PDF   : {rutas['pdf']}")
    console.print(f"  Excel : {rutas['excel']}")
    console.print(f"  Manif.: {rutas['manifiesto']}")
    return 0


def _ejecutar_analisis_plano(configuracion, orquestador) -> int:
    """Reserva sin rich: mismo pipeline, mensajes simples."""
    ctx: dict = {"cfg": configuracion, "rutas": {}}
    n = len(orquestador.PASOS)
    for i, (etiqueta, funcion, _) in enumerate(orquestador.PASOS, start=1):
        print(f"[{i}/{n}] {etiqueta}…", flush=True)
        funcion(ctx, lambda m: print(f"   › {m}", flush=True))
    rutas = ctx["rutas"]
    print(f"\nInforme generado:\n  HTML  : {rutas['html']}\n  PDF   : {rutas['pdf']}"
          f"\n  Excel : {rutas['excel']}\n  Manif.: {rutas['manifiesto']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Asegura el entorno, carga la configuración y despacha el comando."""
    try:
        argumentos = _crear_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    # 1) Entorno: instalar lo que falte en vez de fallar.
    from arranque import ErrorEntorno, asegurar_dependencias

    try:
        instalados = asegurar_dependencias()
        if instalados:
            print(f"[ENTORNO] Instalado automáticamente: {', '.join(instalados)}\n")
    except ErrorEntorno as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # 2) Configuración + comando (ya con el entorno listo).
    from CONTRATOS.errores import ErrorPanelPortfolio
    from CONTRATOS.validacion import cargar_configuracion

    try:
        configuracion = cargar_configuracion()
        if argumentos.comando == "descargar":
            return _ejecutar_descarga(configuracion)
        return _ejecutar_analisis_con_progreso(configuracion)
    except ErrorPanelPortfolio as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

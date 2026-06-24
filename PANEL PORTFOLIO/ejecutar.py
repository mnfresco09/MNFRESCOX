"""Punto de entrada del PANEL PORTFOLIO (Motor de Riesgo Predictivo).

Comandos:
  • descargar → descarga/actualiza los históricos de la cesta.
  • analizar  → ejecuta el motor completo y genera el dashboard (HTML + PDF).

Las importaciones de las capas pesadas son perezosas para que el arranque pueda
instalar dependencias ANTES de importarlas.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="PANEL PORTFOLIO")
    parser.add_argument("comando", choices=("descargar", "analizar"),
                        help="descarga históricos o ejecuta el análisis completo")
    return parser


def _ejecutar_descarga(configuracion) -> int:
    from DESCARGADOR.descargador import descargar_cesta, imprimir_resumen
    resumenes = descargar_cesta(
        configuracion.tickers, configuracion.fecha_inicio,
        configuracion.fecha_fin, configuracion.carpeta_historico,
    )
    imprimir_resumen(resumenes)
    return 0


def _ejecutar_analisis(configuracion) -> int:
    import motor

    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn, Progress, SpinnerColumn, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )
    except ImportError:
        rutas = motor.ejecutar_completo(configuracion, log=lambda m: print("   ›", m, flush=True))
        print(f"\nInforme generado:\n  HTML: {rutas['html']}\n  PDF : {rutas['pdf']}")
        return 0

    console = Console()
    console.rule("[bold cyan]PANEL PORTFOLIO · Motor de Riesgo Predictivo")
    console.print(f"[dim]Cesta:[/] {', '.join(configuracion.tickers)}   "
                  f"[dim]Periodo:[/] {configuracion.fecha_inicio} → {configuracion.fecha_fin}\n")

    import orquestador
    pasos = orquestador.PASOS
    total = sum(peso for _, _, peso in pasos)
    ctx: dict = {"cfg": configuracion, "rutas": {}}
    columnas = (
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30), TextColumn("{task.percentage:>3.0f}%"),
        TextColumn("· transcurrido"), TimeElapsedColumn(),
        TextColumn("· faltan ~"), TimeRemainingColumn(),
    )
    with Progress(*columnas, console=console) as progreso:
        tarea = progreso.add_task("Iniciando…", total=total)
        log = lambda m: progreso.console.print(f"   [dim]›[/] {m}")
        for i, (etiqueta, funcion, peso) in enumerate(pasos, start=1):
            progreso.update(tarea, description=f"[{i}/{len(pasos)}] {etiqueta}")
            funcion(ctx, log)
            progreso.advance(tarea, peso)
        progreso.update(tarea, description="[green]Completado")

    rutas = ctx["rutas"]
    console.print("\n[bold green]Dashboard generado:[/]")
    console.print(f"  HTML: {rutas['html']}")
    console.print(f"  PDF : {rutas['pdf']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        argumentos = _crear_parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    from arranque import ErrorEntorno, asegurar_dependencias
    try:
        instalados = asegurar_dependencias()
        if instalados:
            print(f"[ENTORNO] Instalado automáticamente: {', '.join(instalados)}\n")
    except ErrorEntorno as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from CONTRATOS.errores import ErrorPanelPortfolio
    from CONTRATOS.validacion import cargar_configuracion
    try:
        configuracion = cargar_configuracion()
        if argumentos.comando == "descargar":
            return _ejecutar_descarga(configuracion)
        return _ejecutar_analisis(configuracion)
    except ErrorPanelPortfolio as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

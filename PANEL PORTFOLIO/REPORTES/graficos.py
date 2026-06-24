"""Gráficos del informe (matplotlib → PNG). Cada figura responde UNA pregunta.

  • fan_chart           → ¿Cuánto puedo perder este mes? (pregunta 4)
  • pesos_recomendados  → ¿Qué pesos debo usar? (pregunta 3)
  • descomposicion_mcr  → ¿De dónde viene el riesgo? (pregunta 4, apoyo a 3)
  • var_forecast        → ¿Cuánto puedo perder mañana? (pregunta 4)
  • frontera (apéndice) → mapa riesgo-retorno y perfiles (pregunta 3)
  • correlacion (apénd.) → ¿cómo se relacionan? (pregunta 2)

Los PNG se incrustan igual en HTML (base64) y PDF (ruta). Estilo institucional.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from CONTRATOS.modelos import ReportPayload

from . import estilo


def _fig_a_png(fig, ruta: Path) -> tuple[Path, str]:
    import matplotlib.pyplot as plt
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    b64 = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return ruta, f"data:image/png;base64,{b64}"


def fan_chart(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Fan chart a horizonte de la cartera recomendada, en € sobre capital base."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    sim = payload.recomendada.simulacion
    cap = payload.configuracion.capital_base
    sendas = sim.sendas_percentil
    dias = sendas.index.to_numpy()
    cols = list(sendas.columns)  # p5,p25,p50,p75,p95

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    # Bandas (capital base 1 → euros).
    p = {c: sendas[c].to_numpy() * cap for c in cols}
    if "p5" in p and "p95" in p:
        ax.fill_between(dias, p["p5"], p["p95"], color=estilo.ACENTO_CLARO, alpha=0.30,
                        label="P5–P95 (90% de los escenarios)")
    if "p25" in p and "p75" in p:
        ax.fill_between(dias, p["p25"], p["p75"], color=estilo.ACENTO, alpha=0.28,
                        label="P25–P75 (50% central)")
    if "p50" in p:
        ax.plot(dias, p["p50"], color=estilo.TINTA, lw=2.2, label="Mediana (P50)")
    ax.axhline(cap, color=estilo.SUAVE, lw=1.0, ls="--", alpha=0.8)

    ax.set_title(f"Proyección del capital a {sim.horizonte_dias} días — cartera {estilo.nombre_nivel(payload.recomendada.nivel, payload.configuracion.idioma_reporte)}")
    ax.set_xlabel("Días bursátiles")
    ax.set_ylabel("Capital estimado (€)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend(loc="upper left", fontsize=9)
    # Anotación de cola.
    perdida_eur = sim.perdida_p5 * cap
    ax.annotate(f"Escenario adverso (P5): {estilo.pct(sim.perdida_p5)}  ·  {perdida_eur:,.0f} €",
                xy=(dias[-1], p["p5"][-1]), xytext=(0, -14), textcoords="offset points",
                color=estilo.NEG, fontsize=9, ha="right")
    return _fig_a_png(fig, carpeta / "fan_chart.png")


def pesos_recomendados(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    pesos = payload.recomendada.pesos
    activos = list(pesos.index)
    valores = pesos.to_numpy() * 100

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    colores = [estilo.SERIE[i % len(estilo.SERIE)] for i in range(len(activos))]
    barras = ax.barh(activos, valores, color=colores, height=0.6)
    ax.invert_yaxis()
    ax.set_title("Pesos de la cartera recomendada")
    ax.set_xlabel("Peso (%)")
    ax.set_xlim(0, max(valores.max() * 1.25, 10))
    ax.grid(axis="y", visible=False)
    for b, v in zip(barras, valores):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", color=estilo.TINTA, fontsize=10, fontweight="bold")
    return _fig_a_png(fig, carpeta / "pesos.png")


def descomposicion_mcr(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Peso vs contribución al riesgo (MCR): desenmascara activos 'pequeños pero peligrosos'."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    c = payload.recomendada
    activos = list(c.pesos.index)
    peso = c.pesos.reindex(activos).to_numpy() * 100
    ctr = c.descomposicion.contribucion_pct.reindex(activos).to_numpy() * 100

    x = np.arange(len(activos))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - 0.2, peso, width=0.4, color=estilo.ACENTO_CLARO, label="Peso (%)")
    ax.bar(x + 0.2, ctr, width=0.4, color=estilo.NEG, label="Contribución al riesgo (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(activos, fontsize=9)
    ax.set_title("Descomposición del riesgo (MCR) vs peso")
    ax.set_ylabel("% del total")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9)
    return _fig_a_png(fig, carpeta / "mcr.png")


def var_forecast(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """VaR 99% diario de la recomendada por método (histórico, paramétrico, FHS)."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    f = payload.recomendada.forecast
    cap = payload.configuracion.capital_base
    metodos = ["Histórico", "Paramétrico", "FHS"]
    var99 = np.array([f.var_hist_99, f.var_param_99, f.var_fhs_99]) * 100

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    barras = ax.bar(metodos, var99, color=[estilo.SUAVE, estilo.AMBAR, estilo.NEG], width=0.6)
    ax.set_title("VaR 99% diario estimado (cartera recomendada)")
    ax.set_ylabel("Retorno en la cola (%)")
    ax.grid(axis="x", visible=False)
    for b, v in zip(barras, var99):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.06, f"{v:.2f}%\n{v/100*cap:,.0f}€",
                ha="center", va="top", color="white", fontsize=8.5, fontweight="bold")
    ax.axhline(0, color=estilo.LINEA, lw=1)
    return _fig_a_png(fig, carpeta / "var_forecast.png")


def frontera(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Apéndice: mapa riesgo-retorno con nube factible, frontera y los 4 perfiles."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    fr = payload.frontera
    nube = fr.nube_factible
    puntos = fr.puntos

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.scatter(nube["volatilidad"] * 100, nube["retorno"] * 100, s=6,
               c=nube["sharpe"], cmap="Blues", alpha=0.35, label="Universo factible")
    ax.plot(puntos["volatilidad"] * 100, puntos["retorno"] * 100,
            color=estilo.TINTA, lw=2.0, label="Frontera eficiente")
    for c in payload.candidatos:
        ax.scatter(c.volatilidad_estructural * 100, c.retorno_esperado * 100, s=120,
                   color=estilo.NIVEL_COLOR.get(c.nivel, estilo.ACENTO),
                   edgecolor="white", zorder=5,
                   label=estilo.nombre_nivel(c.nivel, payload.configuracion.idioma_reporte))
    ax.set_title("Frontera eficiente restringida y perfiles (apéndice)")
    ax.set_xlabel("Volatilidad anual (%)")
    ax.set_ylabel("Retorno esperado anual (%)")
    ax.legend(fontsize=8.5, loc="best")
    return _fig_a_png(fig, carpeta / "frontera.png")


_CLASE_COLOR = {"bajo": "#0EA5E9", "medio": "#1D4ED8", "alto": "#7C3AED"}


def frontera_clasificada(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Frontera + nube clasificadas por banda de riesgo (PDF)."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    nube = payload.clasificacion_nube
    cf = payload.clasificacion_frontera
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for clase, color in _CLASE_COLOR.items():
        sub = nube[nube["clase"] == clase]
        if not sub.empty:
            ax.scatter(sub["volatilidad"] * 100, sub["retorno"] * 100, s=5, color=color, alpha=0.15)
        subf = cf[cf["clase"] == clase]
        if not subf.empty:
            ax.scatter(subf["volatilidad"] * 100, subf["retorno"] * 100, s=22, color=color,
                       edgecolor="white", linewidth=0.4,
                       label=estilo.nombre_nivel(clase, payload.configuracion.idioma_reporte))
    for nombre, vol in payload.anclas:
        fila = cf.iloc[(cf["volatilidad"] - vol).abs().argmin()]
        ax.scatter(vol * 100, fila["retorno"] * 100, s=90, color=estilo.TINTA, marker="D",
                   edgecolor="white", zorder=6)
        ax.annotate(nombre, (vol * 100, fila["retorno"] * 100), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=estilo.TINTA)
    ax.set_title("Frontera clasificada por banda de riesgo (anclas + bandas)")
    ax.set_xlabel("Volatilidad anual (%)")
    ax.set_ylabel("Retorno esperado anual (%)")
    ax.legend(fontsize=9, loc="best", title="Banda")
    return _fig_a_png(fig, carpeta / "frontera_clasificada.png")


def correlacion(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Apéndice: matriz de correlación media (pregunta 2)."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    corr = payload.momentos.correlacion
    activos = list(corr.index)
    m = corr.to_numpy()

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(activos)))
    ax.set_yticks(range(len(activos)))
    ax.set_xticklabels(activos, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(activos, fontsize=8.5)
    ax.grid(False)
    for i in range(len(activos)):
        for j in range(len(activos)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(m[i, j]) > 0.55 else estilo.TINTA, fontsize=8.5)
    ax.set_title("Correlación media (apéndice)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_a_png(fig, carpeta / "correlacion.png")


def generar_todos(payload: ReportPayload, carpeta: Path) -> dict[str, tuple[Path, str]]:
    carpeta.mkdir(parents=True, exist_ok=True)
    figuras = {
        "fan_chart": fan_chart(payload, carpeta),
        "pesos": pesos_recomendados(payload, carpeta),
        "mcr": descomposicion_mcr(payload, carpeta),
        "var_forecast": var_forecast(payload, carpeta),
        "frontera": frontera(payload, carpeta),
        "correlacion": correlacion(payload, carpeta),
    }
    if len(payload.clasificacion_frontera) and len(payload.clasificacion_nube):
        figuras["frontera_clasificada"] = frontera_clasificada(payload, carpeta)
    return figuras

"""Gráficos del informe (matplotlib → PNG). Cada figura responde UNA pregunta.

  • fan_chart           → ¿Cuánto puedo perder este mes? (pregunta 4)
  • pesos_recomendados  → ¿Qué pesos debo usar? (pregunta 3)
  • descomposicion_mcr  → ¿De dónde viene el riesgo? (pregunta 4, apoyo a 3)
  • var_forecast        → ¿Cuánto puedo perder mañana? (pregunta 4)
  • comparativa_scores  → ¿Por qué esta cartera gana? (pregunta 2)
  • frontera (apéndice) → mapa riesgo-retorno y perfiles (pregunta 3)
  • correlacion (apénd.) → ¿cómo se relacionan? (pregunta 2)

Los PNG se incrustan igual en HTML (base64) y PDF (ruta). Estilo minimalista.
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
        ax.fill_between(dias, p["p5"], p["p95"], color=estilo.ACENTO_CLARO, alpha=0.22,
                        label="P5–P95 (90% de los escenarios)")
    if "p25" in p and "p75" in p:
        ax.fill_between(dias, p["p25"], p["p75"], color=estilo.ACENTO, alpha=0.20,
                        label="P25–P75 (50% central)")
    if "p50" in p:
        ax.plot(dias, p["p50"], color=estilo.TINTA, lw=2.0, label="Mediana (P50)")
    ax.axhline(cap, color=estilo.MUTED, lw=0.8, ls="--", alpha=0.6)

    ax.set_title(f"Proyección del capital a {sim.horizonte_dias} días", fontweight="bold")
    ax.set_xlabel("Días bursátiles")
    ax.set_ylabel("Capital estimado (€)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0)
    # Anotación de cola.
    perdida_eur = sim.perdida_p5 * cap
    ax.annotate(f"Escenario adverso (P5): {estilo.pct(sim.perdida_p5)}  ·  {perdida_eur:,.0f} €",
                xy=(dias[-1], p["p5"][-1]), xytext=(0, -14), textcoords="offset points",
                color=estilo.NEG, fontsize=8.5, ha="right")
    return _fig_a_png(fig, carpeta / "fan_chart.png")


def equity_drawdown(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Equity y drawdown histórico (in-sample, peso fijo) de la cartera seleccionada.

    Misma lectura que la imagen de referencia pero fondo blanco: equity en verde
    y drawdown en rojo, líneas SIN relleno, sobre el mismo eje temporal (drawdown
    en eje secundario). Desempeño histórico con pesos fijos, no promesa de retorno.
    """
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    cfg = payload.configuracion
    cap = cfg.capital_base
    lr = payload.entrada.log_retornos
    pesos_vec = payload.recomendada.pesos.reindex(lr.columns).fillna(0.0).to_numpy()

    simples = np.expm1(lr.to_numpy())
    port = simples @ pesos_vec
    equity = cap * np.cumprod(1.0 + port)

    fechas = lr.index.to_numpy()
    if len(fechas) > 1:
        f0 = fechas[0] - (fechas[1] - fechas[0])
    else:
        f0 = fechas[0]
    fechas_e = np.concatenate([[f0], fechas])
    equity = np.concatenate([[cap], equity])
    pico = np.maximum.accumulate(equity)
    drawdown = (equity / pico - 1.0) * 100.0

    # Rangos enfrentados: baseline de cada serie en el CENTRO vertical, para que
    # equity crezca hacia arriba y drawdown hacia abajo sin pisarse.
    e_min, e_max = float(equity.min()), float(equity.max())
    pad_e = (e_max - e_min) * 0.05 or max(abs(e_max) * 0.05, 1.0)
    e_top = e_max + pad_e
    e_lo = 2.0 * e_min - e_top                       # e_min queda en el 50%
    d_min = float(drawdown.min())
    pad_d = abs(d_min) * 0.05 or 1.0
    d_bot = d_min - pad_d                             # 0% queda en el 50%

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ln_eq, = ax.plot(fechas_e, equity, color=estilo.VERDE, lw=1.8, label="Equity (€)")
    ax.set_ylabel("Equity (€)", color=estilo.VERDE)
    ax.tick_params(axis="y", labelcolor=estilo.VERDE)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.set_xlabel("Fecha")
    ax.set_ylim(e_lo, e_top)

    ax2 = ax.twinx()
    ln_dd, = ax2.plot(fechas_e, drawdown, color=estilo.NEG, lw=1.2, label="Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)", color=estilo.NEG)
    ax2.tick_params(axis="y", labelcolor=estilo.NEG)
    ax2.axhline(0, color=estilo.LINEA, lw=0.6)
    ax2.set_ylim(d_bot, -d_bot)
    ax2.grid(False)

    nivel = estilo.nombre_nivel(payload.recomendada.nivel, cfg.idioma_reporte)
    ax.set_title(f"Equity y drawdown histórico — cartera {nivel} (in-sample, peso fijo)",
                 fontweight="bold")
    ax.legend([ln_eq, ln_dd], [ln_eq.get_label(), ln_dd.get_label()],
              loc="lower left", fontsize=8.5, framealpha=0)
    return _fig_a_png(fig, carpeta / "equity_drawdown.png")


def pesos_recomendados(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    pesos = payload.recomendada.pesos
    activos = list(pesos.index)
    valores = pesos.to_numpy() * 100

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    colores = [estilo.SERIE[i % len(estilo.SERIE)] for i in range(len(activos))]
    barras = ax.barh(activos, valores, color=colores, height=0.55)
    ax.invert_yaxis()
    ax.set_title("Pesos de la cartera recomendada", fontweight="bold")
    ax.set_xlabel("Peso (%)")
    ax.set_xlim(0, max(valores.max() * 1.25, 10))
    ax.grid(axis="y", visible=False)
    for b, v in zip(barras, valores):
        ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", color=estilo.TINTA, fontsize=9.5, fontweight="bold")
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
    ax.bar(x - 0.2, peso, width=0.38, color=estilo.ACENTO_CLARO, label="Peso (%)")
    ax.bar(x + 0.2, ctr, width=0.38, color=estilo.NEG, alpha=0.85, label="Contribución al riesgo (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(activos, fontsize=9)
    ax.set_title("Peso vs contribución al riesgo (MCR)", fontweight="bold")
    ax.set_ylabel("% del total")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8.5, framealpha=0)
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
    colores = [estilo.MUTED, estilo.AMBAR, estilo.NEG]
    barras = ax.bar(metodos, var99, color=colores, width=0.55)
    ax.set_title("VaR 99% diario estimado", fontweight="bold")
    ax.set_ylabel("Retorno en la cola (%)")
    ax.grid(axis="x", visible=False)
    for b, v in zip(barras, var99):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.06, f"{v:.2f}%\n{v/100*cap:,.0f}€",
                ha="center", va="top", color="white", fontsize=8, fontweight="bold")
    ax.axhline(0, color=estilo.LINEA, lw=0.5)
    return _fig_a_png(fig, carpeta / "var_forecast.png")


def comparativa_scores(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Gráfico de barras horizontales comparando scores de todos los candidatos."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    candidatos = sorted(payload.candidatos, key=lambda c: c.score or 0, reverse=True)
    ganadora = (payload.recomendada.motor_optimizacion, payload.recomendada.nivel)

    nombres = []
    scores = []
    colores = []
    for c in candidatos:
        motor = c.motor_optimizacion or "—"
        nivel = estilo.nombre_nivel(c.nivel, payload.configuracion.idioma_reporte)
        nombres.append(f"{motor} / {nivel}")
        scores.append(c.score or 0)
        es_ganadora = (c.motor_optimizacion, c.nivel) == ganadora
        colores.append(estilo.VERDE if es_ganadora else estilo.LINEA.replace("#E5E7EB", "#D1D5DB"))

    # Fix color fallback
    colores = [estilo.VERDE if (c.motor_optimizacion, c.nivel) == ganadora else "#D1D5DB"
               for c in candidatos]

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, len(candidatos) * 0.7)))
    barras = ax.barh(nombres, scores, color=colores, height=0.5)
    ax.invert_yaxis()
    ax.set_title("Comparativa de candidatos (Score compuesto)", fontweight="bold")
    ax.set_xlabel("Score")
    ax.grid(axis="y", visible=False)
    for b, v in zip(barras, scores):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", color=estilo.TINTA, fontsize=9, fontweight="bold")
    return _fig_a_png(fig, carpeta / "comparativa.png")


def frontera(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Apéndice: mapa riesgo-retorno con nube factible, frontera MV y candidatos."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    fr = payload.frontera
    nube = fr.nube_factible
    puntos = fr.puntos

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.scatter(nube["volatilidad"] * 100, nube["retorno"] * 100, s=5,
               c=nube["sharpe"], cmap="Blues", alpha=0.30, label="Universo factible")
    ax.plot(puntos["volatilidad"] * 100, puntos["retorno"] * 100,
            color=estilo.TINTA, lw=1.8, label="Frontera MV de referencia")
    for c in payload.candidatos:
        motor = c.motor_optimizacion or ""
        ax.scatter(c.volatilidad_estructural * 100, c.retorno_esperado * 100, s=110,
                   color=estilo.MOTOR_COLOR.get(motor, estilo.ACENTO),
                   marker=_marcador_motor(motor),
                   edgecolor="white", zorder=5,
                   label=f"{c.motor_optimizacion}/{estilo.nombre_nivel(c.nivel, payload.configuracion.idioma_reporte)}")
    ax.set_title("Frontera eficiente y candidatos", fontweight="bold")
    ax.set_xlabel("Volatilidad anual (%)")
    ax.set_ylabel("Retorno esperado anual (%)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    return _fig_a_png(fig, carpeta / "frontera.png")


def _marcador_motor(motor: str) -> str:
    return {"MARKOWITZ": "o", "CVAR": "D", "NCO": "s"}.get(motor, "o")


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
            ax.scatter(sub["volatilidad"] * 100, sub["retorno"] * 100, s=4, color=color, alpha=0.12)
        subf = cf[cf["clase"] == clase]
        if not subf.empty:
            ax.scatter(subf["volatilidad"] * 100, subf["retorno"] * 100, s=20, color=color,
                       edgecolor="white", linewidth=0.4,
                       label=estilo.nombre_nivel(clase, payload.configuracion.idioma_reporte))
    for nombre, vol in payload.anclas:
        fila = cf.iloc[(cf["volatilidad"] - vol).abs().argmin()]
        ax.scatter(vol * 100, fila["retorno"] * 100, s=80, color=estilo.TINTA, marker="D",
                   edgecolor="white", zorder=6)
        ax.annotate(nombre, (vol * 100, fila["retorno"] * 100), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=estilo.TINTA)
    ax.set_title("Frontera MV clasificada por banda de riesgo", fontweight="bold")
    ax.set_xlabel("Volatilidad anual (%)")
    ax.set_ylabel("Retorno esperado anual (%)")
    ax.legend(fontsize=8.5, loc="best", title="Banda")
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
    ax.set_title("Matriz de correlación", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _fig_a_png(fig, carpeta / "correlacion.png")


def correlacion_rolling(payload: ReportPayload, carpeta: Path) -> tuple[Path, str]:
    """Correlación media par-a-par en ventana móvil (PDF). Consume la serie ya
    calculada en MetricasHistoricas — el reporting no recalcula."""
    import matplotlib.pyplot as plt
    estilo.aplicar_estilo()
    mh = getattr(payload, "metricas_historicas", None)
    ventana = mh.ventana_rolling if mh is not None else 252
    s = mh.correlacion_rolling if mh is not None else None

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    if s is not None and len(s):
        ax.plot(s.index.to_numpy(), s.to_numpy(), color=estilo.ACENTO, lw=1.4)
        media = float(s.mean())
        ax.axhline(media, color=estilo.MUTED, lw=0.9, ls="--", alpha=0.8)
        ax.annotate(f"media {media:.2f}", xy=(s.index[0], media),
                    xytext=(4, 4), textcoords="offset points",
                    color=estilo.SUAVE, fontsize=8.5)
    ax.set_title(f"Correlación media móvil ({ventana} días)", fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Correlación media")
    return _fig_a_png(fig, carpeta / "correlacion_rolling.png")


def generar_todos(payload: ReportPayload, carpeta: Path) -> dict[str, tuple[Path, str]]:
    carpeta.mkdir(parents=True, exist_ok=True)
    figuras = {
        "fan_chart": fan_chart(payload, carpeta),
        "equity_drawdown": equity_drawdown(payload, carpeta),
        "pesos": pesos_recomendados(payload, carpeta),
        "mcr": descomposicion_mcr(payload, carpeta),
        "var_forecast": var_forecast(payload, carpeta),
        "comparativa": comparativa_scores(payload, carpeta),
        "frontera": frontera(payload, carpeta),
        "correlacion": correlacion(payload, carpeta),
    }
    if len(payload.clasificacion_frontera) and len(payload.clasificacion_nube):
        figuras["frontera_clasificada"] = frontera_clasificada(payload, carpeta)
    mh = getattr(payload, "metricas_historicas", None)
    if mh is not None and len(getattr(mh, "correlacion_rolling", [])):
        figuras["correlacion_rolling"] = correlacion_rolling(payload, carpeta)
    return figuras

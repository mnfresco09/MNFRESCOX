"""Texto interpretativo del informe: traduce cada número a una frase.

Genera secciones con explicaciones DINÁMICAS (leen los resultados reales del
run) para que el informe se lea como el de una mesa profesional: no solo enseña
cifras, dice qué significan y qué conviene mirar. Mantiene la honestidad: nada
de promesas de protección futura.
"""

from __future__ import annotations

from dataclasses import dataclass

from CONTRATOS.modelos import PaqueteReporte

from .formato import AVISO_HONESTIDAD, num, pct


@dataclass(frozen=True)
class Seccion:
    titulo: str
    parrafos: tuple[str, ...]


def _mejor_por_sharpe(metricas) -> str:
    return max(metricas.items(), key=lambda kv: kv[1].sharpe)[0]


def _peor_drawdown(metricas) -> str:
    return min(metricas.items(), key=lambda kv: kv[1].max_drawdown)[0]


def _par_cola_mas_fragil(analisis) -> tuple[str, str, float, float]:
    """Activo que más AUMENTA su correlación con la referencia en las colas."""
    dif = analisis.diferencia_correlacion_cola
    media = analisis.correlacion_media
    ref = None
    # Usa la fila de mayor suma de diferencias como 'referencia' (será el activo de ref).
    objetivo = dif.abs().sum().idxmax()
    serie = dif[objetivo].drop(objetivo)
    activo = serie.idxmax()
    return objetivo, activo, float(media.loc[activo, objetivo]), float(media.loc[activo, objetivo] + serie.loc[activo])


def construir_secciones(paquete: PaqueteReporte) -> list[Seccion]:
    cfg = paquete.configuracion
    datos = paquete.datos
    an = paquete.analisis
    riesgo = paquete.riesgo
    secciones: list[Seccion] = []

    # 1. Resumen ejecutivo -----------------------------------------------------
    mejor = _mejor_por_sharpe(riesgo.metricas)
    peor_dd = _peor_drawdown(riesgo.metricas)
    me = riesgo.metricas[mejor]
    secciones.append(Seccion(
        "Resumen ejecutivo",
        (
            f"Se comparan 6 métodos de asignación sobre una cesta de "
            f"{len(cfg.tickers)} activos ({', '.join(cfg.tickers)}) con cierres diarios "
            f"de {cfg.fecha_inicio} a {cfg.fecha_fin}. Tras alinear calendarios quedan "
            f"{len(datos.log_retornos):,} retornos diarios comunes.",
            f"En el backtest walk-forward (out-of-sample), el método con mejor ratio de "
            f"Sharpe realizado es «{mejor}» (Sharpe {num(me.sharpe)}, retorno anual "
            f"{pct(me.retorno_anual)}, máxima caída {pct(me.max_drawdown)}). El método con "
            f"el peor desplome out-of-sample es «{peor_dd}».",
            AVISO_HONESTIDAD,
        ),
    ))

    # 2. Datos y alineación ----------------------------------------------------
    secciones.append(Seccion(
        "Datos y alineación de calendarios",
        (
            "Solo se usa el precio de cierre diario. Como los activos cotizan en "
            "calendarios distintos (las cripto 365 días; bolsa, futuros y divisas con "
            "festivos y fines de semana), se INTERSECTAN únicamente las fechas en que "
            "TODOS cotizan. No se rellenan huecos ni fines de semana: hacerlo fabricaría "
            "retornos cero falsos y rebajaría artificialmente las correlaciones.",
            f"El calendario común va de {datos.cierres.index[0].date()} a "
            f"{datos.cierres.index[-1].date()} y los retornos se calculan como log-retornos "
            "sobre esos cierres alineados.",
        ),
    ))

    # 3. Análisis: momentos, correlación de cola, PCA, regímenes ---------------
    ref, activo, corr_media, corr_cola = _par_cola_mas_fragil(an)
    pc1 = float(an.pca.varianza_explicada.iloc[0])
    n_para_90 = int((an.pca.varianza_acumulada < 0.90).sum() + 1)
    conteo_reg = an.regimenes.value_counts().to_dict()
    secciones.append(Seccion(
        "Análisis y diversificación",
        (
            "La covarianza se estima con encogimiento de Ledoit-Wolf (no la muestral "
            "cruda), que estabiliza la estimación y evita pesos extremos al optimizar.",
            f"Correlación de cola: en los peores días de «{ref}» ({an.observaciones_cola} "
            f"días, el peor decil), «{activo}» pasa de una correlación media de "
            f"{num(corr_media)} a {num(corr_cola)}. Es el caso más claro de "
            "diversificación que se debilita justo en las caídas, que es cuando importa.",
            f"PCA: la primera componente explica el {pct(pc1)} de la varianza y hacen "
            f"falta {n_para_90} componentes para llegar al 90%. Cuantas más componentes "
            "se necesiten, más genuinamente distintos son los activos (no el mismo factor "
            "repetido).",
            "Régimen de mercado (reglas transparentes sobre el activo de referencia): "
            + ", ".join(f"{k} {v} días" for k, v in conteo_reg.items()) + ".",
        ),
    ))

    # 4. Métodos ---------------------------------------------------------------
    secciones.append(Seccion(
        "Los 6 métodos de asignación",
        (
            "Markowitz (máx Sharpe): maximiza rentabilidad/riesgo sobre la covarianza "
            "Ledoit-Wolf; tiende a concentrarse y a sobreajustar la muestra.",
            "Mínima varianza: la cartera de menor riesgo posible, ignorando el retorno "
            "esperado (que es lo más difícil de estimar).",
            "Risk parity: iguala la contribución al riesgo de cada activo; reparte el "
            "riesgo, no el dinero.",
            "HRP (Hierarchical Risk Parity): agrupa los activos por similitud y reparte "
            "por bisección sin invertir la matriz de covarianza, lo que lo hace robusto al "
            "ruido de estimación.",
            "Min-CVaR: minimiza la pérdida esperada en la cola (peor 5%) usando los "
            "escenarios reales, sin suponer una distribución normal.",
            "Black-Litterman: parte del equilibrio de mercado implícito y, si se añaden "
            "views en configuración, las combina con su confianza; sin views, equivale a "
            "la cartera de mercado.",
        ),
    ))

    # 5. Walk-forward ----------------------------------------------------------
    wf = riesgo.walk_forward
    secciones.append(Seccion(
        "Backtest walk-forward (out-of-sample)",
        (
            f"Se estiman los pesos con una ventana pasada de {cfg.ventana_estimacion} días "
            f"y se aplican al mes siguiente no visto, rebalanceando mensualmente y "
            f"deslizando. El periodo evaluado abarca {wf.equity.index[0].date()} a "
            f"{wf.equity.index[-1].date()} con {len(wf.rebalanceos)} rebalanceos.",
            "Por qué importa: medir dentro de la misma muestra con la que se optimiza es "
            "un espejismo, porque Markowitz se ajusta al ruido de ese tramo y 'gana' en el "
            "papel. Al estimar en el pasado y medir en el futuro, la ventaja in-sample se "
            "desinfla y suelen destacar los métodos robustos (risk parity, HRP, mínima "
            "varianza). Las curvas de equity comparadas en el informe son todas "
            "out-of-sample.",
        ),
    ))

    # 6. Stress + regímenes (honestidad) --------------------------------------
    evaluables = [n for n, s in riesgo.stress.items() if s.evaluable]
    no_eval = [n for n, s in riesgo.stress.items() if not s.evaluable]
    frases_stress = [
        "Cada método se evalúa sobre episodios de crisis conocidos. Un episodio solo se "
        "mide si hay cobertura out-of-sample real en esas fechas."
    ]
    if evaluables:
        frases_stress.append("Episodios evaluables con los datos actuales: " + ", ".join(evaluables) + ".")
    if no_eval:
        frases_stress.append(
            "NO evaluables (sin cobertura OOS en esas fechas; no se inventa nada): "
            + ", ".join(no_eval) + "."
        )
    secciones.append(Seccion("Regímenes y stress testing", tuple(frases_stress)))

    # 7. Diversificación en crisis --------------------------------------------
    secciones.append(Seccion(
        "Diversificación en crisis",
        (
            "El número efectivo de apuestas y el ratio de diversificación se calculan "
            "también sobre la covarianza de los días de crisis. Si caen respecto a su "
            "valor global, la cartera diversifica en calma pero menos en las caídas: "
            "exactamente el riesgo que esconde la correlación media.",
        ),
    ))

    return secciones

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
    it = getattr(cfg, "idioma_reporte", "es") == "it"
    secciones: list[Seccion] = []

    # 1. Resumen ejecutivo -----------------------------------------------------
    mejor = _mejor_por_sharpe(riesgo.metricas)
    peor_dd = _peor_drawdown(riesgo.metricas)
    me = riesgo.metricas[mejor]
    if it:
        resumen = (
            f"Si confrontano i metodi di allocazione attivi su un paniere di "
            f"{len(cfg.tickers)} asset ({', '.join(cfg.tickers)}) con chiusure giornaliere "
            f"dal {cfg.fecha_inicio} al {cfg.fecha_fin}. Dopo l'allineamento dei calendari restano "
            f"{len(datos.log_retornos):,} rendimenti giornalieri comuni.",
            f"Nel backtest walk-forward (out-of-sample), il metodo con il miglior Sharpe realizzato "
            f"è «{mejor}» (Sharpe {num(me.sharpe)}, rendimento annuo {pct(me.retorno_anual)}, "
            f"massimo drawdown {pct(me.max_drawdown)}). Il metodo con il peggior drawdown OOS è «{peor_dd}».",
            "Il report mantiene separati dati in-sample e out-of-sample: nessun risultato storico promette protezione futura.",
        )
    else:
        resumen = (
            f"Se comparan 6 métodos de asignación sobre una cesta de "
            f"{len(cfg.tickers)} activos ({', '.join(cfg.tickers)}) con cierres diarios "
            f"de {cfg.fecha_inicio} a {cfg.fecha_fin}. Tras alinear calendarios quedan "
            f"{len(datos.log_retornos):,} retornos diarios comunes.",
            f"En el backtest walk-forward (out-of-sample), el método con mejor ratio de "
            f"Sharpe realizado es «{mejor}» (Sharpe {num(me.sharpe)}, retorno anual "
            f"{pct(me.retorno_anual)}, máxima caída {pct(me.max_drawdown)}). El método con "
            f"el peor desplome out-of-sample es «{peor_dd}».",
            AVISO_HONESTIDAD,
        )
    secciones.append(Seccion("Resumen ejecutivo", resumen))

    # 2. Datos y alineación ----------------------------------------------------
    secciones.append(Seccion(
        "Datos y alineación de calendarios",
        (
            "Si usa solo il prezzo di chiusura giornaliero. Poiché gli asset hanno calendari "
            "diversi, si usano solo le date in cui tutti quotano. Non si riempiono buchi: "
            "creerebbe rendimenti zero falsi e ridurrebbe artificialmente le correlazioni."
            if it else
            "Solo se usa el precio de cierre diario. Como los activos cotizan en "
            "calendarios distintos (las cripto 365 días; bolsa, futuros y divisas con "
            "festivos y fines de semana), se INTERSECTAN únicamente las fechas en que "
            "TODOS cotizan. No se rellenan huecos ni fines de semana: hacerlo fabricaría "
            "retornos cero falsos y rebajaría artificialmente las correlaciones.",
            f"Il calendario comune va dal {datos.cierres.index[0].date()} al "
            f"{datos.cierres.index[-1].date()} e i rendimenti sono log-rendimenti calcolati "
            "sulle chiusure allineate."
            if it else
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
            "La covarianza si stima con shrinkage Ledoit-Wolf, più stabile della covarianza campionaria grezza."
            if it else
            "La covarianza se estima con encogimiento de Ledoit-Wolf (no la muestral "
            "cruda), que estabiliza la estimación y evita pesos extremos al optimizar.",
            f"Correlazione di coda: nei giorni peggiori di «{ref}» ({an.observaciones_cola} "
            f"giorni, peggior decile), «{activo}» passa da una correlazione media di "
            f"{num(corr_media)} a {num(corr_cola)}. È il caso più chiaro in cui la "
            "diversificazione si indebolisce proprio durante le cadute."
            if it else
            f"Correlación de cola: en los peores días de «{ref}» ({an.observaciones_cola} "
            f"días, el peor decil), «{activo}» pasa de una correlación media de "
            f"{num(corr_media)} a {num(corr_cola)}. Es el caso más claro de "
            "diversificación que se debilita justo en las caídas, que es cuando importa.",
            f"PCA: la prima componente spiega il {pct(pc1)} della varianza e servono "
            f"{n_para_90} componenti per arrivare al 90%. Più componenti servono, più gli "
            "asset sono realmente diversi."
            if it else
            f"PCA: la primera componente explica el {pct(pc1)} de la varianza y hacen "
            f"falta {n_para_90} componentes para llegar al 90%. Cuantas más componentes "
            "se necesiten, más genuinamente distintos son los activos (no el mismo factor "
            "repetido).",
            "Regime di mercato: " + ", ".join(f"{k} {v} giorni" for k, v in conteo_reg.items()) + "."
            if it else
            "Régimen de mercado (reglas transparentes sobre el activo de referencia): "
            + ", ".join(f"{k} {v} días" for k, v in conteo_reg.items()) + ".",
        ),
    ))

    # 4. Métodos ---------------------------------------------------------------
    secciones.append(Seccion(
        "Los 6 métodos de asignación",
        (
            "Nucleo robusto: 1/N, minima varianza, risk parity, HRP, massima diversificazione e Min-CVaR non dipendono da una stima fragile dei rendimenti attesi."
            if it else
            "Markowitz (máx Sharpe): maximiza rentabilidad/riesgo sobre la covarianza "
            "Ledoit-Wolf; tiende a concentrarse y a sobreajustar la muestra.",
            "Diagnostica: Markowitz max Sharpe e massimo rendimento fattibile mostrano cosa promette un ottimizzatore aggressivo, non una raccomandazione."
            if it else
            "Mínima varianza: la cartera de menor riesgo posible, ignorando el retorno "
            "esperado (que es lo más difícil de estimar).",
            "Black-Litterman si calcola solo quando ci sono view reali in configurazione; senza view duplicherebbe un portafoglio equivalente e confonderebbe il report."
            if it else
            "Risk parity: iguala la contribución al riesgo de cada activo; reparte el "
            "riesgo, no el dinero.",
            "Le curve e le tabelle separano sempre promessa in-sample e realtà OOS."
            if it else
            "HRP (Hierarchical Risk Parity): agrupa los activos por similitud y reparte "
            "por bisección sin invertir la matriz de covarianza, lo que lo hace robusto al "
            "ruido de estimación.",
            "" if it else
            "Min-CVaR: minimiza la pérdida esperada en la cola (peor 5%) usando los "
            "escenarios reales, sin suponer una distribución normal.",
            "" if it else
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
            f"I pesi si stimano con una finestra passata di {cfg.ventana_estimacion} giorni "
            f"e si applicano al mese successivo non visto. Il periodo valutato va dal "
            f"{wf.equity.index[0].date()} al {wf.equity.index[-1].date()} con "
            f"{len(wf.rebalanceos)} ribilanciamenti."
            if it else
            f"Se estiman los pesos con una ventana pasada de {cfg.ventana_estimacion} días "
            f"y se aplican al mes siguiente no visto, rebalanceando mensualmente y "
            f"deslizando. El periodo evaluado abarca {wf.equity.index[0].date()} a "
            f"{wf.equity.index[-1].date()} con {len(wf.rebalanceos)} rebalanceos.",
            "Misurare nella stessa finestra usata per ottimizzare è un miraggio. Il walk-forward misura nel futuro non visto e rende visibile il degrado rispetto alla promessa in-sample."
            if it else
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
        "Ogni metodo viene valutato su episodi di crisi noti. Un episodio si misura solo se esiste copertura out-of-sample reale in quelle date."
        if it else
        "Cada método se evalúa sobre episodios de crisis conocidos. Un episodio solo se "
        "mide si hay cobertura out-of-sample real en esas fechas."
    ]
    if evaluables:
        frases_stress.append(("Episodi valutabili con i dati attuali: " if it else "Episodios evaluables con los datos actuales: ") + ", ".join(evaluables) + ".")
    if no_eval:
        frases_stress.append(
            ("NON valutabili (senza copertura OOS in quelle date; non si inventa nulla): " if it else
            "NO evaluables (sin cobertura OOS en esas fechas; no se inventa nada): "
            ) + ", ".join(no_eval) + "."
        )
    secciones.append(Seccion("Regímenes y stress testing", tuple(frases_stress)))

    # 7. Diversificación en crisis --------------------------------------------
    secciones.append(Seccion(
        "Diversificación en crisis",
        (
            "Il numero effettivo di scommesse e il ratio di diversificazione si calcolano anche sulla covarianza dei giorni di crisi. Se scendono rispetto al valore globale, la diversificazione funziona meno proprio nelle cadute."
            if it else
            "El número efectivo de apuestas y el ratio de diversificación se calculan "
            "también sobre la covarianza de los días de crisis. Si caen respecto a su "
            "valor global, la cartera diversifica en calma pero menos en las caídas: "
            "exactamente el riesgo que esconde la correlación media.",
        ),
    ))

    return secciones

"""Textos narrativos, traducciones y glosario del informe ejecutivo.

Cada función genera texto en lenguaje natural para que el informe parezca
escrito por un analista. Las narrativas consumen datos del payload para
producir conclusiones interpretativas, no solo números.

Sistema de internacionalización: TODOS los textos están disponibles en
español ("es") e italiano ("it"). El idioma se determina por
`payload.configuracion.idioma_reporte`.
"""

from __future__ import annotations

from . import estilo

# ═══════════════════════════════════════════════════════════════════════════════
#  DICCIONARIO DE TRADUCCIONES  (es / it)
# ═══════════════════════════════════════════════════════════════════════════════
_T = {
    # ── Portada ────────────────────────────────────────────────────────────
    "titulo_informe": {
        "es": "Informe de cartera",
        "it": "Report di portafoglio",
    },
    "subtitulo_informe": {
        "es": "Análisis cuantitativo de riesgo y optimización",
        "it": "Analisi quantitativa di rischio e ottimizzazione",
    },
    "eyebrow": {
        "es": "PANEL PORTFOLIO · Motor cuantitativo buy-side",
        "it": "PANEL PORTFOLIO · Motore quantitativo buy-side",
    },
    "meta_activos": {"es": "activos", "it": "asset"},
    "meta_periodo": {"es": "Periodo", "it": "Periodo"},
    "meta_capital": {"es": "Capital base", "it": "Capitale base"},
    "meta_horizonte": {"es": "Horizonte", "it": "Orizzonte"},
    "meta_dias": {"es": "días", "it": "giorni"},
    "meta_motor": {"es": "Motor", "it": "Motore"},
    "meta_fecha": {"es": "Fecha de generación", "it": "Data di generazione"},

    # ── Secciones ──────────────────────────────────────────────────────────
    "sec_resumen": {"es": "1 · Resumen ejecutivo", "it": "1 · Riepilogo esecutivo"},
    "sec_cartera": {"es": "2 · Cartera recomendada", "it": "2 · Portafoglio raccomandato"},
    "sec_riesgo": {
        "es": "3 · Riesgo a horizonte — simulación a {n} días",
        "it": "3 · Rischio a orizzonte — simulazione a {n} giorni",
    },
    "sec_var": {"es": "4 · VaR diario estimado", "it": "4 · VaR giornaliero stimato"},
    "sec_comparativa": {"es": "5 · Comparativa de candidatos", "it": "5 · Comparativa dei candidati"},
    "sec_glosario": {"es": "Glosario y metodología", "it": "Glossario e metodologia"},
    "sec_detalle": {
        "es": "Detalle técnico (frontera, estadísticas, correlación)",
        "it": "Dettaglio tecnico (frontiera, statistiche, correlazione)",
    },

    # ── KPI labels ─────────────────────────────────────────────────────────
    "kpi_regimen": {"es": "Régimen de mercado", "it": "Regime di mercato"},
    "kpi_vol_tactica": {"es": "Volatilidad táctica (T+1)", "it": "Volatilità tattica (T+1)"},
    "kpi_cartera_rec": {"es": "Cartera recomendada", "it": "Portafoglio raccomandato"},
    "kpi_score": {"es": "Score compuesto", "it": "Score composto"},
    "kpi_ret_mediano": {"es": "Retorno mediano", "it": "Rendimento mediano"},
    "kpi_adverso_p5": {"es": "Escenario adverso (P5)", "it": "Scenario avverso (P5)"},
    "kpi_prob_perdida": {"es": "Probabilidad de pérdida", "it": "Probabilità di perdita"},
    "kpi_cdar": {"es": "CDaR (cola)", "it": "CDaR (coda)"},
    "kpi_a_horizonte": {"es": "a horizonte", "it": "a orizzonte"},

    # ── Tabla maestra (comparativa) ────────────────────────────────────────
    "col_motor": {"es": "Motor", "it": "Motore"},
    "col_perfil": {"es": "Perfil", "it": "Profilo"},
    "col_pesos": {"es": "Pesos", "it": "Pesi"},
    "col_retorno": {"es": "Retorno", "it": "Rendimento"},
    "col_vol": {"es": "Vol. T+1", "it": "Vol. T+1"},
    "col_var99": {"es": "VaR 99%", "it": "VaR 99%"},
    "col_score": {"es": "Score", "it": "Score"},
    "col_decision": {"es": "Decisión", "it": "Decisione"},
    "decision_recomendada": {"es": "RECOMENDADA", "it": "RACCOMANDATA"},

    # ── Tabla MCR ──────────────────────────────────────────────────────────
    "col_activo": {"es": "Activo", "it": "Asset"},
    "col_peso": {"es": "Peso", "it": "Peso"},
    "col_contrib_riesgo": {"es": "Contrib. riesgo", "it": "Contrib. rischio"},

    # ── Tabla VaR ──────────────────────────────────────────────────────────
    "col_metodo": {"es": "Método", "it": "Metodo"},
    "metodo_historico": {"es": "Histórico", "it": "Storico"},
    "metodo_parametrico": {"es": "Paramétrico", "it": "Parametrico"},

    # ── Tabla estadísticas ─────────────────────────────────────────────────
    "col_ret_medio": {"es": "Ret. medio", "it": "Rend. medio"},
    "col_ret_ajustado": {"es": "Ret. ajustado", "it": "Rend. aggiustato"},
    "col_vol_label": {"es": "Vol", "it": "Vol"},
    "col_vol_tactica": {"es": "Vol T+1", "it": "Vol T+1"},
    "col_asimetria": {"es": "Asimetría", "it": "Asimmetria"},
    "col_curtosis": {"es": "Curtosis", "it": "Curtosi"},

    # ── Footer ─────────────────────────────────────────────────────────────
    "footer": {
        "es": "PANEL PORTFOLIO · Documento informativo, no es asesoramiento de inversión.",
        "it": "PANEL PORTFOLIO · Documento informativo, non costituisce consulenza di investimento.",
    },
    "footer_glosario": {
        "es": ("Metodología: covarianza Ledoit-Wolf (estructural) + EWMA (táctica) · "
               "Frontera restringida · FHS & Monte Carlo · Score multifactor. "
               "Este documento es informativo y no constituye asesoramiento de inversión."),
        "it": ("Metodologia: covarianza Ledoit-Wolf (strutturale) + EWMA (tattica) · "
               "Frontiera vincolata · FHS & Monte Carlo · Score multifattoriale. "
               "Questo documento è informativo e non costituisce consulenza di investimento."),
    },

    # ── Notas y avisos ─────────────────────────────────────────────────────
    "aviso_frontera": {"es": "Aviso", "it": "Avviso"},
    "nota_var_convenciones": {
        "es": ("VaR/CVaR: estimaciones bajo los supuestos del modelo (no «pérdida máxima»). "
               "Convención de signo: negativo = pérdida en la cola."),
        "it": ("VaR/CVaR: stime sotto le ipotesi del modello (non «perdita massima»). "
               "Convenzione di segno: negativo = perdita nella coda."),
    },

    # ── Glosario (columnas) ────────────────────────────────────────────────
    "gloss_termino": {"es": "Término", "it": "Termine"},
    "gloss_definicion": {"es": "Definición", "it": "Definizione"},

    # ── Gráficos (títulos) ─────────────────────────────────────────────────
    "graf_pesos": {"es": "Pesos de la cartera recomendada", "it": "Pesi del portafoglio raccomandato"},
    "graf_mcr": {"es": "Peso vs contribución al riesgo (MCR)", "it": "Peso vs contributo al rischio (MCR)"},
    "graf_fan": {
        "es": "Proyección del capital a {n} días",
        "it": "Proiezione del capitale a {n} giorni",
    },
    "graf_var": {"es": "VaR 99% diario estimado", "it": "VaR 99% giornaliero stimato"},
    "graf_comparativa": {
        "es": "Comparativa de candidatos (Score compuesto)",
        "it": "Comparativa dei candidati (Score composto)",
    },
    "graf_frontera": {
        "es": "Frontera eficiente y candidatos",
        "it": "Frontiera efficiente e candidati",
    },
    "graf_correlacion": {"es": "Matriz de correlación", "it": "Matrice di correlazione"},

    # ── Detalle técnico (apéndice) ─────────────────────────────────────────
    "ap_estadistica": {
        "es": "Estadística individual — ¿qué activos tengo?",
        "it": "Statistica individuale — quali asset possiedo?",
    },
    "ap_frontera": {
        "es": "Frontera MV de referencia y candidatos",
        "it": "Frontiera MV di riferimento e candidati",
    },
    "ap_correlacion": {
        "es": "Correlación — ¿cómo se relacionan?",
        "it": "Correlazione — come si relazionano?",
    },
    "sub_equity_dd": {
        "es": "Equity y drawdown histórico de la cartera seleccionada",
        "it": "Equity e drawdown storico del portafoglio selezionato",
    },
    "sub_metricas_hist": {
        "es": "Métricas históricas realizadas (in-sample)",
        "it": "Metriche storiche realizzate (in-sample)",
    },
    "kpi_maxdd": {"es": "Máx. drawdown histórico", "it": "Max drawdown storico"},
    "kpi_cagr": {"es": "CAGR (anual compuesto)", "it": "CAGR (annuo composto)"},
    "kpi_sharpe_hist": {"es": "Sharpe histórico", "it": "Sharpe storico"},
    "kpi_calmar": {"es": "Calmar (CAGR / |MaxDD|)", "it": "Calmar (CAGR / |MaxDD|)"},
    "sub_corr_rolling": {
        "es": "Correlación media móvil ({n} días)",
        "it": "Correlazione media mobile ({n} giorni)",
    },
    "nota_metricas_hist": {
        "es": ("Desempeño realizado in-sample con los pesos recomendados, no una "
               "promesa de retorno futuro. El máximo drawdown es exacto sobre la "
               "curva reconstruida."),
        "it": ("Performance realizzata in-sample con i pesi raccomandati, non una "
               "promessa di rendimento futuro. Il max drawdown è esatto sulla curva "
               "ricostruita."),
    },
    "nota_equity_dd": {
        "es": ("Reconstrucción in-sample aplicando los pesos recomendados (rebalanceo a "
               "peso fijo) a los retornos desde la fecha de inicio. Es desempeño histórico "
               "con pesos fijos, no una promesa de retorno futuro."),
        "it": ("Ricostruzione in-sample applicando i pesi raccomandati (ribilanciamento a "
               "peso fisso) ai rendimenti dalla data di inizio. È performance storica con "
               "pesi fissi, non una promessa di rendimento futuro."),
    },

    # ── Javascript Interactivo ─────────────────────────────────────────────
    "js_clic_frontera": {
        "es": "Haz clic en cualquier punto de la frontera para ver los pesos aquí.",
        "it": "Fai clic su qualsiasi punto della frontiera per vedere i pesi qui."
    },
    "js_cartera_seleccionada": {
        "es": "Cartera seleccionada",
        "it": "Portafoglio selezionato"
    },
    "js_retorno": {
        "es": "retorno",
        "it": "rendimento"
    },
}


def t(clave: str, idioma: str = "es", **kwargs) -> str:
    """Devuelve la traducción de una clave. Acepta formato con kwargs."""
    texto = _T.get(clave, {}).get(idioma, _T.get(clave, {}).get("es", clave))
    if kwargs:
        texto = texto.format(**kwargs)
    return texto


# ═══════════════════════════════════════════════════════════════════════════════
#  NARRATIVAS GENERADAS CON DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def texto_resumen_ejecutivo(payload) -> str:
    """Párrafo principal de conclusión: qué portfolio gana y por qué."""
    rec = payload.recomendada
    cfg = payload.configuracion
    r = payload.regimen
    idi = cfg.idioma_reporte

    motor = rec.motor_optimizacion or "N/A"
    nivel = estilo.nombre_nivel(rec.nivel, idi)
    ret = estilo.pct(rec.retorno_esperado)
    vol = estilo.pct(rec.volatilidad_tactica)
    var99 = estilo.pct(rec.forecast.var_fhs_99, 2)
    score = f"{rec.score:.2f}"
    regimen = r.etiqueta.replace("_", " ")

    n_candidatos = len(payload.candidatos)
    scores_ordenados = sorted(payload.candidatos, key=lambda c: c.score or 0, reverse=True)
    segundo = scores_ordenados[1] if len(scores_ordenados) > 1 else None

    if idi == "it":
        ventaja = ""
        if segundo and rec.score and segundo.score:
            diff = rec.score - segundo.score
            if diff > 0:
                ventaja = (f" Il vantaggio rispetto alla seconda migliore opzione "
                           f"({segundo.motor_optimizacion}/{estilo.nombre_nivel(segundo.nivel, idi)}) "
                           f"è di {diff:.2f} punti nello score composto.")
        return (
            f"Dopo aver valutato {n_candidatos} combinazioni di motore e profilo di rischio "
            f"in un contesto di {regimen}, il portafoglio selezionato è {motor} / {nivel} "
            f"con un rendimento atteso del {ret}, una volatilità tattica del {vol} e un "
            f"VaR 99% giornaliero del {var99}. Lo score composto risultante è {score}, "
            f"il più alto tra tutti i candidati valutati.{ventaja}"
        )
    else:
        ventaja = ""
        if segundo and rec.score and segundo.score:
            diff = rec.score - segundo.score
            if diff > 0:
                ventaja = (f" La ventaja frente a la segunda mejor opción "
                           f"({segundo.motor_optimizacion}/{estilo.nombre_nivel(segundo.nivel, idi)}) "
                           f"es de {diff:.2f} puntos en el score compuesto.")
        return (
            f"Tras evaluar {n_candidatos} combinaciones de motor y perfil de riesgo en un entorno "
            f"de {regimen}, la cartera seleccionada es {motor} / {nivel} con un retorno "
            f"esperado del {ret}, una volatilidad táctica del {vol} y un VaR 99% diario "
            f"del {var99}. El score compuesto resultante es {score}, el más alto entre "
            f"todos los candidatos evaluados.{ventaja}"
        )


def texto_por_que_gana(payload) -> str:
    """Explicación de las ventajas competitivas de la cartera ganadora."""
    rec = payload.recomendada
    cfg = payload.configuracion
    idi = cfg.idioma_reporte

    puntos = []
    var_rec = abs(rec.forecast.var_fhs_99)
    vars_otros = [abs(c.forecast.var_fhs_99) for c in payload.candidatos
                  if (c.motor_optimizacion, c.nivel) != (rec.motor_optimizacion, rec.nivel)]

    if idi == "it":
        if vars_otros and var_rec <= min(vars_otros):
            puntos.append("presenta il minor rischio di coda (VaR 99%) tra tutti i candidati")
        elif vars_otros and var_rec <= sorted(vars_otros)[len(vars_otros)//2]:
            puntos.append("mantiene un rischio di coda contenuto rispetto agli altri candidati")
        sharpes = [c.sharpe for c in payload.candidatos if c.sharpe is not None]
        if sharpes and rec.sharpe and rec.sharpe >= max(sharpes) * 0.95:
            puntos.append("offre un rapporto rendimento-rischio (Sharpe) competitivo")
        if rec.diversificacion and rec.diversificacion > 1.0:
            puntos.append(f"presenta un rapporto di diversificazione di {rec.diversificacion:.2f}, "
                          f"indicando che non concentra il rischio su pochi asset")
        if rec.simulacion and rec.simulacion.cdar_30d:
            cdar = estilo.pct(rec.simulacion.cdar_30d, 1)
            puntos.append(f"il drawdown atteso di coda (CDaR) si attesta al {cdar}")
        if puntos:
            return f"Questo portafoglio si distingue perché {'; '.join(puntos)}."
        return payload.recomendacion.detalle
    else:
        if vars_otros and var_rec <= min(vars_otros):
            puntos.append("presenta el menor riesgo de cola (VaR 99%) entre todos los candidatos")
        elif vars_otros and var_rec <= sorted(vars_otros)[len(vars_otros)//2]:
            puntos.append("mantiene un riesgo de cola contenido respecto a los demás candidatos")
        sharpes = [c.sharpe for c in payload.candidatos if c.sharpe is not None]
        if sharpes and rec.sharpe and rec.sharpe >= max(sharpes) * 0.95:
            puntos.append("ofrece una relación retorno-riesgo (Sharpe) competitiva")
        if rec.diversificacion and rec.diversificacion > 1.0:
            puntos.append(f"presenta un ratio de diversificación de {rec.diversificacion:.2f}, "
                          f"lo que indica que no concentra el riesgo en pocos activos")
        if rec.simulacion and rec.simulacion.cdar_30d:
            cdar = estilo.pct(rec.simulacion.cdar_30d, 1)
            puntos.append(f"el drawdown esperado de cola (CDaR) se sitúa en {cdar}")
        if puntos:
            return f"Esta cartera destaca porque {'; '.join(puntos)}."
        return payload.recomendacion.detalle


def texto_conclusion_riesgo(payload) -> str:
    """Interpretación narrativa del riesgo a horizonte (fan chart)."""
    sim = payload.recomendada.simulacion
    cfg = payload.configuracion
    cap = cfg.capital_base
    idi = cfg.idioma_reporte

    prob = f"{sim.prob_perdida:.0%}"
    mediana_eur = f"€{sim.retorno_mediano * cap:+,.0f}"
    adverso_eur = f"€{sim.perdida_p5 * cap:+,.0f}"
    horizonte = cfg.horizonte_dias

    if idi == "it":
        return (
            f"In un orizzonte di {horizonte} giorni, la simulazione indica una probabilità "
            f"del {prob} di incorrere in perdite. Nello scenario centrale (mediana), "
            f"il capitale si muoverebbe di {mediana_eur}. In uno scenario avverso (percentile 5), "
            f"la perdita stimata raggiungerebbe i {adverso_eur}."
        )
    return (
        f"En un horizonte de {horizonte} días, la simulación indica una probabilidad "
        f"del {prob} de incurrir en pérdidas. En el escenario central (mediana), "
        f"el capital se movería {mediana_eur}. En un escenario adverso (percentil 5), "
        f"la pérdida estimada alcanzaría los {adverso_eur}."
    )


def texto_conclusion_var(payload) -> str:
    """Interpretación narrativa del VaR diario."""
    f = payload.recomendada.forecast
    cap = payload.configuracion.capital_base
    idi = payload.configuracion.idioma_reporte
    var99 = estilo.pct(f.var_fhs_99, 2)
    var99_eur = f"€{abs(f.var_fhs_99 * cap):,.0f}"

    if idi == "it":
        return (
            f"Il modello FHS (Filtered Historical Simulation) stima che, in un giorno avverso "
            f"con probabilità dell'1%, il portafoglio potrebbe perdere fino al {var99} del suo valore, "
            f"equivalente a {var99_eur}. Questa cifra è una stima sotto le ipotesi del "
            f"modello, non una perdita massima garantita."
        )
    return (
        f"El modelo FHS (Filtered Historical Simulation) estima que, en un día adverso "
        f"con probabilidad del 1%, la cartera podría perder hasta un {var99} de su valor, "
        f"equivalente a {var99_eur}. Esta cifra es una estimación bajo los supuestos del "
        f"modelo, no una pérdida máxima garantizada."
    )


def texto_frontera_referencia(idioma: str = "es") -> str:
    if idioma == "it":
        return (
            "La linea mostra la frontiera Media-Varianza (MV) di riferimento, calcolata "
            "con rendimento atteso aggiustato e covarianza strutturale sotto gli stessi "
            "vincoli di peso. La nuvola sono portafogli fattibili campionati. I "
            "sfidanti CVaR/NCO sono disegnati sugli stessi assi per audit, ma non "
            "ottimizzano la stessa funzione obiettivo; per questo possono trovarsi fuori dalla "
            "linea MV senza violare la teoria."
        )
    return (
        "La línea muestra la frontera Media-Varianza (MV) de referencia, calculada "
        "con retorno esperado ajustado y covarianza estructural bajo las mismas "
        "restricciones de pesos. La nube son carteras factibles muestreadas. Los "
        "retadores CVaR/NCO se dibujan en los mismos ejes para auditoría, pero no "
        "optimizan la misma función objetivo; por eso pueden quedar fuera de la "
        "línea MV sin violar la teoría."
    )


def texto_frontera_breve(idioma: str = "es") -> str:
    if idioma == "it":
        return (
            "Linea nera: frontiera MV di riferimento. Punti: candidati dei motori. "
            "CVaR/NCO possono separarsi perché ottimizzano coda o clustering, non varianza."
        )
    return (
        "Línea negra: frontera MV de referencia. Puntos: candidatos de motores. "
        "CVaR/NCO pueden separarse porque optimizan cola o clustering, no varianza."
    )


def texto_score(idioma: str = "es") -> str:
    if idioma == "it":
        return (
            "Il Súper Score privilegia efficienza robusta (Sharpe, Sortino, K-Ratio "
            "e Calmar), assegna un peso leggero al rendimento atteso e penalizza "
            "CVaR, CDaR, max drawdown, concentrazione del rischio, correlazione e "
            "fragilità da contribuzione eccessiva. L'R² resta una diagnostica "
            "in-sample e non decide il portafoglio vincente."
        )
    return (
        "El Súper Score prioriza eficiencia robusta (Sharpe, Sortino, K-Ratio y "
        "Calmar), da un peso ligero al retorno esperado y penaliza CVaR, CDaR, "
        "máximo drawdown, concentración de riesgo, correlación y fragilidad por "
        "contribución excesiva. El R² se mantiene como diagnóstico in-sample y no "
        "decide la cartera ganadora."
    )


def glosario(idioma: str = "es") -> list[tuple[str, str]]:
    """Lista de (término, definición) para la última página del informe."""
    if idioma == "it":
        return [
            ("Rendimento atteso",
             "Rendimento annualizzato stimato del portafoglio, calcolato come media "
             "geometrica dei rendimenti storici aggiustati con shrinkage conservativo."),
            ("Volatilità tattica (T+1)",
             "Volatilità annualizzata stimata per il prossimo giorno di negoziazione, calcolata "
             "tramite il modello EWMA (Exponentially Weighted Moving Average) che dà più "
             "peso ai dati recenti."),
            ("VaR (Value at Risk)",
             "Perdita massima stimata con un livello di confidenza determinato (95% o 99%) "
             "per un orizzonte temporale dato. Non è una perdita massima assoluta ma una "
             "stima statistica."),
            ("CVaR (Conditional VaR)",
             "Media delle perdite che superano la soglia del VaR. Cattura meglio gli "
             "eventi estremi di coda rispetto al VaR standard."),
            ("FHS (Filtered Historical Simulation)",
             "Metodo di stima che combina la volatilità condizionale attuale (GARCH/EWMA) "
             "con la distribuzione empirica storica dei residui standardizzati."),
            ("CDaR (Conditional Drawdown at Risk)",
             "Media dei peggiori drawdown (cali dal massimo) osservati nelle "
             "simulazioni. Misura la severità attesa delle serie di perdite."),
            ("Score composto",
             "Metrica multifattoriale che ordina i portafogli candidati ponderando rendimento, "
             "VaR, CVaR, CDaR, concentrazione e stabilità della curva di capitale."),
            ("MCR (Marginal Contribution to Risk)",
             "Indica quanto contribuisce ciascun asset al rischio totale del portafoglio. Permette "
             "di identificare asset che apportano rischio sproporzionato rispetto al loro peso."),
            ("Sharpe Ratio",
             "Rendimento eccedente rispetto al tasso privo di rischio diviso per la volatilità. "
             "Misura l'efficienza del rischio assunto."),
            ("Regime di mercato",
             "Classificazione del contesto attuale in bassa volatilità, alta volatilità o "
             "crisi, basata su indicatori di volatilità e tendenza."),
            ("Frontiera efficiente (MV)",
             "Insieme di portafogli che massimizzano il rendimento atteso per ciascun livello di "
             "rischio. Calcolata con il modello classico di Markowitz."),
            ("R² della curva di capitale",
             "Bontà di adattamento della curva di capitale in-sample rispetto a una linea retta. "
             "Un R² alto suggerisce una crescita più stabile."),
            ("K-Ratio",
             "Pendenza della curva di capitale divisa per il suo errore standard. Combina "
             "redditività e consistenza temporale."),
            ("Rapporto di diversificazione",
             "Somma ponderata delle volatilità individuali divisa per la volatilità del "
             "portafoglio. Valori maggiori di 1 indicano beneficio di diversificazione."),
        ]
    return [
        ("Retorno esperado",
         "Rentabilidad anualizada estimada de la cartera, calculada como media "
         "geométrica de los retornos históricos ajustados por shrinkage conservador."),
        ("Volatilidad táctica (T+1)",
         "Volatilidad anualizada estimada para el próximo día de negociación, calculada "
         "mediante el modelo EWMA (Exponentially Weighted Moving Average) que da más "
         "peso a los datos recientes."),
        ("VaR (Value at Risk)",
         "Pérdida máxima estimada con un nivel de confianza determinado (95% o 99%) "
         "para un horizonte temporal dado. No es una pérdida máxima absoluta sino una "
         "estimación estadística."),
        ("CVaR (Conditional VaR)",
         "Media de las pérdidas que superan el umbral del VaR. Captura mejor los "
         "eventos extremos de cola que el VaR estándar."),
        ("FHS (Filtered Historical Simulation)",
         "Método de estimación que combina la volatilidad condicional actual (GARCH/EWMA) "
         "con la distribución empírica histórica de los residuos estandarizados."),
        ("CDaR (Conditional Drawdown at Risk)",
         "Media de los peores drawdowns (caídas desde máximo) observados en las "
         "simulaciones. Mide la severidad esperada de las rachas de pérdidas."),
        ("Score compuesto",
         "Métrica multifactor que ordena las carteras candidatas ponderando retorno, "
         "VaR, CVaR, CDaR, concentración y estabilidad de la curva de capital."),
        ("MCR (Marginal Contribution to Risk)",
         "Indica cuánto contribuye cada activo al riesgo total de la cartera. Permite "
         "identificar activos que aportan riesgo desproporcionado respecto a su peso."),
        ("Sharpe Ratio",
         "Retorno excedente sobre la tasa libre de riesgo dividido por la volatilidad. "
         "Mide la eficiencia del riesgo asumido."),
        ("Régimen de mercado",
         "Clasificación del entorno actual en baja volatilidad, alta volatilidad o "
         "crisis, basada en indicadores de volatilidad y tendencia."),
        ("Frontera eficiente (MV)",
         "Conjunto de carteras que maximizan el retorno esperado para cada nivel de "
         "riesgo. Calculada con el modelo clásico de Markowitz."),
        ("R² de la curva de capital",
         "Bondad de ajuste de la curva de capital in-sample respecto a una línea recta. "
         "Un R² alto sugiere crecimiento más estable."),
        ("K-Ratio",
         "Pendiente de la curva de capital dividida por su error estándar. Combina "
         "rentabilidad y consistencia temporal."),
        ("Ratio de diversificación",
         "Suma ponderada de volatilidades individuales dividida por la volatilidad de "
         "la cartera. Valores mayores a 1 indican beneficio de diversificación."),
    ]

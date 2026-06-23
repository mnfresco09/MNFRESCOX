"""Textos de estructura para reportes en español e italiano."""

from __future__ import annotations

from .formato import AVISO_HONESTIDAD

TEXTOS: dict[str, dict[str, str]] = {
    "es": {
        "lang": "es",
        "honestidad": AVISO_HONESTIDAD,
        "nav_recomendacion": "Pesos recomendados",
        "nav_niveles": "Pesos por riesgo",
        "nav_covarianza": "Covarianza y PCA",
        "nav_validacion": "Validación OOS",
        "nav_regimenes": "Regímenes y stress",
        "nav_diversificacion": "Diversificación en crisis",
        "nav_glosario": "Glosario",
        "eyebrow": "Análisis cuantitativo · pesos por riesgo · out-of-sample",
        "titulo": "Pesos óptimos por nivel de riesgo",
        "subtitulo_nav": "Informe de optimización de cartera",
        "meta_analisis": "Cesta de {n} activos · {inicio} a {fin} · rebalanceo {rebalanceo} · ventana {ventana} días",
        "recomendacion": "Pesos recomendados",
        "retorno_esperado": "Retorno esperado",
        "volatilidad_esperada": "Volatilidad esperada",
        "var_historico": "VaR histórico",
        "maxdd_historico": "Max drawdown hist.",
        "frontera_eficiente": "frontera eficiente",
        "perfil": "perfil {perfil}",
        "cola_diaria": "cola diaria",
        "pesos_fijos": "pesos fijos",
        "niveles": "Pesos por nivel de riesgo",
        "niveles_intro": (
            "La tabla y el área apilada recorren la frontera eficiente desde mínima varianza "
            "hacia más retorno esperado. Sirven para ver qué activos ganan o pierden peso al "
            "subir el riesgo, sin convertir la frontera en una promesa de futuro."
        ),
        "covarianza": "Por qué esos pesos: covarianza, cola y PCA",
        "validacion": "Validación honesta: OOS y promesa vs realidad",
        "tabla_oos": "Tabla maestra OOS: pesos y realidad realizada",
        "promesa_realidad": "Promesa vs realidad",
        "regimenes": "Regímenes y stress testing",
        "diversificacion": "Diversificación en crisis",
        "footer": "Generado por PANEL PORTFOLIO · Informe descriptivo, no constituye asesoramiento de inversión.",
        "resumen_perfil": (
            "Para el perfil {perfil}, la cartera eficiente elegida entrega estos pesos: {pesos}. "
            "Retorno esperado {retorno}, volatilidad esperada {volatilidad}, VaR histórico {var}, "
            "CVaR histórico {cvar} y max drawdown histórico {maxdd}."
        ),
        "pdf_pesos_pie": "Pesos de la cartera eficiente correspondiente al perfil elegido.",
        "pdf_niveles_pie": "Pesos de las carteras eficientes por perfil natural.",
        "pdf_area_pie": "Área apilada de composición a lo largo de la frontera.",
        "pdf_frontera_pie": "Frontera eficiente, nube de carteras aleatorias, niveles de riesgo y punto recomendado.",
        "pdf_convexidad_pie": "Retorno medio diario por escenario OOS; no promete protección futura.",
        "pdf_metodos_pie": "Composición de métodos comparados como referencia secundaria.",
        "frontera_titulo": "Plano riesgo-retorno · clic en cualquier punto para ver su composición",
        "volatilidad_anual": "Volatilidad anual",
        "retorno_anual_esperado": "Retorno anual esperado",
        "carteras_aleatorias": "Carteras aleatorias",
        "frontera_eficiente_nombre": "Frontera eficiente",
        "nivel_prefijo": "Nivel",
        "perfil_recomendado": "Perfil recomendado",
        "peso_recomendado": "Peso recomendado",
        "peso": "Peso",
        "nivel_riesgo": "Nivel de riesgo",
        "composicion_frontera_titulo": "Composición a lo largo de la frontera eficiente",
        "convexidad_titulo": "Convexidad: retorno medio diario según qué hizo la cesta (OOS)",
        "retorno_medio_diario": "Retorno medio diario",
        "todo_baja": "Todo baja",
        "mixto": "Mixto",
        "todo_sube": "Todo sube",
        "equity_titulo": "Curvas de capital out-of-sample (walk-forward, base 1.0)",
        "fecha": "Fecha",
        "capital_acumulado": "Capital acumulado (×)",
        "drawdown_titulo": "Curvas bajo el agua: caída desde máximos (out-of-sample)",
        "drawdown": "Drawdown",
        "corr_media_titulo": "Correlación media (todo el periodo)",
        "corr_cola_titulo": "Exceso de correlación en las colas (cola − media): rojo = la diversificación se evapora",
        "pca_titulo": "PCA: cuántos factores independientes mueven la cesta",
        "varianza_explicada": "Varianza explicada",
        "acumulada": "Acumulada",
        "pesos_metodos_titulo": "Composición de cada cartera (pesos por activo)",
        "regimen_titulo": "Retorno anualizado por régimen (OOS) · verde = creció, rojo = sufrió",
        "diversificacion_titulo": "Número efectivo de apuestas: global vs. crisis",
        "enb_global": "Nº efectivo (global)",
        "enb_crisis": "Nº efectivo (crisis)",
        "apuestas_independientes": "Apuestas independientes",
    },
    "it": {
        "lang": "it",
        "honestidad": (
            "Questo report è un'ANALISI DESCRITTIVA del comportamento passato del paniere in "
            "regimi passati. Non è una previsione né una raccomandazione di investimento e non "
            "garantisce alcuna protezione nella prossima crisi: correlazioni, volatilità e "
            "rendimenti cambiano nel tempo. Il backtest è walk-forward (out-of-sample) e, anche "
            "così, un buon risultato storico non garantisce risultati futuri."
        ),
        "nav_recomendacion": "Pesi consigliati",
        "nav_niveles": "Pesi per rischio",
        "nav_covarianza": "Covarianza e PCA",
        "nav_validacion": "Validazione OOS",
        "nav_regimenes": "Regimi e stress",
        "nav_diversificacion": "Diversificazione in crisi",
        "nav_glosario": "Glossario",
        "eyebrow": "Analisi quantitativa · pesi per rischio · out-of-sample",
        "titulo": "Pesi ottimali per livello di rischio",
        "subtitulo_nav": "Report di ottimizzazione del portafoglio",
        "meta_analisis": "Paniere di {n} asset · {inicio} a {fin} · ribilanciamento {rebalanceo} · finestra {ventana} giorni",
        "recomendacion": "Pesi consigliati",
        "retorno_esperado": "Rendimento atteso",
        "volatilidad_esperada": "Volatilità attesa",
        "var_historico": "VaR storico",
        "maxdd_historico": "Max drawdown stor.",
        "frontera_eficiente": "frontiera efficiente",
        "perfil": "profilo {perfil}",
        "cola_diaria": "coda giornaliera",
        "pesos_fijos": "pesi fissi",
        "niveles": "Pesi per livello di rischio",
        "niveles_intro": (
            "La tabella e l'area impilata percorrono la frontiera efficiente dalla minima varianza "
            "verso un rendimento atteso più alto. Mostrano quali asset aumentano o riducono il peso "
            "quando sale il rischio, senza trasformare la frontiera in una promessa futura."
        ),
        "covarianza": "Perché questi pesi: covarianza, coda e PCA",
        "validacion": "Validazione onesta: OOS e promessa vs realtà",
        "tabla_oos": "Tabella principale OOS: pesi e realtà realizzata",
        "promesa_realidad": "Promessa vs realtà",
        "regimenes": "Regimi e stress testing",
        "diversificacion": "Diversificazione in crisi",
        "footer": "Generato da PANEL PORTFOLIO · Report descrittivo, non costituisce consulenza di investimento.",
        "resumen_perfil": (
            "Per il profilo {perfil}, il portafoglio efficiente selezionato assegna questi pesi: {pesos}. "
            "Rendimento atteso {retorno}, volatilità attesa {volatilidad}, VaR storico {var}, "
            "CVaR storico {cvar} e max drawdown storico {maxdd}."
        ),
        "pdf_pesos_pie": "Pesi del portafoglio efficiente corrispondente al profilo scelto.",
        "pdf_niveles_pie": "Pesi dei portafogli efficienti per profilo naturale.",
        "pdf_area_pie": "Area impilata della composizione lungo la frontiera.",
        "pdf_frontera_pie": "Frontiera efficiente, nube di portafogli casuali, livelli di rischio e punto consigliato.",
        "pdf_convexidad_pie": "Rendimento medio giornaliero per scenario OOS; non promette protezione futura.",
        "pdf_metodos_pie": "Composizione dei metodi confrontati come riferimento secondario.",
        "frontera_titulo": "Piano rischio-rendimento · clic su un punto per vedere i pesi",
        "volatilidad_anual": "Volatilità annua",
        "retorno_anual_esperado": "Rendimento annuo atteso",
        "carteras_aleatorias": "Portafogli casuali",
        "frontera_eficiente_nombre": "Frontiera efficiente",
        "nivel_prefijo": "Livello",
        "perfil_recomendado": "Profilo consigliato",
        "peso_recomendado": "Peso consigliato",
        "peso": "Peso",
        "nivel_riesgo": "Livello di rischio",
        "composicion_frontera_titulo": "Composizione lungo la frontiera efficiente",
        "convexidad_titulo": "Convessità: rendimento medio giornaliero per scenario (OOS)",
        "retorno_medio_diario": "Rendimento medio giornaliero",
        "todo_baja": "Tutto scende",
        "mixto": "Misto",
        "todo_sube": "Tutto sale",
        "equity_titulo": "Curve di capitale out-of-sample (walk-forward, base 1.0)",
        "fecha": "Data",
        "capital_acumulado": "Capitale accumulato (×)",
        "drawdown_titulo": "Curve di drawdown: caduta dai massimi (out-of-sample)",
        "drawdown": "Drawdown",
        "corr_media_titulo": "Correlazione media (intero periodo)",
        "corr_cola_titulo": "Eccesso di correlazione nelle code (coda − media): rosso = diversificazione che svanisce",
        "pca_titulo": "PCA: quanti fattori indipendenti muovono il paniere",
        "varianza_explicada": "Varianza spiegata",
        "acumulada": "Cumulata",
        "pesos_metodos_titulo": "Composizione di ogni portafoglio (pesi per asset)",
        "regimen_titulo": "Rendimento annuo per regime (OOS) · verde = crescita, rosso = sofferenza",
        "diversificacion_titulo": "Numero effettivo di scommesse: globale vs crisi",
        "enb_global": "N. effettivo (globale)",
        "enb_crisis": "N. effettivo (crisi)",
        "apuestas_independientes": "Scommesse indipendenti",
    },
}

PERFILES = {
    "es": {
        "conservador": "Conservador",
        "moderado": "Moderado",
        "agresivo": "Agresivo",
        "personalizado": "Personalizado",
    },
    "it": {
        "conservador": "Conservativo",
        "moderado": "Moderato",
        "agresivo": "Aggressivo",
        "personalizado": "Personalizzato",
    },
}

COLUMNAS = {
    "it": {
        "Retorno anual (OOS)": "Rendimento annuo (OOS)",
        "Volatilidad (OOS)": "Volatilità (OOS)",
        "Sharpe (OOS)": "Sharpe (OOS)",
        "Sortino (OOS)": "Sortino (OOS)",
        "Calmar (OOS)": "Calmar (OOS)",
        "Max drawdown (OOS)": "Max drawdown (OOS)",
        "VaR 95% (OOS)": "VaR 95% (OOS)",
        "CVaR 95% (OOS)": "CVaR 95% (OOS)",
        "Sharpe esperado (in-sample)": "Sharpe atteso (in-sample)",
        "Sharpe realizado (OOS)": "Sharpe realizzato (OOS)",
        "Degradación de Sharpe": "Degrado dello Sharpe",
        "Retorno esperado (in-sample)": "Rendimento atteso (in-sample)",
        "Retorno realizado (OOS)": "Rendimento realizzato (OOS)",
        "Retorno esperado": "Rendimento atteso",
        "Volatilidad esperada": "Volatilità attesa",
        "VaR histórico": "VaR storico",
        "CVaR histórico": "CVaR storico",
        "Max drawdown histórico": "Max drawdown storico",
        "Nivel": "Livello",
        "Método": "Metodo",
    }
}

GLOSARIO_IT = {
    "Rendimento atteso": "Rendimento medio annuo stimato dalla storia; non è una promessa.",
    "Volatilità": "Quanto oscilla il portafoglio; più volatilità significa più incertezza.",
    "Sharpe": "Rendimento per unità di rischio totale. Più alto è, meglio è remunerato il rischio.",
    "Sortino": "Come lo Sharpe, ma penalizza solo i movimenti negativi.",
    "Calmar": "Rendimento annuo diviso per il peggior drawdown.",
    "Max drawdown": "La maggiore caduta da un massimo al minimo successivo.",
    "VaR 95%": "Perdita giornaliera superata solo nel 5% dei giorni, stimata storicamente.",
    "CVaR 95%": "Perdita media nel 5% dei giorni peggiori.",
    "Ledoit-Wolf": "Covarianza stabilizzata per ridurre il rumore e limitare pesi estremi.",
    "Correlazione di coda": "Correlazione misurata nei giorni peggiori; rivela quando la diversificazione svanisce.",
    "Numero effettivo di scommesse": "Quante esposizioni realmente indipendenti contiene il portafoglio.",
    "Walk-forward": "Stimare i pesi sul passato e misurarli sul futuro non visto.",
    "Massima diversificazione": "Portafoglio che premia asset che si muovono in modo diverso.",
    "Cattura rialzista": "Quanto segue il portafoglio quando il mercato sale.",
    "Cattura ribassista": "Quanto segue il portafoglio quando il mercato scende.",
    "Asimmetria (convessità)": "Cattura rialzista meno cattura ribassista.",
    "Equipesata (1/N)": "Ripartire il capitale in parti uguali tra tutti gli asset.",
}

METODOS_IT = {
    "Equiponderada (1/N)": "Equipesata (1/N)",
    "Mínima varianza": "Minima varianza",
    "Risk parity": "Risk parity",
    "HRP": "HRP",
    "Máxima diversificación": "Massima diversificazione",
    "Min-CVaR": "Min-CVaR",
    "Markowitz (máx Sharpe)": "Markowitz (max Sharpe)",
    "Markowitz (máx retorno factible)": "Markowitz (max rendimento fattibile)",
    "Black-Litterman": "Black-Litterman",
}


def idioma(paquete) -> str:
    valor = getattr(paquete.configuracion, "idioma_reporte", "es")
    return valor if valor in TEXTOS else "es"


def t(paquete, clave: str) -> str:
    lang = idioma(paquete)
    return TEXTOS[lang].get(clave, TEXTOS["es"][clave])


def perfil_visible(paquete, nivel: str) -> str:
    return PERFILES.get(idioma(paquete), PERFILES["es"]).get(nivel, nivel.capitalize())


def columna_visible(paquete, columna: str) -> str:
    if columna.startswith("peso ·"):
        return columna.replace("peso ·", "peso ·" if idioma(paquete) == "it" else "peso ·", 1)
    return COLUMNAS.get(idioma(paquete), {}).get(columna, columna)


def metodo_visible(paquete, metodo: str) -> str:
    if idioma(paquete) == "it":
        return METODOS_IT.get(metodo, metodo)
    return metodo


def glosario(paquete):
    if idioma(paquete) == "it":
        return GLOSARIO_IT
    from .formato import GLOSARIO
    return GLOSARIO

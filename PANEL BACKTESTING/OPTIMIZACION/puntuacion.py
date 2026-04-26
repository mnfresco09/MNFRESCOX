def calcular_score(metricas: dict) -> float:
    """
    Funcion de puntuacion inicial definida por la guia.
    Version 1: el score es el ROI total del backtest.
    """
    return float(metricas["roi_total"])

"""Excepciones con contexto de etapa para detener el pipeline."""


class ErrorPanelPortfolio(RuntimeError):
    """Error base controlado del panel."""

    def __init__(self, etapa: str, mensaje: str) -> None:
        self.etapa = etapa
        self.mensaje = mensaje
        super().__init__(f"[{etapa}] {mensaje}")


class ErrorConfiguracion(ErrorPanelPortfolio):
    """Configuración inválida."""

    def __init__(self, mensaje: str) -> None:
        super().__init__("CONFIGURACION", mensaje)


class ErrorDatos(ErrorPanelPortfolio):
    """Fallo de descarga, persistencia o preparación de datos."""


class ErrorAnalisis(ErrorPanelPortfolio):
    """Fallo al calcular momentos, covarianza, correlación, PCA o regímenes."""

    def __init__(self, mensaje: str) -> None:
        super().__init__("ANALISIS", mensaje)


class ErrorOptimizacion(ErrorPanelPortfolio):
    """Fallo de un método de asignación."""


class ErrorRiesgo(ErrorPanelPortfolio):
    """Fallo al calcular riesgo o el backtest walk-forward."""

    def __init__(self, mensaje: str) -> None:
        super().__init__("RIESGO", mensaje)


class ErrorReporte(ErrorPanelPortfolio):
    """Fallo al generar una salida requerida."""

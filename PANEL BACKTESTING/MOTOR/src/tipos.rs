// ---------------------------------------------------------------------------
// tipos.rs — Estructuras de datos del motor de simulación
//
// Todas las structs que viajan entre Python y Rust, más los tipos internos
// del simulador. PyO3 convierte automáticamente entre estos y los tipos Python.
// ---------------------------------------------------------------------------

use pyo3::prelude::*;

/// Dirección de un trade.
/// Se mapea 1:1 con Señal en Python: 1 = LONG, -1 = SHORT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direccion {
    Long,
    Short,
}

impl Direccion {
    /// Convierte el valor entero de la señal Python al enum Rust.
    pub fn from_signal(val: i8) -> Result<Self, String> {
        match val {
            1 => Ok(Direccion::Long),
            -1 => Ok(Direccion::Short),
            _ => Err(format!(
                "Señal inválida: {val}. Solo se aceptan 0, 1 (LONG) o -1 (SHORT)."
            )),
        }
    }
}

/// Tipo de salida soportado por el motor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExitType {
    Fixed,
    Bars,
}

impl ExitType {
    pub fn from_str(val: &str) -> Result<Self, String> {
        match val {
            "FIXED" => Ok(ExitType::Fixed),
            "BARS" => Ok(ExitType::Bars),
            _ => Err(format!(
                "EXIT_TYPE inválido: '{val}'. Opciones soportadas por el motor: FIXED, BARS."
            )),
        }
    }
}

/// Configuración de la simulación. Se recibe una vez desde Python al inicio.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Capital inicial en USD
    pub saldo_inicial: f64,
    /// Colateral por operación en USD
    pub saldo_por_trade: f64,
    /// Multiplicador de apalancamiento (>= 1)
    pub apalancamiento: f64,
    /// Saldo mínimo para seguir operando — si cae por debajo, el backtest para
    pub saldo_minimo: f64,
    /// Comisión como fracción decimal (ej: 0.0005 = 0.05%)
    pub comision_pct: f64,
    /// 1 = solo apertura, 2 = apertura y cierre
    pub comision_lados: u8,
    /// Tipo de salida
    pub exit_type: ExitType,
    /// Stop Loss como % del colateral (ej: 20.0 = 20%)
    pub exit_sl_pct: f64,
    /// Take Profit como % del colateral (ej: 40.0 = 40%)
    pub exit_tp_pct: f64,
    /// Máximo de velas para tipo BARS (0 si no aplica)
    pub exit_velas: usize,
}

/// Datos OHLCV de una vela individual, extraídos del DataFrame Python.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Vela {
    /// Timestamp en microsegundos desde epoch UTC
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Resultado de un trade individual. Se devuelve a Python como dict.
#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
pub struct TradeResult {
    /// Índice de la vela donde se detectó la señal (vela N)
    #[pyo3(get)]
    pub idx_señal: usize,
    /// Índice de la vela donde se abrió el trade (vela N+1)
    #[pyo3(get)]
    pub idx_entrada: usize,
    /// Índice de la vela donde se cerró el trade
    #[pyo3(get)]
    pub idx_salida: usize,
    /// Timestamp de la señal (microsegundos epoch UTC)
    #[pyo3(get)]
    pub ts_señal: i64,
    /// Timestamp de la entrada
    #[pyo3(get)]
    pub ts_entrada: i64,
    /// Timestamp de la salida
    #[pyo3(get)]
    pub ts_salida: i64,
    /// 1 = LONG, -1 = SHORT
    #[pyo3(get)]
    pub direccion: i8,
    /// Precio de entrada (open de la vela N+1)
    #[pyo3(get)]
    pub precio_entrada: f64,
    /// Precio de salida (donde se cumplió la condición)
    #[pyo3(get)]
    pub precio_salida: f64,
    /// Colateral usado (USD)
    #[pyo3(get)]
    pub colateral: f64,
    /// Tamaño de la posición = colateral * apalancamiento / precio_entrada
    #[pyo3(get)]
    pub tamaño_posicion: f64,
    /// Comisiones totales pagadas (USD)
    #[pyo3(get)]
    pub comision_total: f64,
    /// P&L neto después de comisiones (USD)
    #[pyo3(get)]
    pub pnl: f64,
    /// ROI sobre el colateral (fracción decimal, ej: 0.15 = 15%)
    #[pyo3(get)]
    pub roi: f64,
    /// Saldo después de cerrar este trade (USD)
    #[pyo3(get)]
    pub saldo_post: f64,
    /// Motivo de cierre: "SL", "TP", "BARS", "END"
    #[pyo3(get)]
    pub motivo_salida: String,
    /// Número de velas que duró el trade
    #[pyo3(get)]
    pub duracion_velas: usize,
}

/// Resultado completo de una simulación. Contiene todos los trades y métricas resumen.
#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
pub struct SimResult {
    /// Lista de todos los trades ejecutados
    #[pyo3(get)]
    pub trades: Vec<TradeResult>,
    /// Saldo final después de todos los trades
    #[pyo3(get)]
    pub saldo_final: f64,
    /// Saldo inicial de la simulación
    #[pyo3(get)]
    pub saldo_inicial: f64,
    /// Total de trades ejecutados
    #[pyo3(get)]
    pub total_trades: usize,
    /// Trades ganadores (PnL > 0)
    #[pyo3(get)]
    pub trades_ganadores: usize,
    /// Trades perdedores (PnL <= 0)
    #[pyo3(get)]
    pub trades_perdedores: usize,
    /// Win rate como fracción decimal
    #[pyo3(get)]
    pub win_rate: f64,
    /// ROI total sobre el capital inicial
    #[pyo3(get)]
    pub roi_total: f64,
    /// PnL neto total (USD)
    #[pyo3(get)]
    pub pnl_total: f64,
    /// PnL promedio por trade (USD)
    #[pyo3(get)]
    pub pnl_promedio: f64,
    /// Máximo drawdown como fracción decimal
    #[pyo3(get)]
    pub max_drawdown: f64,
    /// Curva de equity: saldo después de cada trade
    #[pyo3(get)]
    pub equity_curve: Vec<f64>,
    /// Si el backtest fue detenido por saldo insuficiente
    #[pyo3(get)]
    pub parado_por_saldo: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direccion_rechaza_senal_invalida() {
        let err = Direccion::from_signal(2).unwrap_err();
        assert!(err.contains("Señal inválida"));
    }

    #[test]
    fn test_exit_type_rechaza_valor_invalido() {
        let err = ExitType::from_str("CUSTOM").unwrap_err();
        assert!(err.contains("EXIT_TYPE inválido"));
    }
}

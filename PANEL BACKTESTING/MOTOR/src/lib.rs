// ---------------------------------------------------------------------------
// lib.rs — Punto de entrada del crate. Bindings PyO3.
//
// Expone una única función a Python: simulate_trades()
// Se importa desde Python como:
//   from motor_backtesting import simulate_trades
// ---------------------------------------------------------------------------

mod tipos;
mod capital;
mod simulador;

use pyo3::prelude::*;
use tipos::{SimConfig, SimResult, TradeResult, Vela};

/// Función principal expuesta a Python.
///
/// Recibe las columnas OHLCV como listas, las señales como lista de i8,
/// y todos los parámetros de configuración como argumentos individuales.
///
/// Devuelve un SimResult con todos los trades y métricas.
#[pyfunction]
#[pyo3(signature = (
    timestamps, opens, highs, lows, closes, volumes, señales,
    saldo_inicial, saldo_por_trade, apalancamiento, saldo_minimo,
    comision_pct, comision_lados,
    exit_type, exit_sl_pct, exit_tp_pct, exit_velas
))]
fn simulate_trades(
    timestamps: Vec<i64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    señales: Vec<i8>,
    saldo_inicial: f64,
    saldo_por_trade: f64,
    apalancamiento: f64,
    saldo_minimo: f64,
    comision_pct: f64,
    comision_lados: u8,
    exit_type: String,
    exit_sl_pct: f64,
    exit_tp_pct: f64,
    exit_velas: usize,
) -> PyResult<SimResult> {
    let n = timestamps.len();

    // Validaciones básicas
    if n != opens.len() || n != highs.len() || n != lows.len()
        || n != closes.len() || n != volumes.len() || n != señales.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Todas las columnas deben tener el mismo número de filas.",
        ));
    }

    if n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Se necesitan al menos 2 velas para simular.",
        ));
    }

    // Construir vector de velas
    let velas: Vec<Vela> = (0..n)
        .map(|i| Vela {
            timestamp: timestamps[i],
            open: opens[i],
            high: highs[i],
            low: lows[i],
            close: closes[i],
            volume: volumes[i],
        })
        .collect();

    // Construir configuración
    let config = SimConfig {
        saldo_inicial,
        saldo_por_trade,
        apalancamiento,
        saldo_minimo,
        comision_pct,
        comision_lados,
        exit_type,
        exit_sl_pct,
        exit_tp_pct,
        exit_velas,
    };

    // Ejecutar simulación
    let resultado = simulador::simular(&velas, &señales, &config);

    Ok(resultado)
}

/// Módulo Python. El nombre DEBE coincidir con lib.name en Cargo.toml.
#[pymodule]
fn motor_backtesting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_trades, m)?)?;
    m.add_class::<TradeResult>()?;
    m.add_class::<SimResult>()?;
    Ok(())
}

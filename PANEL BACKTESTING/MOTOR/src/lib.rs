// ---------------------------------------------------------------------------
// lib.rs — Punto de entrada del crate. Bindings PyO3.
//
// Expone una única función a Python: simulate_trades()
// Se importa desde Python como:
//   from motor_backtesting import simulate_trades
// ---------------------------------------------------------------------------

mod capital;
mod simulador;
mod tipos;

use pyo3::prelude::*;
use tipos::{Direccion, ExitType, SimConfig, SimResult, TradeResult, Vela};

/// Función principal expuesta a Python.
///
/// Recibe las columnas OHLCV como listas, las señales como lista de i8,
/// y todos los parámetros de configuración como argumentos individuales.
///
/// Devuelve un SimResult con todos los trades y métricas.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
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
    if n != opens.len()
        || n != highs.len()
        || n != lows.len()
        || n != closes.len()
        || n != volumes.len()
        || n != señales.len()
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

    validar_columnas(&opens, &highs, &lows, &closes, &volumes, &señales)?;
    let exit_type = validar_config(
        saldo_inicial,
        saldo_por_trade,
        apalancamiento,
        saldo_minimo,
        comision_pct,
        comision_lados,
        &exit_type,
        exit_sl_pct,
        exit_tp_pct,
        exit_velas,
    )?;

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

fn validar_columnas(
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[f64],
    señales: &[i8],
) -> PyResult<()> {
    for i in 0..opens.len() {
        let open = opens[i];
        let high = highs[i];
        let low = lows[i];
        let close = closes[i];
        let volume = volumes[i];

        if !open.is_finite() || !high.is_finite() || !low.is_finite() || !close.is_finite() {
            return value_error(format!("OHLC contiene NaN o infinito en la fila {i}."));
        }
        if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 {
            return value_error(format!("OHLC debe ser mayor que 0 en la fila {i}."));
        }
        if high < low {
            return value_error(format!("high no puede ser menor que low en la fila {i}."));
        }
        if open > high || open < low || close > high || close < low {
            return value_error(format!(
                "open y close deben estar dentro del rango high/low en la fila {i}."
            ));
        }
        if !volume.is_finite() || volume < 0.0 {
            return value_error(format!("volume debe ser finito y >= 0 en la fila {i}."));
        }
        if señales[i] != 0 {
            Direccion::from_signal(señales[i]).map_err(pyo3::exceptions::PyValueError::new_err)?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validar_config(
    saldo_inicial: f64,
    saldo_por_trade: f64,
    apalancamiento: f64,
    saldo_minimo: f64,
    comision_pct: f64,
    comision_lados: u8,
    exit_type: &str,
    exit_sl_pct: f64,
    exit_tp_pct: f64,
    exit_velas: usize,
) -> PyResult<ExitType> {
    if !saldo_inicial.is_finite() || saldo_inicial <= 0.0 {
        return value_error("saldo_inicial debe ser finito y mayor que 0.");
    }
    if !saldo_por_trade.is_finite() || saldo_por_trade <= 0.0 {
        return value_error("saldo_por_trade debe ser finito y mayor que 0.");
    }
    if saldo_por_trade > saldo_inicial {
        return value_error("saldo_por_trade no puede ser mayor que saldo_inicial.");
    }
    if !apalancamiento.is_finite() || apalancamiento < 1.0 {
        return value_error("apalancamiento debe ser finito y >= 1.");
    }
    if !saldo_minimo.is_finite() || saldo_minimo < 0.0 {
        return value_error("saldo_minimo debe ser finito y >= 0.");
    }
    if !comision_pct.is_finite() || !(0.0..1.0).contains(&comision_pct) {
        return value_error("comision_pct debe ser finito y estar entre 0 y 1.");
    }
    if comision_lados != 1 && comision_lados != 2 {
        return value_error("comision_lados debe ser 1 (apertura) o 2 (apertura y cierre).");
    }
    if !exit_sl_pct.is_finite() || exit_sl_pct < 0.0 {
        return value_error("exit_sl_pct debe ser finito y >= 0.");
    }
    if !exit_tp_pct.is_finite() || exit_tp_pct < 0.0 {
        return value_error("exit_tp_pct debe ser finito y >= 0.");
    }

    let parsed_exit_type =
        ExitType::from_str(exit_type).map_err(pyo3::exceptions::PyValueError::new_err)?;

    match parsed_exit_type {
        ExitType::Fixed => {
            if exit_sl_pct <= 0.0 {
                return value_error("FIXED requiere exit_sl_pct mayor que 0.");
            }
            if exit_tp_pct <= 0.0 {
                return value_error("FIXED requiere exit_tp_pct mayor que 0.");
            }
        }
        ExitType::Bars => {
            if exit_velas == 0 {
                return value_error("BARS requiere exit_velas mayor que 0.");
            }
        }
    }

    Ok(parsed_exit_type)
}

fn value_error<T>(message: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(message.into()))
}

/// Módulo Python. El nombre DEBE coincidir con lib.name en Cargo.toml.
#[pymodule]
fn motor_backtesting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_trades, m)?)?;
    m.add_class::<TradeResult>()?;
    m.add_class::<SimResult>()?;
    Ok(())
}

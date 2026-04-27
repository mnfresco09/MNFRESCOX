// ---------------------------------------------------------------------------
// lib.rs — Bindings PyO3 con buffers NumPy zero-copy y liberación de GIL.
//
// Funciones expuestas a Python:
//   - simulate_metrics(...)  → Metricas         (Optuna usa esta en cada trial)
//   - simulate_full(...)     → SimResultFull    (replay para reportes)
//
// El motor sólo expone simulación y gestión de capital. Los indicadores
// son responsabilidad exclusiva de cada estrategia (NUCLEO/base_estrategia.py).
// La simulación libera el GIL durante el cómputo, lo que permite escalar
// con `n_jobs > 1`.
// ---------------------------------------------------------------------------

mod capital;
mod simulador;
mod tipos;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use tipos::{ExitType, Metricas, SimConfig, SimResultFull, VelasSoA};

// ─── helpers ────────────────────────────────────────────────────────────────

fn value_error<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

fn validar_longitudes(
    n: usize,
    opens: usize,
    highs: usize,
    lows: usize,
    closes: usize,
    senales: usize,
    salidas_custom: usize,
) -> PyResult<()> {
    if opens != n || highs != n || lows != n || closes != n || senales != n || salidas_custom != n {
        return value_error(
            "Todas las columnas deben tener el mismo número de filas que `timestamps`.",
        );
    }
    if n < 2 {
        return value_error("Se necesitan al menos 2 velas para simular.");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn construir_config(
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
) -> PyResult<SimConfig> {
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
        return value_error("comision_lados debe ser 1 o 2.");
    }
    if !exit_sl_pct.is_finite() || exit_sl_pct < 0.0 {
        return value_error("exit_sl_pct debe ser finito y >= 0.");
    }
    if !exit_tp_pct.is_finite() || exit_tp_pct < 0.0 {
        return value_error("exit_tp_pct debe ser finito y >= 0.");
    }
    let parsed = ExitType::from_str(exit_type).map_err(pyo3::exceptions::PyValueError::new_err)?;
    match parsed {
        ExitType::Fixed => {
            if exit_sl_pct <= 0.0 {
                return value_error("FIXED requiere exit_sl_pct > 0.");
            }
            if exit_tp_pct <= 0.0 {
                return value_error("FIXED requiere exit_tp_pct > 0.");
            }
        }
        ExitType::Bars => {
            if exit_velas == 0 {
                return value_error("BARS requiere exit_velas > 0.");
            }
        }
        ExitType::Custom => {
            if exit_sl_pct <= 0.0 {
                return value_error("CUSTOM requiere exit_sl_pct > 0.");
            }
            if exit_tp_pct > 0.0 {
                return value_error("CUSTOM no usa exit_tp_pct; debe ser 0.");
            }
            if exit_velas != 0 {
                return value_error("CUSTOM no usa exit_velas; debe ser 0.");
            }
        }
    }
    Ok(SimConfig {
        saldo_inicial,
        saldo_por_trade,
        apalancamiento,
        saldo_minimo,
        comision_pct,
        comision_lados,
        exit_type: parsed,
        exit_sl_pct,
        exit_tp_pct,
        exit_velas,
    })
}

// ─── simulación slim (Optuna) ───────────────────────────────────────────────

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    timestamps, opens, highs, lows, closes, senales, salidas_custom,
    saldo_inicial, saldo_por_trade, apalancamiento, saldo_minimo,
    comision_pct, comision_lados,
    exit_type, exit_sl_pct, exit_tp_pct, exit_velas
))]
fn simulate_metrics<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, i64>,
    opens: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    closes: PyReadonlyArray1<'py, f64>,
    senales: PyReadonlyArray1<'py, i8>,
    salidas_custom: PyReadonlyArray1<'py, i8>,
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
) -> PyResult<Metricas> {
    let ts = timestamps.as_slice()?;
    let op = opens.as_slice()?;
    let hi = highs.as_slice()?;
    let lo = lows.as_slice()?;
    let cl = closes.as_slice()?;
    let se = senales.as_slice()?;
    let sx = salidas_custom.as_slice()?;
    validar_longitudes(ts.len(), op.len(), hi.len(), lo.len(), cl.len(), se.len(), sx.len())?;

    let config = construir_config(
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

    let velas = VelasSoA {
        timestamps: ts,
        opens: op,
        highs: hi,
        lows: lo,
        closes: cl,
    };

    let metricas = py.detach(|| simulador::simular_metricas(velas, se, sx, &config));
    Ok(metricas)
}

// ─── simulación full (replay para reportes) ─────────────────────────────────

/// Resultado columnar de un replay. Los arrays se materializan a numpy en
/// el momento de acceder al atributo correspondiente.
#[pyclass]
struct SimResult {
    inner: Option<SimResultFull>,
}

#[pymethods]
impl SimResult {
    #[getter]
    fn metricas(&self) -> PyResult<Metricas> {
        Ok(self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("SimResult ya consumido"))?
            .metricas
            .clone())
    }

    #[getter] fn saldo_inicial(&self) -> PyResult<f64> { self.metricas().map(|m| m.saldo_inicial) }
    #[getter] fn saldo_final(&self)   -> PyResult<f64> { self.metricas().map(|m| m.saldo_final) }
    #[getter] fn parado_por_saldo(&self) -> PyResult<bool> { self.metricas().map(|m| m.parado_por_saldo) }

    fn equity_curve<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let r = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SimResult ya consumido")
        })?;
        Ok(PyArray1::from_slice(py, &r.equity_curve))
    }

    /// Devuelve un dict de numpy arrays con todas las columnas de trades.
    /// Consume el SimResult: tras llamarlo, los buffers internos quedan vacíos
    /// para liberar memoria inmediatamente.
    fn take_trades<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let r = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SimResult ya consumido")
        })?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("idx_senal", r.idx_senal.into_pyarray(py))?;
        dict.set_item("idx_entrada", r.idx_entrada.into_pyarray(py))?;
        dict.set_item("idx_salida", r.idx_salida.into_pyarray(py))?;
        dict.set_item("ts_senal", r.ts_senal.into_pyarray(py))?;
        dict.set_item("ts_entrada", r.ts_entrada.into_pyarray(py))?;
        dict.set_item("ts_salida", r.ts_salida.into_pyarray(py))?;
        dict.set_item("direccion", r.direccion.into_pyarray(py))?;
        dict.set_item("precio_entrada", r.precio_entrada.into_pyarray(py))?;
        dict.set_item("precio_salida", r.precio_salida.into_pyarray(py))?;
        dict.set_item("colateral", r.colateral.into_pyarray(py))?;
        dict.set_item("tamano_posicion", r.tamano_posicion.into_pyarray(py))?;
        dict.set_item("comision_total", r.comision_total.into_pyarray(py))?;
        dict.set_item("pnl", r.pnl.into_pyarray(py))?;
        dict.set_item("roi", r.roi.into_pyarray(py))?;
        dict.set_item("saldo_post", r.saldo_post.into_pyarray(py))?;
        dict.set_item("motivo_salida", r.motivo_salida.into_pyarray(py))?;
        dict.set_item("duracion_velas", r.duracion_velas.into_pyarray(py))?;
        dict.set_item("equity_curve", r.equity_curve.into_pyarray(py))?;
        Ok(dict)
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    timestamps, opens, highs, lows, closes, senales, salidas_custom,
    saldo_inicial, saldo_por_trade, apalancamiento, saldo_minimo,
    comision_pct, comision_lados,
    exit_type, exit_sl_pct, exit_tp_pct, exit_velas
))]
fn simulate_full<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, i64>,
    opens: PyReadonlyArray1<'py, f64>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    closes: PyReadonlyArray1<'py, f64>,
    senales: PyReadonlyArray1<'py, i8>,
    salidas_custom: PyReadonlyArray1<'py, i8>,
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
    let ts = timestamps.as_slice()?;
    let op = opens.as_slice()?;
    let hi = highs.as_slice()?;
    let lo = lows.as_slice()?;
    let cl = closes.as_slice()?;
    let se = senales.as_slice()?;
    let sx = salidas_custom.as_slice()?;
    validar_longitudes(ts.len(), op.len(), hi.len(), lo.len(), cl.len(), se.len(), sx.len())?;

    let config = construir_config(
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

    let velas = VelasSoA {
        timestamps: ts,
        opens: op,
        highs: hi,
        lows: lo,
        closes: cl,
    };

    let full = py.detach(|| simulador::simular_full(velas, se, sx, &config));
    Ok(SimResult { inner: Some(full) })
}

// ─── módulo ────────────────────────────────────────────────────────────────

#[pymodule]
fn motor_backtesting(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_full, m)?)?;
    m.add_class::<Metricas>()?;
    m.add_class::<SimResult>()?;
    Ok(())
}

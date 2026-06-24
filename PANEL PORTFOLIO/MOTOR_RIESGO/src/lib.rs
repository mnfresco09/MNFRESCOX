// ---------------------------------------------------------------------------
// lib.rs — Bindings PyO3 del MICRO-MOTOR DE RIESGO (PANEL PORTFOLIO).
//
// Funciones expuestas a Python (vía RIESGO/motor_bindings.py):
//   - fhs(residuos, sigma_next, niveles) -> (var95, var99, cvar95, cvar99)
//   - montecarlo(retornos, horizonte, n_traj, percentiles, seed)
//        -> (sendas[n_perc, horizonte], prob_perdida, cdar, ret_mediano, p5)
//
// AISLADO de PANEL BACKTESTING/MOTOR: crate y nombre de módulo propios. Los
// buffers NumPy viajan zero-copy; el cómputo libera el GIL (allow_threads) para
// que rayon paralelice sin bloquear el intérprete.
// ---------------------------------------------------------------------------

// Los módulos internos se renombran (fhs_core / mc_core) para no colisionar con
// las #[pyfunction] `fhs` y `montecarlo`, que en pyo3 0.28 generan un tipo del
// mismo nombre en este espacio de nombres.
#[path = "fhs.rs"]
mod fhs_core;
#[path = "montecarlo.rs"]
mod mc_core;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

fn value_error<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(pyo3::exceptions::PyValueError::new_err(msg.into()))
}

/// Filtered Historical Simulation: VaR/CVaR 95 y 99 a T+1.
#[pyfunction]
#[pyo3(signature = (residuos, sigma_next, niveles))]
fn fhs<'py>(
    py: Python<'py>,
    residuos: PyReadonlyArray1<'py, f64>,
    sigma_next: f64,
    niveles: PyReadonlyArray1<'py, f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    let res = residuos.as_slice()?;
    let niv = niveles.as_slice()?;
    if res.len() < 10 {
        return value_error("FHS requiere al menos 10 residuos.");
    }
    if !sigma_next.is_finite() || sigma_next <= 0.0 {
        return value_error("sigma_next debe ser finito y positivo.");
    }
    let salida = py.detach(|| fhs_core::filtered_historical_simulation(res, sigma_next, niv));
    Ok(salida)
}

/// Monte Carlo por bootstrapping: percentiles del fan chart + agregados.
#[pyfunction]
#[pyo3(signature = (retornos, horizonte, n_trayectorias, percentiles, seed))]
fn montecarlo<'py>(
    py: Python<'py>,
    retornos: PyReadonlyArray1<'py, f64>,
    horizonte: usize,
    n_trayectorias: usize,
    percentiles: PyReadonlyArray1<'py, f64>,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, f64, f64, f64)> {
    let ret = retornos.as_slice()?;
    let perc = percentiles.as_slice()?;
    if ret.len() < 30 {
        return value_error("Monte Carlo requiere al menos 30 retornos históricos.");
    }
    if horizonte == 0 || n_trayectorias == 0 {
        return value_error("horizonte y n_trayectorias deben ser positivos.");
    }

    let perc_vec = perc.to_vec();
    let resumen = py.detach(|| {
        mc_core::simular(ret, horizonte, n_trayectorias, &perc_vec, seed)
    });

    // Vec<Vec<f64>> [n_perc][horizonte] → Array2 (n_perc, horizonte) → numpy.
    let n_perc = resumen.sendas.len();
    let mut plano = Vec::with_capacity(n_perc * horizonte);
    for fila in &resumen.sendas {
        plano.extend_from_slice(fila);
    }
    let arr = Array2::from_shape_vec((n_perc, horizonte), plano)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok((
        arr.into_pyarray(py),
        resumen.prob_perdida,
        resumen.cdar,
        resumen.retorno_mediano,
        resumen.perdida_p5,
    ))
}

#[pymodule]
fn motor_riesgo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fhs, m)?)?;
    m.add_function(wrap_pyfunction!(montecarlo, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// montecarlo.rs — Simulación futura por bootstrapping con reemplazo.
//
// Genera N trayectorias del capital a `horizonte` días remuestreando con
// reemplazo de los retornos históricos diarios. Paraleliza con `rayon`. NUNCA
// devuelve la matriz completa de trayectorias: calcula internamente y entrega
// solo agregados:
//   - sendas de percentiles (fan chart): para cada día, los percentiles del
//     capital base 1 entre todas las trayectorias.
//   - probabilidad de pérdida a horizonte: P(retorno_final < 0).
//   - CDaR (Conditional Drawdown at Risk) 95%: media del peor 5% de drawdowns
//     máximos por trayectoria.
//   - retorno mediano y pérdida P5 a horizonte.
// ---------------------------------------------------------------------------

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Resultado agregado de la simulación.
pub struct ResumenSimulacion {
    pub sendas: Vec<Vec<f64>>, // [percentil][dia] capital base 1
    pub prob_perdida: f64,
    pub cdar: f64,
    pub retorno_mediano: f64,
    pub perdida_p5: f64,
}

/// Percentil (estilo numpy "linear") sobre una muestra YA ordenada.
fn percentil_ordenado(ordenados: &[f64], p: f64) -> f64 {
    let n = ordenados.len();
    if n == 0 {
        return f64::NAN;
    }
    let pos = (p / 100.0) * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    ordenados[lo] + frac * (ordenados[hi] - ordenados[lo])
}

pub fn simular(
    retornos: &[f64],
    horizonte: usize,
    n_traj: usize,
    percentiles: &[f64],
    seed: u64,
) -> ResumenSimulacion {
    let n_ret = retornos.len();

    // Cada trayectoria devuelve: (capital_diario[horizonte], ret_final, max_drawdown).
    // Semilla por trayectoria derivada de la semilla base → reproducible y paralelo.
    let trayectorias: Vec<(Vec<f64>, f64, f64)> = (0..n_traj)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            let mut capital = 1.0_f64;
            let mut pico = 1.0_f64;
            let mut max_dd = 0.0_f64;
            let mut senda = Vec::with_capacity(horizonte);
            for _ in 0..horizonte {
                let idx = rng.gen_range(0..n_ret);
                capital *= 1.0 + retornos[idx];
                if capital > pico {
                    pico = capital;
                }
                let dd = capital / pico - 1.0;
                if dd < max_dd {
                    max_dd = dd;
                }
                senda.push(capital);
            }
            (senda, capital - 1.0, max_dd)
        })
        .collect();

    // Sendas de percentiles: para cada día, ordenar capital entre trayectorias.
    let mut sendas: Vec<Vec<f64>> = vec![vec![0.0; horizonte]; percentiles.len()];
    let mut columna: Vec<f64> = vec![0.0; n_traj];
    for d in 0..horizonte {
        for (t, tr) in trayectorias.iter().enumerate() {
            columna[t] = tr.0[d];
        }
        columna.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (pi, &p) in percentiles.iter().enumerate() {
            sendas[pi][d] = percentil_ordenado(&columna, p);
        }
    }

    // Retornos finales y drawdowns.
    let mut ret_final: Vec<f64> = trayectorias.iter().map(|t| t.1).collect();
    let mut max_dd: Vec<f64> = trayectorias.iter().map(|t| t.2).collect();
    let prob_perdida = ret_final.iter().filter(|&&r| r < 0.0).count() as f64 / n_traj as f64;

    ret_final.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    max_dd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let retorno_mediano = percentil_ordenado(&ret_final, 50.0);
    let perdida_p5 = percentil_ordenado(&ret_final, 5.0);

    // CDaR 95%: media del peor 5% de los drawdowns máximos (más negativos).
    let umbral = percentil_ordenado(&max_dd, 5.0);
    let peores: Vec<f64> = max_dd.iter().copied().filter(|&x| x <= umbral).collect();
    let cdar = if peores.is_empty() {
        umbral
    } else {
        peores.iter().sum::<f64>() / peores.len() as f64
    };

    ResumenSimulacion {
        sendas,
        prob_perdida,
        cdar,
        retorno_mediano,
        perdida_p5,
    }
}

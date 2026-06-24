// ---------------------------------------------------------------------------
// fhs.rs — Filtered Historical Simulation (VaR / CVaR a T+1).
//
// Recibe residuos ESTANDARIZADOS (z_t = r_t / sigma_t) y la volatilidad táctica
// de mañana (sigma_{T+1}). Escala los residuos por sigma_{T+1}, los ordena y lee
// el VaR (cuantil) y el CVaR (media de la cola) para cada nivel de confianza.
//
// VaR/CVaR se devuelven como retornos NEGATIVOS (pérdida en la cola). Son
// estimaciones bajo los supuestos del modelo, NUNCA "pérdida máxima".
// ---------------------------------------------------------------------------

/// Calcula (VaR, CVaR) para un nivel de confianza dado sobre una muestra YA
/// ordenada de menor a mayor.
fn var_cvar_ordenado(ordenados: &[f64], nivel: f64) -> (f64, f64) {
    let n = ordenados.len();
    let alpha = 1.0 - nivel; // cola izquierda
    // Índice del cuantil alpha (interpolación lineal estilo numpy "linear").
    let pos = alpha * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    let var = ordenados[lo] + frac * (ordenados[hi] - ordenados[lo]);
    // CVaR = media de los valores <= VaR.
    let mut suma = 0.0;
    let mut cuenta = 0usize;
    for &x in ordenados {
        if x <= var {
            suma += x;
            cuenta += 1;
        }
    }
    let cvar = if cuenta > 0 { suma / cuenta as f64 } else { var };
    (var, cvar)
}

/// Devuelve (var95, var99, cvar95, cvar99) escalando residuos por sigma_next.
/// `niveles` se asume = [nivel_95, nivel_99] (en ese orden).
pub fn filtered_historical_simulation(
    residuos: &[f64],
    sigma_next: f64,
    niveles: &[f64],
) -> (f64, f64, f64, f64) {
    let mut escalados: Vec<f64> = residuos.iter().map(|z| z * sigma_next).collect();
    escalados.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n95 = *niveles.get(0).unwrap_or(&0.95);
    let n99 = *niveles.get(1).unwrap_or(&0.99);
    let (var95, cvar95) = var_cvar_ordenado(&escalados, n95);
    let (var99, cvar99) = var_cvar_ordenado(&escalados, n99);
    (var95, var99, cvar95, cvar99)
}

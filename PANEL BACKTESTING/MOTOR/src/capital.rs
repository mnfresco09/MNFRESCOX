// ---------------------------------------------------------------------------
// capital.rs — Cálculos de capital, comisiones y niveles de salida.
//
// REGLA DE COMISIONES:
//   La comisión es un porcentaje del NOCIONAL = tamaño_posición × precio.
//   El nocional de entrada y el de salida son distintos cuando el precio
//   se mueve, así que cada lado se calcula con su propio precio:
//
//       comisión_entrada = tamaño × precio_entrada × comision_pct
//       comisión_salida  = tamaño × precio_salida  × comision_pct  (si lados==2)
//
//   `comision_lados`:
//     1 → sólo se cobra apertura.
//     2 → se cobra apertura y cierre (ambos sobre su nocional real).
// ---------------------------------------------------------------------------

use crate::tipos::Direccion;

#[inline]
pub fn calcular_tamano_posicion(colateral: f64, apalancamiento: f64, precio_entrada: f64) -> f64 {
    (colateral * apalancamiento) / precio_entrada
}

/// Comisión de un único lado (apertura o cierre) sobre su nocional real.
///
/// nocional = tamaño_posición × precio_lado.
#[inline]
pub fn comision_lado(tamano_posicion: f64, precio: f64, comision_pct: f64) -> f64 {
    tamano_posicion * precio * comision_pct
}

#[inline]
pub fn calcular_pnl_bruto(
    direccion: Direccion,
    tamano_posicion: f64,
    precio_entrada: f64,
    precio_salida: f64,
) -> f64 {
    match direccion {
        Direccion::Long => tamano_posicion * (precio_salida - precio_entrada),
        Direccion::Short => tamano_posicion * (precio_entrada - precio_salida),
    }
}

#[inline]
pub fn calcular_precio_sl(
    direccion: Direccion,
    precio_entrada: f64,
    sl_pct: f64,
    apalancamiento: f64,
) -> f64 {
    let mov = sl_pct / 100.0 / apalancamiento;
    match direccion {
        Direccion::Long => precio_entrada * (1.0 - mov),
        Direccion::Short => precio_entrada * (1.0 + mov),
    }
}

#[inline]
pub fn calcular_precio_tp(
    direccion: Direccion,
    precio_entrada: f64,
    tp_pct: f64,
    apalancamiento: f64,
) -> f64 {
    let mov = tp_pct / 100.0 / apalancamiento;
    match direccion {
        Direccion::Long => precio_entrada * (1.0 + mov),
        Direccion::Short => precio_entrada * (1.0 - mov),
    }
}

#[inline]
pub fn calcular_distancia_por_pct_colateral(
    precio_entrada: f64,
    pct_colateral: f64,
    apalancamiento: f64,
) -> f64 {
    precio_entrada * pct_colateral / 100.0 / apalancamiento
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tamano_posicion() {
        let t = calcular_tamano_posicion(500.0, 10.0, 50_000.0);
        assert!((t - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_comision_lado_es_pct_del_nocional() {
        // tamaño 0.1 BTC × precio 50_000 × 0.05% = 2.5 USD por lado
        let c = comision_lado(0.1, 50_000.0, 0.0005);
        assert!((c - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_comision_salida_mayor_si_precio_sube() {
        // Long: precio sube → nocional de salida mayor → comisión mayor.
        let c_in = comision_lado(0.1, 50_000.0, 0.0005);
        let c_out = comision_lado(0.1, 75_000.0, 0.0005);
        assert!(c_out > c_in);
        assert!((c_out - 3.75).abs() < 1e-10);
    }

    #[test]
    fn test_pnl_long() {
        let p = calcular_pnl_bruto(Direccion::Long, 0.1, 50_000.0, 51_000.0);
        assert!((p - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_precio_sl_long() {
        let sl = calcular_precio_sl(Direccion::Long, 50_000.0, 20.0, 10.0);
        assert!((sl - 49_000.0).abs() < 1e-6);
    }

    #[test]
    fn test_precio_tp_long() {
        let tp = calcular_precio_tp(Direccion::Long, 50_000.0, 40.0, 10.0);
        assert!((tp - 52_000.0).abs() < 1e-6);
    }

    #[test]
    fn test_distancia_pct_colateral() {
        let distancia = calcular_distancia_por_pct_colateral(50_000.0, 6.0, 10.0);
        assert!((distancia - 300.0).abs() < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// capital.rs — Gestión de capital, comisiones y apalancamiento
// ---------------------------------------------------------------------------

use crate::tipos::Direccion;

/// Tamaño de posición = (colateral × apalancamiento) / precio_entrada
pub fn calcular_tamaño_posicion(colateral: f64, apalancamiento: f64, precio_entrada: f64) -> f64 {
    (colateral * apalancamiento) / precio_entrada
}

/// Comisiones totales en USD. Se calcula sobre el valor nocional.
pub fn calcular_comision(
    colateral: f64,
    apalancamiento: f64,
    comision_pct: f64,
    comision_lados: u8,
) -> f64 {
    let nocional = colateral * apalancamiento;
    nocional * comision_pct * comision_lados as f64
}

/// P&L bruto (antes de comisiones).
pub fn calcular_pnl_bruto(
    direccion: Direccion,
    tamaño_posicion: f64,
    precio_entrada: f64,
    precio_salida: f64,
) -> f64 {
    match direccion {
        Direccion::Long  => tamaño_posicion * (precio_salida - precio_entrada),
        Direccion::Short => tamaño_posicion * (precio_entrada - precio_salida),
    }
}

/// Precio exacto donde se activa el Stop Loss.
pub fn calcular_precio_sl(
    direccion: Direccion,
    precio_entrada: f64,
    sl_pct: f64,
    apalancamiento: f64,
) -> f64 {
    let mov = sl_pct / 100.0 / apalancamiento;
    match direccion {
        Direccion::Long  => precio_entrada * (1.0 - mov),
        Direccion::Short => precio_entrada * (1.0 + mov),
    }
}

/// Precio exacto donde se activa el Take Profit.
pub fn calcular_precio_tp(
    direccion: Direccion,
    precio_entrada: f64,
    tp_pct: f64,
    apalancamiento: f64,
) -> f64 {
    let mov = tp_pct / 100.0 / apalancamiento;
    match direccion {
        Direccion::Long  => precio_entrada * (1.0 + mov),
        Direccion::Short => precio_entrada * (1.0 - mov),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tamaño_posicion() {
        let t = calcular_tamaño_posicion(500.0, 10.0, 50_000.0);
        assert!((t - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_comision_dos_lados() {
        let c = calcular_comision(500.0, 10.0, 0.0005, 2);
        assert!((c - 5.0).abs() < 1e-10);
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
}

// ---------------------------------------------------------------------------
// tipos.rs — Estructuras de datos del motor de simulación
//
// Sólo se exponen a Python las structs estrictamente necesarias:
//   - Metricas        : escalares calculados durante la simulación.
//   - SimResultFull   : métricas + columnas de trades + curva de equity.
//
// El motor NO devuelve trades como objetos Python por trial: en el modo
// "slim" (Optuna) sólo viaja Metricas. En el modo "full" (replay para
// reportes) se devuelven columnas en `Vec<T>`, que el wrapper convierte
// a numpy arrays sin copia adicional.
// ---------------------------------------------------------------------------

use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direccion {
    Long,
    Short,
}

impl Direccion {
    pub fn from_signal(val: i8) -> Result<Self, String> {
        match val {
            1 => Ok(Direccion::Long),
            -1 => Ok(Direccion::Short),
            _ => Err(format!(
                "Señal inválida: {val}. Solo se aceptan 0, 1 (LONG) o -1 (SHORT)."
            )),
        }
    }

    pub fn as_i8(self) -> i8 {
        match self {
            Direccion::Long => 1,
            Direccion::Short => -1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExitType {
    Fixed,
    Bars,
    Trailing,
    Custom,
}

impl ExitType {
    pub fn from_str(val: &str) -> Result<Self, String> {
        match val {
            "FIXED" => Ok(ExitType::Fixed),
            "BARS" => Ok(ExitType::Bars),
            "TRAILING" => Ok(ExitType::Trailing),
            "CUSTOM" => Ok(ExitType::Custom),
            _ => Err(format!(
                "EXIT_TYPE inválido: '{val}'. Opciones: FIXED, BARS, TRAILING, CUSTOM."
            )),
        }
    }
}

/// Códigos compactos para el motivo de salida.
/// Se exponen como u8 para mantener el array de motivos pequeño en RAM.
/// El wrapper Python traduce estos códigos a strings cuando es necesario.
pub mod motivo {
    pub const SL: u8 = 0;
    pub const TP: u8 = 1;
    pub const BARS: u8 = 2;
    pub const CUSTOM: u8 = 3;
    pub const TRAILING: u8 = 4;
    pub const END: u8 = 5;
}

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub saldo_inicial: f64,
    pub saldo_por_trade: f64,
    pub apalancamiento: f64,
    pub saldo_minimo: f64,
    pub comision_pct: f64,
    pub comision_lados: u8,
    pub exit_type: ExitType,
    pub exit_sl_pct: f64,
    pub exit_tp_pct: f64,
    pub exit_velas: usize,
    pub exit_trail_act_pct: f64,
    pub exit_trail_dist_pct: f64,
    pub paridad_riesgo: bool,
    pub paridad_riesgo_max_pct: f64,
    pub paridad_apalancamiento_min: f64,
    pub paridad_apalancamiento_max: f64,
    pub exit_sl_ewma_mult: f64,
    pub exit_tp_ewma_mult: f64,
    pub exit_trail_act_ewma_mult: f64,
    pub exit_trail_dist_ewma_mult: f64,
    pub paridad_skip_bajo_min: bool,
}

/// Vista SoA (Struct-of-Arrays) sobre las velas. No copia datos: las slices
/// vienen directamente de los buffers NumPy enviados desde Python.
#[derive(Debug, Clone, Copy)]
pub struct VelasSoA<'a> {
    pub timestamps: &'a [i64],
    pub opens: &'a [f64],
    pub highs: &'a [f64],
    pub lows: &'a [f64],
    pub closes: &'a [f64],
}

impl<'a> VelasSoA<'a> {
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
}

/// Métricas escalares de una simulación. Esto es lo único que viaja a
/// Python en el modo Optuna (un struct pequeño por trial).
#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
pub struct Metricas {
    #[pyo3(get)]
    pub saldo_inicial: f64,
    #[pyo3(get)]
    pub saldo_final: f64,
    #[pyo3(get)]
    pub total_trades: u64,
    #[pyo3(get)]
    pub trades_long: u64,
    #[pyo3(get)]
    pub trades_short: u64,
    #[pyo3(get)]
    pub trades_ganadores: u64,
    #[pyo3(get)]
    pub trades_perdedores: u64,
    #[pyo3(get)]
    pub trades_neutros: u64,
    #[pyo3(get)]
    pub win_rate: f64,
    #[pyo3(get)]
    pub roi_total: f64,
    #[pyo3(get)]
    pub expectancy: f64,
    #[pyo3(get)]
    pub pnl_bruto_total: f64,
    #[pyo3(get)]
    pub pnl_total: f64,
    #[pyo3(get)]
    pub pnl_promedio: f64,
    #[pyo3(get)]
    pub max_drawdown: f64,
    /// f64::INFINITY si hay ganancias y no hay pérdidas; 0.0 si no hay ganancias.
    #[pyo3(get)]
    pub profit_factor: f64,
    #[pyo3(get)]
    pub sharpe_ratio: f64,
    #[pyo3(get)]
    pub duracion_media_seg: f64,
    #[pyo3(get)]
    pub duracion_media_velas: f64,
    #[pyo3(get)]
    pub parado_por_saldo: bool,
}

impl Metricas {
    pub fn vacia(saldo_inicial: f64) -> Self {
        Metricas {
            saldo_inicial,
            saldo_final: saldo_inicial,
            total_trades: 0,
            trades_long: 0,
            trades_short: 0,
            trades_ganadores: 0,
            trades_perdedores: 0,
            trades_neutros: 0,
            win_rate: 0.0,
            roi_total: 0.0,
            expectancy: 0.0,
            pnl_bruto_total: 0.0,
            pnl_total: 0.0,
            pnl_promedio: 0.0,
            max_drawdown: 0.0,
            profit_factor: 0.0,
            sharpe_ratio: 0.0,
            duracion_media_seg: 0.0,
            duracion_media_velas: 0.0,
            parado_por_saldo: false,
        }
    }
}

/// Resultado completo (replay): métricas + columnas de trades + equity_curve.
/// Sólo se construye para los top-N trials que alimentan reportes.
///
/// Los `Vec<T>` se exponen a Python como numpy arrays vía el módulo `lib.rs`
/// usando `PyArray1::from_vec_bound`, que mueve la propiedad sin copia extra.
#[derive(Debug)]
pub struct SimResultFull {
    pub metricas: Metricas,
    // Columnas de trades — cada una con length == metricas.total_trades
    pub idx_senal: Vec<u64>,
    pub idx_entrada: Vec<u64>,
    pub idx_salida: Vec<u64>,
    pub ts_senal: Vec<i64>,
    pub ts_entrada: Vec<i64>,
    pub ts_salida: Vec<i64>,
    pub direccion: Vec<i8>,
    pub precio_entrada: Vec<f64>,
    pub precio_salida: Vec<f64>,
    pub colateral: Vec<f64>,
    pub apalancamiento: Vec<f64>,
    pub tamano_posicion: Vec<f64>,
    pub risk_vol_ewma: Vec<f64>,
    pub risk_sl_dist_pct: Vec<f64>,
    pub comision_total: Vec<f64>,
    pub pnl: Vec<f64>,
    pub roi: Vec<f64>,
    pub saldo_post: Vec<f64>,
    pub motivo_salida: Vec<u8>,
    pub duracion_velas: Vec<u64>,
    pub equity_curve: Vec<f64>,
}

impl SimResultFull {
    pub fn vacio(saldo_inicial: f64) -> Self {
        SimResultFull {
            metricas: Metricas::vacia(saldo_inicial),
            idx_senal: Vec::new(),
            idx_entrada: Vec::new(),
            idx_salida: Vec::new(),
            ts_senal: Vec::new(),
            ts_entrada: Vec::new(),
            ts_salida: Vec::new(),
            direccion: Vec::new(),
            precio_entrada: Vec::new(),
            precio_salida: Vec::new(),
            colateral: Vec::new(),
            apalancamiento: Vec::new(),
            tamano_posicion: Vec::new(),
            risk_vol_ewma: Vec::new(),
            risk_sl_dist_pct: Vec::new(),
            comision_total: Vec::new(),
            pnl: Vec::new(),
            roi: Vec::new(),
            saldo_post: Vec::new(),
            motivo_salida: Vec::new(),
            duracion_velas: Vec::new(),
            equity_curve: vec![saldo_inicial],
        }
    }

    pub fn reservar(&mut self, capacidad: usize) {
        self.idx_senal.reserve(capacidad);
        self.idx_entrada.reserve(capacidad);
        self.idx_salida.reserve(capacidad);
        self.ts_senal.reserve(capacidad);
        self.ts_entrada.reserve(capacidad);
        self.ts_salida.reserve(capacidad);
        self.direccion.reserve(capacidad);
        self.precio_entrada.reserve(capacidad);
        self.precio_salida.reserve(capacidad);
        self.colateral.reserve(capacidad);
        self.apalancamiento.reserve(capacidad);
        self.tamano_posicion.reserve(capacidad);
        self.risk_vol_ewma.reserve(capacidad);
        self.risk_sl_dist_pct.reserve(capacidad);
        self.comision_total.reserve(capacidad);
        self.pnl.reserve(capacidad);
        self.roi.reserve(capacidad);
        self.saldo_post.reserve(capacidad);
        self.motivo_salida.reserve(capacidad);
        self.duracion_velas.reserve(capacidad);
        self.equity_curve.reserve(capacidad);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direccion_rechaza_senal_invalida() {
        assert!(Direccion::from_signal(2).is_err());
    }

    #[test]
    fn test_exit_type_acepta_custom() {
        assert_eq!(ExitType::from_str("CUSTOM").unwrap(), ExitType::Custom);
    }

    #[test]
    fn test_exit_type_acepta_trailing() {
        assert_eq!(ExitType::from_str("TRAILING").unwrap(), ExitType::Trailing);
    }

    #[test]
    fn test_exit_type_rechaza_valor_invalido() {
        assert!(ExitType::from_str("INVALIDO").is_err());
    }
}

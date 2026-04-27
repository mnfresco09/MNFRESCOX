// ---------------------------------------------------------------------------
// simulador.rs — Loop principal vela a vela (SoA + acumuladores)
//
// REGLA CRÍTICA: la entrada ocurre SIEMPRE en el open de la vela N+1.
//
// Diseño de memoria:
//   - Las velas se procesan vía VelasSoA (slices del buffer numpy de Python).
//     No hay copia ni Vec<Vela>: cada acceso es a la columna directamente,
//     amigable para el cache.
//   - Métricas (max DD, profit factor, sharpe, expectancy, conteos…) se
//     acumulan al vuelo dentro del loop. No requieren almacenar trades.
//   - El modo "full" recibe un callback `on_trade` que decide si materializa
//     las columnas de cada trade. En modo "slim" el callback es no-op.
// ---------------------------------------------------------------------------

use crate::capital;
use crate::tipos::{
    motivo, Direccion, ExitType, Metricas, SimConfig, SimResultFull, VelasSoA,
};

struct TradeAbierto {
    idx_senal: usize,
    idx_entrada: usize,
    direccion: Direccion,
    precio_entrada: f64,
    colateral: f64,
    tamano_posicion: f64,
    /// Comisión cobrada al abrir (sobre nocional de entrada). Es fija y
    /// se conoce nada más entrar.
    comision_entrada: f64,
    /// % de comisión por lado. Se guarda para poder cobrar el lado de
    /// salida sobre su nocional real cuando el trade cierre.
    comision_pct: f64,
    /// 1 = sólo apertura, 2 = apertura + cierre.
    comision_lados: u8,
    precio_sl: f64,
    precio_tp: f64,
    max_velas: usize,
    usar_custom: bool,
}

#[derive(Clone, Copy)]
pub struct TradeEvent {
    pub idx_senal: usize,
    pub idx_entrada: usize,
    pub idx_salida: usize,
    pub ts_senal: i64,
    pub ts_entrada: i64,
    pub ts_salida: i64,
    pub direccion: Direccion,
    pub precio_entrada: f64,
    pub precio_salida: f64,
    pub colateral: f64,
    pub tamano_posicion: f64,
    pub comision_total: f64,
    pub pnl: f64,
    pub roi: f64,
    pub saldo_post: f64,
    pub motivo: u8,
    pub duracion_velas: usize,
}

struct Acumuladores {
    saldo_inicial: f64,
    saldo: f64,
    pnl_total: f64,
    pnl_bruto_total: f64,
    sum_retornos: f64,
    sum_retornos_sq: f64,
    sum_dur_us: i128,
    sum_dur_velas: u128,
    total_trades: u64,
    trades_long: u64,
    trades_short: u64,
    trades_ganadores: u64,
    trades_perdedores: u64,
    trades_neutros: u64,
    ganancias: f64,
    perdidas: f64,
    parado_por_saldo: bool,
    max_equity: f64,
    max_dd: f64,
}

impl Acumuladores {
    fn nuevo(saldo_inicial: f64) -> Self {
        Acumuladores {
            saldo_inicial,
            saldo: saldo_inicial,
            pnl_total: 0.0,
            pnl_bruto_total: 0.0,
            sum_retornos: 0.0,
            sum_retornos_sq: 0.0,
            sum_dur_us: 0,
            sum_dur_velas: 0,
            total_trades: 0,
            trades_long: 0,
            trades_short: 0,
            trades_ganadores: 0,
            trades_perdedores: 0,
            trades_neutros: 0,
            ganancias: 0.0,
            perdidas: 0.0,
            parado_por_saldo: false,
            max_equity: saldo_inicial,
            max_dd: 0.0,
        }
    }

    fn registrar(&mut self, ev: &TradeEvent) {
        self.total_trades += 1;
        match ev.direccion {
            Direccion::Long => self.trades_long += 1,
            Direccion::Short => self.trades_short += 1,
        }
        if ev.pnl > 0.0 {
            self.trades_ganadores += 1;
            self.ganancias += ev.pnl;
        } else if ev.pnl < 0.0 {
            self.trades_perdedores += 1;
            self.perdidas += -ev.pnl;
        } else {
            self.trades_neutros += 1;
        }
        self.pnl_total += ev.pnl;
        self.pnl_bruto_total += ev.pnl + ev.comision_total;
        self.sum_retornos += ev.roi;
        self.sum_retornos_sq += ev.roi * ev.roi;
        self.sum_dur_us += (ev.ts_salida as i128) - (ev.ts_entrada as i128);
        self.sum_dur_velas += ev.duracion_velas as u128;
        self.saldo = ev.saldo_post;
        if self.saldo > self.max_equity {
            self.max_equity = self.saldo;
        }
        if self.max_equity > 0.0 {
            let dd = (self.max_equity - self.saldo) / self.max_equity;
            if dd > self.max_dd {
                self.max_dd = dd;
            }
        }
    }

    fn finalizar(self) -> Metricas {
        let n = self.total_trades;
        if n == 0 {
            let mut m = Metricas::vacia(self.saldo_inicial);
            m.parado_por_saldo = self.parado_por_saldo;
            return m;
        }
        let nf = n as f64;
        let win_rate = self.trades_ganadores as f64 / nf;
        let pnl_promedio = self.pnl_total / nf;
        let expectancy = self.sum_retornos / nf;
        let roi_total = if self.saldo_inicial != 0.0 {
            self.pnl_total / self.saldo_inicial
        } else {
            0.0
        };
        let profit_factor = if self.perdidas == 0.0 {
            if self.ganancias > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            self.ganancias / self.perdidas
        };
        let sharpe_ratio = if n >= 2 {
            let media = self.sum_retornos / nf;
            // Varianza muestral: SUM(r^2)/n - media^2 daría poblacional;
            // ajustamos a (n-1) para coincidir con el cálculo previo Python.
            let var_pop = self.sum_retornos_sq / nf - media * media;
            let var_muestral = var_pop * nf / (nf - 1.0);
            let desviacion = if var_muestral > 0.0 {
                var_muestral.sqrt()
            } else {
                0.0
            };
            if desviacion > 0.0 {
                media / desviacion
            } else {
                0.0
            }
        } else {
            0.0
        };
        let duracion_media_seg = (self.sum_dur_us as f64 / nf) / 1_000_000.0;
        let duracion_media_velas = self.sum_dur_velas as f64 / nf;

        Metricas {
            saldo_inicial: self.saldo_inicial,
            saldo_final: self.saldo,
            total_trades: n,
            trades_long: self.trades_long,
            trades_short: self.trades_short,
            trades_ganadores: self.trades_ganadores,
            trades_perdedores: self.trades_perdedores,
            trades_neutros: self.trades_neutros,
            win_rate,
            roi_total,
            expectancy,
            pnl_bruto_total: self.pnl_bruto_total,
            pnl_total: self.pnl_total,
            pnl_promedio,
            max_drawdown: self.max_dd,
            profit_factor,
            sharpe_ratio,
            duracion_media_seg,
            duracion_media_velas,
            parado_por_saldo: self.parado_por_saldo,
        }
    }
}

/// Versión "slim": sólo devuelve métricas. Usada por Optuna en cada trial.
pub fn simular_metricas(
    velas: VelasSoA,
    senales: &[i8],
    salidas_custom: &[i8],
    config: &SimConfig,
) -> Metricas {
    let mut acc = Acumuladores::nuevo(config.saldo_inicial);
    simular_core(velas, senales, salidas_custom, config, &mut acc, |_| {});
    acc.finalizar()
}

/// Versión "full": métricas + columnas de trades + curva de equity.
/// Sólo se usa en el replay de los top-N trials para alimentar reportes.
pub fn simular_full(
    velas: VelasSoA,
    senales: &[i8],
    salidas_custom: &[i8],
    config: &SimConfig,
) -> SimResultFull {
    let mut acc = Acumuladores::nuevo(config.saldo_inicial);
    let mut full = SimResultFull::vacio(config.saldo_inicial);
    // Estimación grosera: como mucho un trade cada 2 velas.
    full.reservar((velas.len() / 2).max(64));

    simular_core(velas, senales, salidas_custom, config, &mut acc, |ev| {
        full.idx_senal.push(ev.idx_senal as u64);
        full.idx_entrada.push(ev.idx_entrada as u64);
        full.idx_salida.push(ev.idx_salida as u64);
        full.ts_senal.push(ev.ts_senal);
        full.ts_entrada.push(ev.ts_entrada);
        full.ts_salida.push(ev.ts_salida);
        full.direccion.push(ev.direccion.as_i8());
        full.precio_entrada.push(ev.precio_entrada);
        full.precio_salida.push(ev.precio_salida);
        full.colateral.push(ev.colateral);
        full.tamano_posicion.push(ev.tamano_posicion);
        full.comision_total.push(ev.comision_total);
        full.pnl.push(ev.pnl);
        full.roi.push(ev.roi);
        full.saldo_post.push(ev.saldo_post);
        full.motivo_salida.push(ev.motivo);
        full.duracion_velas.push(ev.duracion_velas as u64);
        full.equity_curve.push(ev.saldo_post);
    });

    full.metricas = acc.finalizar();
    full
}

fn simular_core<F>(
    velas: VelasSoA,
    senales: &[i8],
    salidas_custom: &[i8],
    config: &SimConfig,
    acc: &mut Acumuladores,
    mut on_trade: F,
) where
    F: FnMut(TradeEvent),
{
    let n = velas.len();
    debug_assert_eq!(n, senales.len(), "velas y senales deben coincidir");
    debug_assert_eq!(n, salidas_custom.len(), "velas y salidas_custom deben coincidir");

    let mut trade_abierto: Option<TradeAbierto> = None;
    let mut senal_pendiente: Option<(usize, Direccion)> = None;

    for i in 0..n {
        // 1. ABRIR
        if let Some((idx_senal, direccion)) = senal_pendiente.take() {
            if trade_abierto.is_none() {
                if acc.saldo < config.saldo_minimo || acc.saldo < config.saldo_por_trade {
                    acc.parado_por_saldo = true;
                    break;
                }
                let precio_entrada = velas.opens[i];
                let colateral = config.saldo_por_trade;
                let tamano = capital::calcular_tamano_posicion(
                    colateral,
                    config.apalancamiento,
                    precio_entrada,
                );
                let comision_entrada =
                    capital::comision_lado(tamano, precio_entrada, config.comision_pct);
                let precio_sl = if config.exit_sl_pct > 0.0 {
                    capital::calcular_precio_sl(
                        direccion,
                        precio_entrada,
                        config.exit_sl_pct,
                        config.apalancamiento,
                    )
                } else {
                    0.0
                };
                let precio_tp = if config.exit_type == ExitType::Fixed && config.exit_tp_pct > 0.0 {
                    capital::calcular_precio_tp(
                        direccion,
                        precio_entrada,
                        config.exit_tp_pct,
                        config.apalancamiento,
                    )
                } else {
                    0.0
                };
                let max_velas = if config.exit_type == ExitType::Bars {
                    config.exit_velas
                } else {
                    0
                };
                let usar_custom = config.exit_type == ExitType::Custom;

                trade_abierto = Some(TradeAbierto {
                    idx_senal,
                    idx_entrada: i,
                    direccion,
                    precio_entrada,
                    colateral,
                    tamano_posicion: tamano,
                    comision_entrada,
                    comision_pct: config.comision_pct,
                    comision_lados: config.comision_lados,
                    precio_sl,
                    precio_tp,
                    max_velas,
                    usar_custom,
                });
            }
        }

        // 2. CERRAR (si aplica)
        if let Some(ref trade) = trade_abierto {
            let velas_transcurridas = i - trade.idx_entrada + 1;
            if let Some((precio_salida, motivo)) = evaluar_salida(
                trade,
                velas.opens[i],
                velas.highs[i],
                velas.lows[i],
                velas.closes[i],
                velas_transcurridas,
                salidas_custom[i],
            ) {
                let ev = construir_evento(trade, &velas, i, precio_salida, motivo, acc.saldo);
                acc.registrar(&ev);
                on_trade(ev);
                trade_abierto = None;
                if acc.saldo < config.saldo_minimo {
                    acc.parado_por_saldo = true;
                    break;
                }
            }
        }

        // 3. NUEVA SEÑAL (se ejecuta en la siguiente vela)
        if trade_abierto.is_none() && senales[i] != 0 {
            if let Ok(direccion) = Direccion::from_signal(senales[i]) {
                senal_pendiente = Some((i, direccion));
            }
        }
    }

    // Liquidación al close de la última vela si quedó posición abierta.
    if let Some(trade) = trade_abierto.take() {
        let idx_salida = n - 1;
        let velas_transcurridas = idx_salida - trade.idx_entrada + 1;
        if velas_transcurridas > 0 {
            let ev = construir_evento(
                &trade,
                &velas,
                idx_salida,
                velas.closes[idx_salida],
                motivo::END,
                acc.saldo,
            );
            acc.registrar(&ev);
            on_trade(ev);
        }
    }
}

fn construir_evento(
    trade: &TradeAbierto,
    velas: &VelasSoA,
    idx_salida: usize,
    precio_salida: f64,
    motivo: u8,
    saldo_previo: f64,
) -> TradeEvent {
    // Comisión de salida sobre el nocional REAL de salida (precio_salida).
    // En long con beneficio el nocional sube → la comisión sube; en pérdida baja.
    // Esto elimina el sesgo que tenía la versión que cobraba ambos lados con el
    // nocional de entrada.
    let comision_salida = if trade.comision_lados == 2 {
        capital::comision_lado(trade.tamano_posicion, precio_salida, trade.comision_pct)
    } else {
        0.0
    };
    let comision_total = trade.comision_entrada + comision_salida;

    let pnl_bruto = capital::calcular_pnl_bruto(
        trade.direccion,
        trade.tamano_posicion,
        trade.precio_entrada,
        precio_salida,
    );
    let pnl_neto = pnl_bruto - comision_total;
    let saldo_post = saldo_previo + pnl_neto;
    let roi = if trade.colateral != 0.0 {
        pnl_neto / trade.colateral
    } else {
        0.0
    };
    TradeEvent {
        idx_senal: trade.idx_senal,
        idx_entrada: trade.idx_entrada,
        idx_salida,
        ts_senal: velas.timestamps[trade.idx_senal],
        ts_entrada: velas.timestamps[trade.idx_entrada],
        ts_salida: velas.timestamps[idx_salida],
        direccion: trade.direccion,
        precio_entrada: trade.precio_entrada,
        precio_salida,
        colateral: trade.colateral,
        tamano_posicion: trade.tamano_posicion,
        comision_total,
        pnl: pnl_neto,
        roi,
        saldo_post,
        motivo,
        duracion_velas: idx_salida - trade.idx_entrada + 1,
    }
}

#[inline]
fn evaluar_salida(
    trade: &TradeAbierto,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    velas_transcurridas: usize,
    salida_custom: i8,
) -> Option<(f64, u8)> {
    // 1. Stop Loss (peor caso dentro de la vela)
    if trade.precio_sl > 0.0 {
        match trade.direccion {
            Direccion::Long if low <= trade.precio_sl => {
                let precio = if open < trade.precio_sl { open } else { trade.precio_sl };
                return Some((precio, motivo::SL));
            }
            Direccion::Short if high >= trade.precio_sl => {
                let precio = if open > trade.precio_sl { open } else { trade.precio_sl };
                return Some((precio, motivo::SL));
            }
            _ => {}
        }
    }

    // 2. Salida CUSTOM (al close)
    if trade.usar_custom {
        match (trade.direccion, salida_custom) {
            (Direccion::Long, 1) | (Direccion::Short, -1) => {
                return Some((close, motivo::CUSTOM));
            }
            _ => {}
        }
    }

    // 3. Take Profit (mejor caso dentro de la vela)
    if trade.precio_tp > 0.0 {
        match trade.direccion {
            Direccion::Long if high >= trade.precio_tp => {
                let precio = if open > trade.precio_tp { open } else { trade.precio_tp };
                return Some((precio, motivo::TP));
            }
            Direccion::Short if low <= trade.precio_tp => {
                let precio = if open < trade.precio_tp { open } else { trade.precio_tp };
                return Some((precio, motivo::TP));
            }
            _ => {}
        }
    }

    // 4. Máximo de velas (BARS)
    if trade.max_velas > 0 && velas_transcurridas >= trade.max_velas {
        return Some((close, motivo::BARS));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_fixed() -> SimConfig {
        SimConfig {
            saldo_inicial: 10_000.0,
            saldo_por_trade: 500.0,
            apalancamiento: 10.0,
            saldo_minimo: 1_000.0,
            comision_pct: 0.0005,
            comision_lados: 2,
            exit_type: ExitType::Fixed,
            exit_sl_pct: 20.0,
            exit_tp_pct: 40.0,
            exit_velas: 0,
        }
    }

    fn cfg_custom() -> SimConfig {
        let mut c = cfg_fixed();
        c.exit_type = ExitType::Custom;
        c.exit_tp_pct = 0.0;
        c.exit_velas = 0;
        c
    }

    fn build_soa<'a>(
        ts: &'a [i64],
        o: &'a [f64],
        h: &'a [f64],
        l: &'a [f64],
        c: &'a [f64],
    ) -> VelasSoA<'a> {
        VelasSoA {
            timestamps: ts,
            opens: o,
            highs: h,
            lows: l,
            closes: c,
        }
    }

    #[test]
    fn test_entrada_en_vela_siguiente() {
        let ts = [1, 2, 3];
        let o = [100.0, 103.0, 108.0];
        let h = [105.0, 110.0, 200.0];
        let l = [95.0, 100.0, 100.0];
        let c = [102.0, 108.0, 150.0];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let senales = [1i8, 0, 0];
        let salidas = [0i8, 0, 0];
        let r = simular_full(velas, &senales, &salidas, &cfg_fixed());
        assert_eq!(r.metricas.total_trades, 1);
        assert_eq!(r.idx_senal[0], 0);
        assert_eq!(r.idx_entrada[0], 1);
        assert!((r.precio_entrada[0] - 103.0).abs() < 1e-10);
    }

    #[test]
    fn test_senales_ignoradas_con_trade_abierto() {
        let ts = [1, 2, 3, 4];
        let o = [100.0, 103.0, 104.0, 105.0];
        let h = [105.0, 105.0, 106.0, 200.0];
        let l = [95.0, 101.0, 103.0, 100.0];
        let c = [102.0, 104.0, 105.0, 150.0];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let senales = [1i8, 0, -1, 0];
        let salidas = [0i8, 0, 0, 0];
        let r = simular_full(velas, &senales, &salidas, &cfg_fixed());
        assert_eq!(r.metricas.total_trades, 1);
        assert_eq!(r.direccion[0], 1);
    }

    #[test]
    fn test_cierra_trade_abierto_al_final() {
        let ts = [1, 2, 3];
        let o = [100.0, 100.0, 101.0];
        let h = [101.0, 101.0, 102.0];
        let l = [99.0, 99.0, 100.0];
        let c = [100.0, 100.5, 101.5];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let r = simular_full(velas, &[1i8, 0, 0], &[0i8, 0, 0], &cfg_fixed());
        assert_eq!(r.metricas.total_trades, 1);
        assert_eq!(r.motivo_salida[0], motivo::END);
        assert!((r.precio_salida[0] - 101.5).abs() < 1e-10);
    }

    #[test]
    fn test_gap_long_ejecuta_sl_en_open() {
        let ts = [1, 2, 3];
        let o = [100.0, 100.0, 95.0];
        let h = [101.0, 101.0, 97.0];
        let l = [99.0, 99.0, 94.0];
        let c = [100.0, 100.5, 96.0];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let r = simular_full(velas, &[1i8, 0, 0], &[0i8, 0, 0], &cfg_fixed());
        assert_eq!(r.metricas.total_trades, 1);
        assert_eq!(r.motivo_salida[0], motivo::SL);
        assert!((r.precio_salida[0] - 95.0).abs() < 1e-10);
    }

    #[test]
    fn test_bars_cierra_por_numero_de_velas() {
        let ts = [1, 2, 3, 4];
        let o = [100.0, 100.0, 101.0, 102.0];
        let h = [101.0, 101.0, 102.0, 103.0];
        let l = [99.0, 99.0, 100.0, 101.0];
        let c = [100.0, 100.5, 101.5, 102.5];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let mut cfg = cfg_fixed();
        cfg.exit_type = ExitType::Bars;
        cfg.exit_tp_pct = 0.0;
        cfg.exit_velas = 2;
        let r = simular_full(velas, &[1i8, 0, 0, 0], &[0i8, 0, 0, 0], &cfg);
        assert_eq!(r.metricas.total_trades, 1);
        assert_eq!(r.motivo_salida[0], motivo::BARS);
    }

    #[test]
    fn test_custom_cierra_long_al_close() {
        let ts = [1, 2, 3, 4];
        let o = [100.0, 100.0, 101.0, 102.0];
        let h = [101.0, 101.0, 102.0, 103.0];
        let l = [99.0, 99.0, 100.0, 101.0];
        let c = [100.0, 100.5, 101.5, 102.5];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let r = simular_full(velas, &[1i8, 0, 0, 0], &[0i8, 0, 1, 0], &cfg_custom());
        assert_eq!(r.metricas.total_trades, 1);
        assert_eq!(r.motivo_salida[0], motivo::CUSTOM);
        assert!((r.precio_salida[0] - 101.5).abs() < 1e-10);
    }

    #[test]
    fn test_comision_salida_usa_nocional_real() {
        // Trade que termina al close de la última vela (motivo END), ganador.
        // Nocional entrada: tamaño × 100 ; nocional salida: tamaño × 110.
        // Con apalancamiento=1, colateral=100, tamaño = 1.0, comision_pct=1%:
        //   comisión entrada = 1.0 × 100 × 0.01 = 1.0
        //   comisión salida  = 1.0 × 110 × 0.01 = 1.1
        //   total            = 2.1   (no 2.0 como con la versión sesgada)
        //   pnl_bruto        = 1.0 × (110 - 100) = 10.0
        //   pnl_neto         = 10.0 - 2.1 = 7.9
        let ts = [1, 2, 3];
        let o = [100.0, 100.0, 110.0];
        let h = [101.0, 101.0, 110.0];
        let l = [99.0, 99.0, 100.0];
        let c = [100.0, 100.0, 110.0];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let cfg = SimConfig {
            saldo_inicial: 1_000.0,
            saldo_por_trade: 100.0,
            apalancamiento: 1.0,
            saldo_minimo: 0.0,
            comision_pct: 0.01,
            comision_lados: 2,
            exit_type: ExitType::Fixed,
            exit_sl_pct: 50.0,
            exit_tp_pct: 50.0,
            exit_velas: 0,
        };
        let r = simular_full(velas, &[1i8, 0, 0], &[0i8, 0, 0], &cfg);
        assert_eq!(r.metricas.total_trades, 1);
        assert!((r.comision_total[0] - 2.1).abs() < 1e-10);
        assert!((r.pnl[0] - 7.9).abs() < 1e-10);
    }

    #[test]
    fn test_comision_lados_uno_no_cobra_salida() {
        let ts = [1, 2, 3];
        let o = [100.0, 100.0, 110.0];
        let h = [101.0, 101.0, 110.0];
        let l = [99.0, 99.0, 100.0];
        let c = [100.0, 100.0, 110.0];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let cfg = SimConfig {
            saldo_inicial: 1_000.0,
            saldo_por_trade: 100.0,
            apalancamiento: 1.0,
            saldo_minimo: 0.0,
            comision_pct: 0.01,
            comision_lados: 1,
            exit_type: ExitType::Fixed,
            exit_sl_pct: 50.0,
            exit_tp_pct: 50.0,
            exit_velas: 0,
        };
        let r = simular_full(velas, &[1i8, 0, 0], &[0i8, 0, 0], &cfg);
        assert!((r.comision_total[0] - 1.0).abs() < 1e-10);
        assert!((r.pnl[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_metricas_slim_y_full_coinciden() {
        let ts = [1, 2, 3, 4, 5];
        let o = [100.0, 100.0, 101.0, 102.0, 103.0];
        let h = [101.0, 101.0, 102.0, 103.0, 104.0];
        let l = [99.0, 99.0, 100.0, 101.0, 102.0];
        let c = [100.0, 100.5, 101.5, 102.5, 103.5];
        let velas = build_soa(&ts, &o, &h, &l, &c);
        let s = [1i8, 0, 0, 0, 0];
        let x = [0i8; 5];
        let m_slim = simular_metricas(velas, &s, &x, &cfg_fixed());
        let m_full = simular_full(velas, &s, &x, &cfg_fixed()).metricas;
        assert_eq!(m_slim.total_trades, m_full.total_trades);
        assert!((m_slim.saldo_final - m_full.saldo_final).abs() < 1e-10);
        assert!((m_slim.pnl_total - m_full.pnl_total).abs() < 1e-10);
        assert!((m_slim.max_drawdown - m_full.max_drawdown).abs() < 1e-12);
        assert!((m_slim.sharpe_ratio - m_full.sharpe_ratio).abs() < 1e-12);
    }
}

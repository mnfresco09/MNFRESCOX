// ---------------------------------------------------------------------------
// simulador.rs — Loop principal vela a vela
//
// REGLA CRÍTICA: la entrada ocurre SIEMPRE en el open de la vela N+1,
// nunca en la vela de la señal. El sistema lo garantiza por construcción.
// ---------------------------------------------------------------------------

use crate::capital;
use crate::tipos::{Direccion, ExitType, SimConfig, SimResult, TradeResult, Vela};

/// Estado interno de un trade abierto. No se expone a Python.
struct TradeAbierto {
    idx_señal: usize,
    idx_entrada: usize,
    direccion: Direccion,
    precio_entrada: f64,
    colateral: f64,
    tamaño_posicion: f64,
    comision_total: f64,
    precio_sl: f64,
    precio_tp: f64,    // 0.0 si no aplica (BARS/CUSTOM sin TP)
    max_velas: usize,  // 0 si no aplica (FIXED/CUSTOM)
    usar_custom: bool, // true si la estrategia decide el cierre
}

struct CierreTrade {
    idx_salida: usize,
    precio_salida: f64,
    motivo: String,
}

/// Ejecuta la simulación completa vela a vela.
///
/// Recibe las velas OHLCV, las señales generadas por la estrategia y la config.
/// Devuelve un SimResult con todos los trades y métricas.
pub fn simular(
    velas: &[Vela],
    señales: &[i8],
    salidas_custom: &[i8],
    config: &SimConfig,
) -> SimResult {
    let n = velas.len();
    assert_eq!(
        n,
        señales.len(),
        "El número de velas y señales debe coincidir"
    );
    assert_eq!(
        n,
        salidas_custom.len(),
        "El número de velas y salidas custom debe coincidir"
    );

    let mut saldo = config.saldo_inicial;
    let mut trades: Vec<TradeResult> = Vec::new();
    let mut equity_curve: Vec<f64> = vec![saldo];
    let mut trade_abierto: Option<TradeAbierto> = None;
    let mut parado_por_saldo = false;
    let mut señal_pendiente: Option<(usize, Direccion)> = None;

    for i in 0..n {
        // --- 1. ¿Hay que ABRIR un trade en esta vela? (señal de la vela anterior) ---
        if let Some((idx_señal, direccion)) = señal_pendiente.take() {
            if trade_abierto.is_none() {
                // Verificar saldo suficiente
                if saldo < config.saldo_minimo || saldo < config.saldo_por_trade {
                    parado_por_saldo = true;
                    break;
                }

                let precio_entrada = velas[i].open;
                let colateral = config.saldo_por_trade;
                let tamaño = capital::calcular_tamaño_posicion(
                    colateral,
                    config.apalancamiento,
                    precio_entrada,
                );
                let comision = capital::calcular_comision(
                    colateral,
                    config.apalancamiento,
                    config.comision_pct,
                    config.comision_lados,
                );

                // Calcular niveles de SL y TP
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
                    idx_señal,
                    idx_entrada: i,
                    direccion,
                    precio_entrada,
                    colateral,
                    tamaño_posicion: tamaño,
                    comision_total: comision,
                    precio_sl,
                    precio_tp,
                    max_velas,
                    usar_custom,
                });
            }
        }

        // --- 2. ¿Hay que CERRAR el trade abierto? ---
        if let Some(ref trade) = trade_abierto {
            let vela = &velas[i];
            let velas_transcurridas = i - trade.idx_entrada + 1;

            if let Some((precio_salida, motivo)) =
                evaluar_salida(trade, vela, velas_transcurridas, salidas_custom[i])
            {
                saldo = cerrar_trade(
                    trade,
                    velas,
                    CierreTrade {
                        idx_salida: i,
                        precio_salida,
                        motivo,
                    },
                    saldo,
                    &mut trades,
                    &mut equity_curve,
                );
                trade_abierto = None;

                // Verificar saldo después del cierre
                if saldo < config.saldo_minimo {
                    parado_por_saldo = true;
                    break;
                }
            }
        }

        // --- 3. ¿Hay señal nueva en esta vela? (se ejecutará en la siguiente) ---
        if trade_abierto.is_none() && señales[i] != 0 {
            if let Ok(direccion) = Direccion::from_signal(señales[i]) {
                señal_pendiente = Some((i, direccion));
            }
        }
    }

    // Si el histórico termina con una posición abierta, se liquida al close
    // de la última vela para que el resultado refleje todo el riesgo tomado.
    if let Some(ref trade) = trade_abierto {
        let idx_salida = n - 1;
        let velas_transcurridas = idx_salida - trade.idx_entrada + 1;
        if velas_transcurridas > 0 {
            saldo = cerrar_trade(
                trade,
                velas,
                CierreTrade {
                    idx_salida,
                    precio_salida: velas[idx_salida].close,
                    motivo: "END".to_string(),
                },
                saldo,
                &mut trades,
                &mut equity_curve,
            );
        }
    }

    // --- Calcular métricas resumen ---
    let total = trades.len();
    let ganadores = trades.iter().filter(|t| t.pnl > 0.0).count();
    let perdedores = total - ganadores;
    let win_rate = if total > 0 {
        ganadores as f64 / total as f64
    } else {
        0.0
    };
    let pnl_total: f64 = trades.iter().map(|t| t.pnl).sum();
    let pnl_promedio = if total > 0 {
        pnl_total / total as f64
    } else {
        0.0
    };
    let roi_total = pnl_total / config.saldo_inicial;
    let max_dd = calcular_max_drawdown(&equity_curve);

    SimResult {
        trades,
        saldo_final: saldo,
        saldo_inicial: config.saldo_inicial,
        total_trades: total,
        trades_ganadores: ganadores,
        trades_perdedores: perdedores,
        win_rate,
        roi_total,
        pnl_total,
        pnl_promedio,
        max_drawdown: max_dd,
        equity_curve,
        parado_por_saldo,
    }
}

fn cerrar_trade(
    trade: &TradeAbierto,
    velas: &[Vela],
    cierre: CierreTrade,
    saldo_previo: f64,
    trades: &mut Vec<TradeResult>,
    equity_curve: &mut Vec<f64>,
) -> f64 {
    let pnl_bruto = capital::calcular_pnl_bruto(
        trade.direccion,
        trade.tamaño_posicion,
        trade.precio_entrada,
        cierre.precio_salida,
    );
    let pnl_neto = pnl_bruto - trade.comision_total;
    let saldo_post = saldo_previo + pnl_neto;
    let roi = pnl_neto / trade.colateral;
    let direccion = match trade.direccion {
        Direccion::Long => 1i8,
        Direccion::Short => -1i8,
    };

    trades.push(TradeResult {
        idx_señal: trade.idx_señal,
        idx_entrada: trade.idx_entrada,
        idx_salida: cierre.idx_salida,
        ts_señal: velas[trade.idx_señal].timestamp,
        ts_entrada: velas[trade.idx_entrada].timestamp,
        ts_salida: velas[cierre.idx_salida].timestamp,
        direccion,
        precio_entrada: trade.precio_entrada,
        precio_salida: cierre.precio_salida,
        colateral: trade.colateral,
        tamaño_posicion: trade.tamaño_posicion,
        comision_total: trade.comision_total,
        pnl: pnl_neto,
        roi,
        saldo_post,
        motivo_salida: cierre.motivo,
        duracion_velas: cierre.idx_salida - trade.idx_entrada + 1,
    });
    equity_curve.push(saldo_post);

    saldo_post
}

/// Evalúa si el trade debe cerrarse en esta vela.
/// Devuelve Some((precio_salida, motivo)) si debe cerrarse, None si sigue abierto.
///
/// Orden de prioridad dentro de la misma vela:
///   1. Stop Loss (usa precio worst-case: low para LONG, high para SHORT)
///   2. Salida CUSTOM de la estrategia -> cierra al close de la vela
///   3. Take Profit (usa precio best-case: high para LONG, low para SHORT)
///   4. Máximo de velas (BARS) → cierra al close de la vela
fn evaluar_salida(
    trade: &TradeAbierto,
    vela: &Vela,
    velas_transcurridas: usize,
    salida_custom: i8,
) -> Option<(f64, String)> {
    // --- Stop Loss ---
    if trade.precio_sl > 0.0 {
        match trade.direccion {
            Direccion::Long if vela.low <= trade.precio_sl => {
                let precio = if vela.open < trade.precio_sl {
                    vela.open
                } else {
                    trade.precio_sl
                };
                return Some((precio, "SL".to_string()));
            }
            Direccion::Short if vela.high >= trade.precio_sl => {
                let precio = if vela.open > trade.precio_sl {
                    vela.open
                } else {
                    trade.precio_sl
                };
                return Some((precio, "SL".to_string()));
            }
            _ => {}
        }
    }

    // --- Salida CUSTOM ---
    if trade.usar_custom {
        match (trade.direccion, salida_custom) {
            (Direccion::Long, 1) | (Direccion::Short, -1) => {
                return Some((vela.close, "CUSTOM".to_string()));
            }
            _ => {}
        }
    }

    // --- Take Profit ---
    if trade.precio_tp > 0.0 {
        match trade.direccion {
            Direccion::Long if vela.high >= trade.precio_tp => {
                let precio = if vela.open > trade.precio_tp {
                    vela.open
                } else {
                    trade.precio_tp
                };
                return Some((precio, "TP".to_string()));
            }
            Direccion::Short if vela.low <= trade.precio_tp => {
                let precio = if vela.open < trade.precio_tp {
                    vela.open
                } else {
                    trade.precio_tp
                };
                return Some((precio, "TP".to_string()));
            }
            _ => {}
        }
    }

    // --- Máximo de velas (BARS) ---
    if trade.max_velas > 0 && velas_transcurridas >= trade.max_velas {
        return Some((vela.close, "BARS".to_string()));
    }

    None
}

/// Calcula el máximo drawdown a partir de la curva de equity.
fn calcular_max_drawdown(equity: &[f64]) -> f64 {
    let mut max_equity = equity[0];
    let mut max_dd = 0.0_f64;

    for &valor in equity.iter() {
        if valor > max_equity {
            max_equity = valor;
        }
        let dd = (max_equity - valor) / max_equity;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_fixed() -> SimConfig {
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

    fn config_custom() -> SimConfig {
        let mut cfg = config_fixed();
        cfg.exit_type = ExitType::Custom;
        cfg.exit_tp_pct = 0.0;
        cfg.exit_velas = 0;
        cfg
    }

    fn vela(ts: i64, o: f64, h: f64, l: f64, c: f64) -> Vela {
        Vela {
            timestamp: ts,
            open: o,
            high: h,
            low: l,
            close: c,
            volume: 100.0,
        }
    }

    fn sin_salidas(n: usize) -> Vec<i8> {
        vec![0; n]
    }

    #[test]
    fn test_entrada_en_vela_siguiente() {
        // Señal en vela 0, entrada debe ser en vela 1 al open
        let velas = vec![
            vela(1, 100.0, 105.0, 95.0, 102.0),  // señal aquí
            vela(2, 103.0, 110.0, 100.0, 108.0), // entrada aquí
            vela(3, 108.0, 200.0, 100.0, 150.0), // TP aquí
        ];
        let señales = vec![1, 0, 0];
        let salidas = sin_salidas(velas.len());
        let cfg = config_fixed();
        let result = simular(&velas, &señales, &salidas, &cfg);

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.trades[0].idx_señal, 0);
        assert_eq!(result.trades[0].idx_entrada, 1);
        assert!((result.trades[0].precio_entrada - 103.0).abs() < 1e-10);
    }

    #[test]
    fn test_señales_ignoradas_con_trade_abierto() {
        // Señal en vela 0, otra señal en vela 2 (debe ignorarse)
        let velas = vec![
            vela(1, 100.0, 105.0, 95.0, 102.0),
            vela(2, 103.0, 105.0, 101.0, 104.0),
            vela(3, 104.0, 106.0, 103.0, 105.0), // señal aquí, ignorada
            vela(4, 105.0, 200.0, 100.0, 150.0), // TP aquí
        ];
        let señales = vec![1, 0, -1, 0];
        let salidas = sin_salidas(velas.len());
        let cfg = config_fixed();
        let result = simular(&velas, &señales, &salidas, &cfg);

        // Solo un trade, la señal SHORT fue ignorada
        assert_eq!(result.total_trades, 1);
        assert_eq!(result.trades[0].direccion, 1); // LONG
    }

    #[test]
    fn test_max_drawdown_simple() {
        let equity = vec![10_000.0, 10_500.0, 9_500.0, 10_200.0];
        let dd = calcular_max_drawdown(&equity);
        // Peak 10,500, valley 9,500 → DD = 1000/10500 ≈ 0.09524
        assert!((dd - 1000.0 / 10500.0).abs() < 1e-10);
    }

    #[test]
    fn test_cierra_trade_abierto_al_final() {
        let velas = vec![
            vela(1, 100.0, 101.0, 99.0, 100.0),
            vela(2, 100.0, 101.0, 99.0, 100.5),
            vela(3, 101.0, 102.0, 100.0, 101.5),
        ];
        let señales = vec![1, 0, 0];
        let salidas = sin_salidas(velas.len());
        let result = simular(&velas, &señales, &salidas, &config_fixed());

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.trades[0].idx_salida, 2);
        assert_eq!(result.trades[0].motivo_salida, "END");
        assert!((result.trades[0].precio_salida - 101.5).abs() < 1e-10);
    }

    #[test]
    fn test_gap_long_ejecuta_sl_en_open_si_salta_el_stop() {
        let velas = vec![
            vela(1, 100.0, 101.0, 99.0, 100.0),
            vela(2, 100.0, 101.0, 99.0, 100.5),
            vela(3, 95.0, 97.0, 94.0, 96.0),
        ];
        let señales = vec![1, 0, 0];
        let salidas = sin_salidas(velas.len());
        let result = simular(&velas, &señales, &salidas, &config_fixed());

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.trades[0].motivo_salida, "SL");
        assert!((result.trades[0].precio_salida - 95.0).abs() < 1e-10);
    }

    #[test]
    fn test_gap_short_ejecuta_sl_en_open_si_salta_el_stop() {
        let velas = vec![
            vela(1, 100.0, 101.0, 99.0, 100.0),
            vela(2, 100.0, 101.0, 99.0, 100.5),
            vela(3, 105.0, 106.0, 104.0, 105.5),
        ];
        let señales = vec![-1, 0, 0];
        let salidas = sin_salidas(velas.len());
        let result = simular(&velas, &señales, &salidas, &config_fixed());

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.trades[0].motivo_salida, "SL");
        assert!((result.trades[0].precio_salida - 105.0).abs() < 1e-10);
    }

    #[test]
    fn test_bars_cierra_por_numero_de_velas() {
        let velas = vec![
            vela(1, 100.0, 101.0, 99.0, 100.0),
            vela(2, 100.0, 101.0, 99.0, 100.5),
            vela(3, 101.0, 102.0, 100.0, 101.5),
            vela(4, 102.0, 103.0, 101.0, 102.5),
        ];
        let señales = vec![1, 0, 0, 0];
        let salidas = sin_salidas(velas.len());
        let mut cfg = config_fixed();
        cfg.exit_type = ExitType::Bars;
        cfg.exit_tp_pct = 0.0;
        cfg.exit_velas = 2;

        let result = simular(&velas, &señales, &salidas, &cfg);

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.trades[0].idx_salida, 2);
        assert_eq!(result.trades[0].motivo_salida, "BARS");
    }

    #[test]
    fn test_custom_cierra_long_al_close_de_la_vela() {
        let velas = vec![
            vela(1, 100.0, 101.0, 99.0, 100.0),
            vela(2, 100.0, 101.0, 99.0, 100.5),
            vela(3, 101.0, 102.0, 100.0, 101.5),
            vela(4, 102.0, 103.0, 101.0, 102.5),
        ];
        let señales = vec![1, 0, 0, 0];
        let salidas = vec![0, 0, 1, 0];

        let result = simular(&velas, &señales, &salidas, &config_custom());

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.trades[0].idx_salida, 2);
        assert_eq!(result.trades[0].motivo_salida, "CUSTOM");
        assert!((result.trades[0].precio_salida - 101.5).abs() < 1e-10);
    }

    #[test]
    fn test_custom_mantiene_prioridad_de_sl() {
        let velas = vec![
            vela(1, 100.0, 101.0, 99.0, 100.0),
            vela(2, 100.0, 101.0, 99.0, 100.5),
            vela(3, 100.0, 101.0, 95.0, 99.0),
        ];
        let señales = vec![1, 0, 0];
        let salidas = vec![0, 0, 1];

        let result = simular(&velas, &señales, &salidas, &config_custom());

        assert_eq!(result.total_trades, 1);
        assert_eq!(result.trades[0].motivo_salida, "SL");
    }
}

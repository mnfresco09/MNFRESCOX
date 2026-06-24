# MOTOR_RIESGO — micro-motor de riesgo (Rust)

Crate **independiente** del PANEL PORTFOLIO para los cálculos iterativos pesados.
No comparte nada con `PANEL BACKTESTING/MOTOR/` (aislamiento estricto).

## Qué expone

- `fhs(residuos, sigma_next, niveles) -> (var95, var99, cvar95, cvar99)`
  Filtered Historical Simulation: VaR/CVaR a T+1.
- `montecarlo(retornos, horizonte, n_trayectorias, percentiles, seed)`
  `-> (sendas[n_perc, horizonte], prob_perdida, cdar, ret_mediano, perdida_p5)`
  Bootstrapping con reemplazo, paralelizado con `rayon`. Acepta `seed`
  (reproducible). **Nunca** devuelve la matriz completa de trayectorias: solo
  percentiles para el fan chart y agregados.

## Compilación

Se compila **bajo demanda** desde Python (`RIESGO/motor_bindings.py` llama a
`cargo build --release`). También manual:

```bash
cd "PANEL PORTFOLIO/MOTOR_RIESGO"
cargo build --release
```

La librería resultante (`target/release/libmotor_riesgo.{dylib,so,dll}`) la carga
el binding automáticamente.

## Fallback

Si `cargo` no está disponible o la compilación falla, el binding cae a una
implementación NumPy equivalente y marca la fuente como `python_fallback`. El
panel sigue funcionando; solo cambia el rendimiento.

# Pre-registro de hipótesis — <NOMBRE_ESTRATEGIA>

> **Se rellena ANTES de programar la estrategia y ANTES de mirar ningún
> resultado.** Un edge sin razón económica previa es minería de datos: si solo
> lo encontraste buscando, probablemente no existe. Este documento es la Puerta 0
> del protocolo y el ancla contra el autoengaño — fija los criterios *a priori*.

- **Estrategia:** <nombre> (ID <n>)
- **Autor:** <tú>
- **Fecha de pre-registro:** <AAAA-MM-DD>  *(antes de programar nada)*
- **Activos objetivo:** <BTC / GOLD / ...>
- **Timeframes objetivo:** <1h / 4h / ...>

---

## 1. La ineficiencia que se explota

¿Qué comportamiento concreto del mercado genera el edge? Sé específico: no
"el RSI funciona", sino *qué* hace el flujo de órdenes / los participantes /
la microestructura para crear la oportunidad.

> _Escribe aquí._

## 2. Por qué debería persistir

Si el edge es real, ¿por qué no lo ha arbitrado ya todo el mundo? (límites al
arbitraje, coste de capital, sesgo de comportamiento, fricción estructural...).
Un edge sin razón para persistir es un edge que ya murió o nunca existió.

> _Escribe aquí._

## 3. En qué regímenes esperas que funcione (y en cuáles NO)

¿Alcista, bajista, lateral? ¿Alta o baja volatilidad? Comprometerte aquí evita
racionalizar después por qué "solo funciona en su régimen favorito".

> _Escribe aquí._

## 4. Qué resultado te haría DESCARTARLA

Define el fallo **antes** de ver los números. Ej.: "si el Sharpe OOS cae por
debajo del 50% del IS, muere"; "si colapsa al duplicar costes, era microestructura".

> _Escribe aquí._

## 5. Umbrales de decisión fijados A PRIORI

Cópialos del protocolo (Parte IV) o ajústalos **en abstracto**, nunca tras ver el
resultado de esta estrategia concreta.

| Métrica | 🟢 Verde | 🟡 Ámbar | 🔴 Rojo |
|---|---|---|---|
| DSR (deflactado contra N real) | > 0.95 | 0.90 – 0.95 | < 0.90 |
| PBO | < 0.20 | 0.20 – 0.50 | > 0.50 |
| Sharpe OOS / Sharpe IS | ≥ 0.70 | 0.50 – 0.70 | < 0.50 |
| Distribución Sharpe OOS (CPCV) | p25 > 0 | mediana > 0, p25 < 0 | mediana ≤ 0 |
| Supervivencia a 2× costes | claramente positivo | degrada | colapsa |
| Nº de trades | ≥ 100 | 30 – 100 | < 30 |
| WFA efficiency | > 0.6 | 0.5 – 0.6 | < 0.5 |
| Bootstrap p5 equity final | > capital inicial | ≈ capital inicial | < capital inicial |
| Holdout bloqueado | coherente con OOS | degrada pero positivo | colapsa / negativo |

## 6. Registro de iteraciones (testing múltiple)

Cada vez que ajustas un parámetro y vuelves a pasar el protocolo, **suma una
prueba a tu N**. Anótalo aquí; el DSR debe deflactarse en consecuencia.

| Fecha | Qué cambié | Por qué | Puerta donde falló |
|---|---|---|---|
| | | | |

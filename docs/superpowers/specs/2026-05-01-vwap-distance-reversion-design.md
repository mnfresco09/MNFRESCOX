# VWAP Distance Reversion - Diseno

## Objetivo

Crear una estrategia nueva de reversion a la media basada solo en la distancia
normalizada del precio respecto a VWAP. La estrategia no modifica `vwapcvd.py`
ni reutiliza sus helpers por importacion: tendra su propio calculo completo,
limitado a VWAP y `VWAP DIST Z`.

## Alcance

- Nuevo archivo en `PANEL BACKTESTING/ESTRATEGIAS/`.
- Nueva clase de estrategia autodetectable por el registro existente.
- Calculo propio de:
  - VWAP EWM.
  - Distancia raw: `(close - vwap) / vwap`.
  - Distancia suavizada.
  - Normalizacion tipo Z con clipping.
- Generacion de entradas long y short por activacion y reversion del umbral.
- Indicadores para reporte HTML.
- Pruebas automatizadas especificas de la estrategia.

Queda fuera de alcance:

- Cambiar `vwapcvd.py`.
- Usar CVD o columnas `taker_buy_volume` / `taker_sell_volume`.
- Crear salidas custom. Las salidas seguiran siendo las configuradas por
  `EXIT_TYPE`.
- Cambiar el runner, el motor o el registro de estrategias.

## Parametros

La estrategia tendra estos parametros:

- `halflife_bars`: vida media del VWAP EWM y suavizado principal.
- `normalization_multiplier`: multiplicador de la vida media para la ventana
  de normalizacion.
- `vwap_clip_sigmas`: numero de sigmas usado para limitar valores extremos
  antes de calcular la desviacion final.
- `umbral_distance_z`: umbral simetrico de activacion/reversion.

Valores por defecto propuestos:

- `halflife_bars = 15`
- `normalization_multiplier = 3.0`
- `vwap_clip_sigmas = 2.5`
- `umbral_distance_z = 0.5`

## Reglas De Senal

La estrategia opera sobre `distance_z` y mantiene un estado interno durante el
recorrido de la serie.

Short:

1. Se arma seguimiento short cuando `distance_z` cruza el umbral superior desde
   `<= +umbral` hasta `> +umbral`.
2. Una vez armado, se genera entrada short cuando `distance_z` cruza de vuelta
   desde `>= +umbral` hasta `< +umbral`.
3. La senal devuelta es `-1`.

Long:

1. Se arma seguimiento long cuando `distance_z` cruza el umbral inferior desde
   `>= -umbral` hasta `< -umbral`.
2. Una vez armado, se genera entrada long cuando `distance_z` cruza de vuelta
   desde `<= -umbral` hasta `> -umbral`.
3. La senal devuelta es `1`.

Despues de emitir una entrada, el seguimiento de ese lado queda desarmado. Para
una nueva entrada hace falta una nueva activacion por cruce externo del umbral.

## Warmup Y Robustez

Se bloquearan senales durante el warmup:

`max(halflife_bars * 9, round(halflife_bars * normalization_multiplier * 3))`

Las senales solo se evaluaran cuando `close`, `vwap`, `distance_z_prev` y
`distance_z` sean finitos. Volumen no positivo se tratara de forma defensiva:
si no hay volumen real util en toda la serie, el VWAP usara peso unitario para
evitar divisiones invalidas.

## Reporte

`indicadores_para_grafica()` devolvera:

- Overlay de VWAP.
- Pane `VWAP DIST Z` con niveles:
  - `+umbral`
  - `0`
  - `-umbral`

## Pruebas

Se anadira un test nuevo para validar:

- La estrategia no requiere columnas de CVD.
- Un cruce superior arma seguimiento y la vuelta bajo el umbral genera short.
- Un cruce inferior arma seguimiento y la vuelta sobre el umbral genera long.
- No hay entrada sin activacion previa.
- Las primeras velas bloqueadas por warmup no generan senales.
- Los indicadores del reporte tienen el formato esperado.

## Integracion

El registro de estrategias ya escanea `ESTRATEGIAS/*.py`, por lo que no hace
falta registrar la clase a mano. El nuevo `ID` debe ser unico respecto a las
estrategias existentes.

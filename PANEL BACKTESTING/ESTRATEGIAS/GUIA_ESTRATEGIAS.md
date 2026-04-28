# Guia seria para crear estrategias

Esta guia define el contrato real de una estrategia dentro de `PANEL BACKTESTING`.
Una estrategia que no respete este contrato no debe ejecutarse en optimizacion.

## Contrato obligatorio

Cada archivo de `ESTRATEGIAS/` debe contener una clase que herede de
`BaseEstrategia` y defina:

- `ID`: entero unico. No puede repetirse.
- `NOMBRE`: nombre legible para reportes y carpetas.
- `COLUMNAS_REQUERIDAS`: columnas extra que necesita la estrategia, si aplica.
- `parametros_por_defecto()`: parametros razonables para pruebas fijas.
- `espacio_busqueda(trial)`: rangos que Optuna puede explorar.
- `generar_señales(df, params)`: serie `Int8` con `0`, `1`, `-1`.

Si se quiere usar `EXIT_TYPE = "CUSTOM"` tambien debe definir:

- `generar_salidas(df, params)`: serie `Int8` con `0`, `1`, `-1`.

## Semantica exacta

`generar_señales()`:

- `0`: no abrir nada.
- `1`: abrir LONG en el `open` de la vela siguiente.
- `-1`: abrir SHORT en el `open` de la vela siguiente.

`generar_salidas()` para `CUSTOM`:

- `0`: no cerrar nada.
- `1`: cerrar un LONG abierto.
- `-1`: cerrar un SHORT abierto.

La salida CUSTOM se confirma con la vela donde aparece la condicion, pero no
cierra en ese mismo `close`: el motor la ejecuta en el `open` de la siguiente
vela disponible. El Stop Loss de seguridad se evalua antes que la salida CUSTOM.
Si una vela toca el SL y tambien aparece una salida custom, el trade cierra por
`SL`. Si la salida custom queda pendiente y el siguiente `open` ya cruza el SL,
el trade tambien cierra por `SL`.

Las salidas `FIXED` y `BARS` se ejecutan en el timeframe mas bajo disponible en
`HISTORICO`, aunque la estrategia opere en un timeframe superior. La señal sigue
naciendo en la vela de estrategia y la entrada se mantiene en el `open` de la
siguiente vela de estrategia. `CUSTOM` calcula la condicion en el timeframe de
la estrategia, proyecta esa condicion al timeframe de ejecucion y cierra en el
`open` siguiente para evitar look-ahead.

## Regla anti-lookahead

La estrategia no puede usar informacion futura. En la practica:

- No usar `shift(-1)` para decidir señales o salidas.
- No calcular indicadores con velas posteriores a la vela actual.
- No entrar en el `close` de la misma vela donde se genera la señal.
- No cerrar una salida custom en el mismo `close` donde se confirma.
- No rellenar nulos de indicadores con datos futuros.
- No modificar el DataFrame recibido.

El motor protege la entrada: una señal en vela `N` siempre entra en el `open` de
`N+1`. La estrategia solo decide la condicion; el motor decide la ejecucion.

## Rango de parametros

Los rangos de `espacio_busqueda()` deben ser defendibles:

- Evitar rangos enormes sin sentido operativo.
- Mantener separadas las variables de entrada y salida.
- Usar enteros para periodos de indicadores.
- Usar floats solo cuando el parametro realmente sea continuo.
- Si un parametro tiene dependencias, validar la relacion dentro de la estrategia.

Ejemplo: para medias moviles, la media rapida no deberia ser mayor que la lenta
si la logica asume tendencia por cruce.

## Plantilla

```python
import polars as pl

from NUCLEO.base_estrategia import BaseEstrategia
from NUCLEO.tipos import Señal


class MiEstrategia(BaseEstrategia):
    ID = 100
    NOMBRE = "Mi Estrategia"
    COLUMNAS_REQUERIDAS = set()

    def parametros_por_defecto(self) -> dict:
        return {
            "periodo": 20,
        }

    def espacio_busqueda(self, trial) -> dict:
        return {
            "periodo": trial.suggest_int("periodo", 10, 80),
        }

    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        periodo = int(params["periodo"])
        media = self.ema(df["close"], periodo)
        media_prev = media.shift(1)

        señal_long = ((df["close"].shift(1) <= media_prev) & (df["close"] > media)).fill_null(False)
        señal_short = ((df["close"].shift(1) >= media_prev) & (df["close"] < media)).fill_null(False)

        señales = pl.Series([Señal.NINGUNA] * len(df), dtype=pl.Int8)
        señales = señales.scatter(señal_long.arg_true(), Señal.LONG)
        señales = señales.scatter(señal_short.arg_true(), Señal.SHORT)
        return señales

    def generar_salidas(self, df: pl.DataFrame, params: dict) -> pl.Series:
        periodo = int(params["periodo"])
        media = self.ema(df["close"], periodo)

        salida_long = (df["close"] < media).fill_null(False)
        salida_short = (df["close"] > media).fill_null(False)

        salidas = pl.Series([0] * len(df), dtype=pl.Int8)
        salidas = salidas.scatter(salida_long.arg_true(), 1)
        salidas = salidas.scatter(salida_short.arg_true(), -1)
        return salidas
```

## Checklist antes de optimizar

Antes de lanzar una optimizacion grande:

- `ID` unico y `NOMBRE` estable.
- `generar_señales()` devuelve exactamente `len(df)` filas.
- `generar_salidas()` devuelve exactamente `len(df)` filas si se usa CUSTOM.
- Las series no contienen nulos.
- Las series solo contienen `-1`, `0`, `1`.
- Los primeros valores de indicadores con warmup no producen señales falsas.
- Las señales no dependen de futuro.
- Los rangos de Optuna son realistas.
- Una prueba corta genera `trades.csv`, `equity.csv`, `resumen.json`,
  `auditoria.json`, Excel y HTML sin perder filas.

## Como se audita

El sistema verifica por cada trial:

- Datos ordenados y sin duplicados.
- OHLC coherente.
- Resampleo sin crear fechas fuera del historico.
- Señales alineadas con las velas.
- Salidas custom alineadas con las velas cuando aplica.
- Entradas siempre en `open` de `N+1`.
- Cierres por `SL`, `TP`, `BARS`, `CUSTOM` o `END` en precios coherentes.
- PnL agregado desde trades, `saldo_final` y `equity_curve` consistentes.

Si cualquiera de estos puntos falla, el run debe parar con error explicito.

"""Perturbaciones multivariantes en memoria, sin modificar HISTORICO.

La tabla de condicionales se calibra automaticamente desde el Parquet cargado:
volumen relativo y order flow observados por cubo de retorno. El usuario solo
decide la granularidad de los cubos y el percentil que define el rango valido.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any

import numpy as np
import polars as pl

try:
    from numba import njit
    _NUMBA_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - validado en runtime si se activan perturbaciones
    njit = None
    _NUMBA_IMPORT_ERROR = exc


_COLUMNAS_PRECIO = ("open", "high", "low", "close")

_BANDA_MAX_PRECIO = 0.15
_FUERZA_AMORTIGUACION = 0.10
_ESCALA_VOLATILIDAD = 0.50
_VENTANA_VOLATILIDAD = 20
_SIGMA_RANGO_VELA = 0.05
_RUIDO_POSICION_OHLC = 0.08
_INERCIA_ORDER_FLOW = 0.30
_VENTANA_MEDIA_VOLUMEN = 20
_MIN_MUESTRAS_CUBO = 30


@dataclass(frozen=True)
class TablaCondicionales:
    bordes: np.ndarray
    vol_min: np.ndarray
    vol_max: np.ndarray
    sell_min: np.ndarray
    sell_max: np.ndarray


@dataclass(frozen=True)
class BasePerturbaciones:
    filas: int
    ts_inicio_us: int | None
    ts_fin_us: int | None
    open_orig: np.ndarray
    high_orig: np.ndarray
    low_orig: np.ndarray
    close_orig: np.ndarray
    volume_orig: np.ndarray
    media_volumen_orig: np.ndarray
    midpoint_orig: np.ndarray
    rango_rel_orig: np.ndarray
    frac_open_orig: np.ndarray
    frac_close_orig: np.ndarray
    volatilidad_local: np.ndarray
    prop_taker_sell_inicial: float


@dataclass(frozen=True)
class ConfiguracionPerturbaciones:
    activa: bool
    seed_global: int | None
    granularidad_cubos: float
    percentil_tabla: float
    tabla: TablaCondicionales | None = None
    base: BasePerturbaciones | None = None

    @classmethod
    def desde_config(cls, cfg: Any) -> "ConfiguracionPerturbaciones":
        usar_seed = bool(getattr(cfg, "USAR_SEED", True))
        return cls(
            activa=bool(getattr(cfg, "PERTURBACIONES_ACTIVAS", False)),
            seed_global=getattr(cfg, "PERTURBACIONES_SEED", None) if usar_seed else None,
            granularidad_cubos=float(getattr(cfg, "GRANULARIDAD_CUBOS", 0.005)),
            percentil_tabla=float(getattr(cfg, "PERCENTIL_TABLA", 0.10)),
        )

    def con_tabla_desde(self, df: pl.DataFrame) -> "ConfiguracionPerturbaciones":
        if not self.activa:
            return self
        base = _precalcular_base(df)
        tabla = construir_tabla_condicionales(
            df,
            granularidad=self.granularidad_cubos,
            percentil=self.percentil_tabla,
        )
        return replace(self, tabla=tabla, base=base)


def seed_para_trial(
    config: ConfiguracionPerturbaciones,
    *,
    trial_numero: int,
    activo: str,
    timeframe: str,
    estrategia_id: int,
    salida_tipo: str,
) -> int | None:
    if not config.activa:
        return None

    if config.seed_global is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])

    payload = (
        f"{int(config.seed_global)}|{int(trial_numero)}|{activo}|{timeframe}|"
        f"{int(estrategia_id)}|{salida_tipo}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def aplicar_perturbaciones(
    df: pl.DataFrame,
    config: ConfiguracionPerturbaciones,
    *,
    seed: int | None,
) -> pl.DataFrame:
    if not config.activa:
        return df
    if seed is None:
        raise ValueError("[PERT] Las perturbaciones activas requieren una seed de trial.")
    if df.height < 2:
        return df

    _validar_columnas_minimas(df)

    kernel = _requerir_kernel_numba()
    rng = np.random.default_rng(int(seed))
    base = _base_para_dataframe(df, config)
    n = base.filas
    tabla = config.tabla
    if tabla is None:
        tabla = construir_tabla_condicionales(
            df,
            granularidad=config.granularidad_cubos,
            percentil=config.percentil_tabla,
        )

    taker_buy, has_taker_buy = _array_float_opcional(df, "taker_buy_volume", n)
    taker_sell, has_taker_sell = _array_float_opcional(df, "taker_sell_volume", n)
    vol_delta, has_vol_delta = _array_float_opcional(df, "vol_delta", n)
    num_trades, has_num_trades = _array_int_opcional(df, "num_trades", n)

    open_p, high_p, low_p, close_p, volume_p = kernel(
        base.open_orig,
        base.high_orig,
        base.low_orig,
        base.close_orig,
        base.volume_orig,
        base.media_volumen_orig,
        base.midpoint_orig,
        base.rango_rel_orig,
        base.frac_open_orig,
        base.frac_close_orig,
        base.volatilidad_local,
        taker_buy,
        taker_sell,
        vol_delta,
        num_trades,
        bool(has_taker_buy),
        bool(has_taker_sell),
        bool(has_vol_delta),
        bool(has_num_trades),
        tabla.bordes,
        tabla.vol_min,
        tabla.vol_max,
        tabla.sell_min,
        tabla.sell_max,
        float(base.prop_taker_sell_inicial),
        float(_BANDA_MAX_PRECIO),
        float(_FUERZA_AMORTIGUACION),
        float(_ESCALA_VOLATILIDAD),
        float(_SIGMA_RANGO_VELA),
        float(_RUIDO_POSICION_OHLC),
        float(_INERCIA_ORDER_FLOW),
        rng.normal(0.0, 1.0, n).astype(np.float64),
        rng.normal(0.0, 1.0, n).astype(np.float64),
        rng.random(n).astype(np.float64),
        rng.random(n).astype(np.float64),
        rng.random(n).astype(np.float64),
        rng.uniform(-1.0, 1.0, n).astype(np.float64),
    )

    updates: dict[str, np.ndarray] = {
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": volume_p,
    }

    if has_taker_buy:
        updates["taker_buy_volume"] = taker_buy
    if has_taker_sell:
        updates["taker_sell_volume"] = taker_sell
    if has_vol_delta:
        updates["vol_delta"] = vol_delta
    if has_num_trades:
        updates["num_trades"] = num_trades

    if "quote_volume" in df.columns:
        precio_medio = (high_p + low_p) * 0.5
        updates["quote_volume"] = volume_p * precio_medio
    if "taker_buy_quote_volume" in df.columns:
        precio_medio = (high_p + low_p) * 0.5
        updates["taker_buy_quote_volume"] = taker_buy * precio_medio

    perturbado = df.with_columns([pl.Series(col, values) for col, values in updates.items()])
    _validar_invariantes(perturbado)
    return perturbado


def _validar_columnas_minimas(df: pl.DataFrame) -> None:
    faltantes = [col for col in (*_COLUMNAS_PRECIO, "volume") if col not in df.columns]
    if faltantes:
        raise ValueError(f"[PERT] Faltan columnas obligatorias para perturbar: {faltantes}")


def _f64(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].cast(pl.Float64).to_numpy()


def _precalcular_base(df: pl.DataFrame) -> BasePerturbaciones:
    _validar_columnas_minimas(df)
    open_orig = _f64(df, "open")
    high_orig = _f64(df, "high")
    low_orig = _f64(df, "low")
    close_orig = _f64(df, "close")
    volume_orig = _f64(df, "volume")
    media_volumen_orig = _media_rodante_pasada(volume_orig, _VENTANA_MEDIA_VOLUMEN)

    rango_orig = np.maximum(high_orig - low_orig, np.maximum(close_orig, 1.0) * 1e-9)
    midpoint_orig = (high_orig + low_orig) * 0.5
    midpoint_orig = np.maximum(midpoint_orig, 1e-9)
    rango_rel_orig = np.maximum(rango_orig / midpoint_orig, 1e-9)
    frac_open_orig = _fraccion_array(open_orig, low_orig, rango_orig)
    frac_close_orig = _fraccion_array(close_orig, low_orig, rango_orig)
    volatilidad_local = _media_rodante_pasada(rango_rel_orig, _VENTANA_VOLATILIDAD)
    ts_inicio_us, ts_fin_us = _limites_timestamp_us(df)

    return BasePerturbaciones(
        filas=df.height,
        ts_inicio_us=ts_inicio_us,
        ts_fin_us=ts_fin_us,
        open_orig=open_orig,
        high_orig=high_orig,
        low_orig=low_orig,
        close_orig=close_orig,
        volume_orig=volume_orig,
        media_volumen_orig=media_volumen_orig,
        midpoint_orig=midpoint_orig,
        rango_rel_orig=rango_rel_orig,
        frac_open_orig=frac_open_orig,
        frac_close_orig=frac_close_orig,
        volatilidad_local=volatilidad_local,
        prop_taker_sell_inicial=_prop_taker_sell_original(df, 0),
    )


def _base_para_dataframe(df: pl.DataFrame, config: ConfiguracionPerturbaciones) -> BasePerturbaciones:
    base = config.base
    if base is None or base.filas != df.height:
        return _precalcular_base(df)

    ts_inicio_us, ts_fin_us = _limites_timestamp_us(df)
    if base.ts_inicio_us != ts_inicio_us or base.ts_fin_us != ts_fin_us:
        return _precalcular_base(df)
    return base


def _limites_timestamp_us(df: pl.DataFrame) -> tuple[int | None, int | None]:
    if "timestamp" not in df.columns or df.height == 0:
        return None, None
    dtype = df.schema.get("timestamp")
    if isinstance(dtype, pl.Datetime):
        limites = df.select(
            pl.col("timestamp").first().dt.epoch("us").alias("inicio"),
            pl.col("timestamp").last().dt.epoch("us").alias("fin"),
        ).row(0)
    else:
        limites = df.select(
            pl.col("timestamp").first().cast(pl.Int64).alias("inicio"),
            pl.col("timestamp").last().cast(pl.Int64).alias("fin"),
        ).row(0)
    return int(limites[0]), int(limites[1])


def _media_rodante_pasada(valores: np.ndarray, ventana: int) -> np.ndarray:
    ventana = max(int(ventana), 1)
    n = valores.shape[0]
    cumsum = np.concatenate(([0.0], np.cumsum(valores, dtype=np.float64)))
    idx = np.arange(n)
    inicio = np.maximum(0, idx - ventana)
    cuenta = idx - inicio
    out = np.empty(n, dtype=np.float64)
    out[0] = valores[0]
    validos = cuenta > 0
    out[validos] = (cumsum[idx[validos]] - cumsum[inicio[validos]]) / cuenta[validos]
    out[~validos] = valores[0]
    return np.maximum(out, 1e-12)


def _fraccion_array(valor: np.ndarray, low: np.ndarray, rango: np.ndarray) -> np.ndarray:
    out = np.divide(valor - low, rango, out=np.full_like(valor, 0.5), where=rango > 0)
    return np.clip(out, 0.01, 0.99)


def construir_tabla_condicionales(
    df: pl.DataFrame,
    *,
    granularidad: float,
    percentil: float,
) -> TablaCondicionales:
    _validar_columnas_tabla(df)
    if not (0.0 < percentil < 0.5):
        raise ValueError("[PERT] PERCENTIL_TABLA debe estar entre 0 y 0.5.")

    close = _f64(df, "close")
    volume = _f64(df, "volume")
    taker_sell = _f64(df, "taker_sell_volume")

    media_vol = _media_rodante_pasada(volume, _VENTANA_MEDIA_VOLUMEN)
    retorno = np.empty_like(close)
    retorno[0] = np.nan
    retorno[1:] = np.divide(
        close[1:] - close[:-1],
        close[:-1],
        out=np.full(close.shape[0] - 1, np.nan, dtype=np.float64),
        where=close[:-1] > 0,
    )
    volumen_relativo = np.divide(
        volume,
        media_vol,
        out=np.full_like(volume, np.nan),
        where=media_vol > 0,
    )
    prop_sell = np.divide(
        taker_sell,
        volume,
        out=np.full_like(volume, np.nan),
        where=volume > 0,
    )

    bordes = _bordes_cubos(granularidad)
    cubos = np.searchsorted(bordes[1:-1], retorno, side="right")
    validos = (
        np.isfinite(retorno)
        & np.isfinite(volumen_relativo)
        & np.isfinite(prop_sell)
        & (volume > 0)
    )
    if not validos.any():
        raise ValueError("[PERT] No hay muestras validas para construir la tabla de condicionales.")

    n_cubos = bordes.shape[0] - 1
    vol_min, vol_max = _percentiles_por_cubo(
        volumen_relativo[validos],
        cubos[validos],
        n_cubos,
        percentil,
        limite_min=0.0,
        limite_max=None,
    )
    sell_min, sell_max = _percentiles_por_cubo(
        prop_sell[validos],
        cubos[validos],
        n_cubos,
        percentil,
        limite_min=0.0,
        limite_max=1.0,
    )
    return TablaCondicionales(
        bordes=bordes,
        vol_min=vol_min,
        vol_max=vol_max,
        sell_min=sell_min,
        sell_max=sell_max,
    )


def _validar_columnas_tabla(df: pl.DataFrame) -> None:
    faltantes = [
        col
        for col in ("close", "volume", "taker_sell_volume")
        if col not in df.columns
    ]
    if faltantes:
        raise ValueError(f"[PERT] Faltan columnas para construir la tabla automatica: {faltantes}")


def _bordes_cubos(granularidad: float) -> np.ndarray:
    if granularidad <= 0:
        raise ValueError("[PERT] GRANULARIDAD_CUBOS debe ser mayor que 0.")
    g = float(granularidad)
    return np.array(
        [-np.inf, -4*g, -2*g, -g, -0.4*g, -0.1*g, 0.1*g, 0.4*g, g, 2*g, 4*g, np.inf],
        dtype=np.float64,
    )


def _percentiles_por_cubo(
    valores: np.ndarray,
    cubos: np.ndarray,
    n_cubos: int,
    percentil: float,
    *,
    limite_min: float | None,
    limite_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    lo_global = float(np.quantile(valores, percentil))
    hi_global = float(np.quantile(valores, 1.0 - percentil))
    lo = np.full(n_cubos, np.nan, dtype=np.float64)
    hi = np.full(n_cubos, np.nan, dtype=np.float64)

    for cubo in range(n_cubos):
        en_cubo = valores[cubos == cubo]
        if en_cubo.shape[0] >= _MIN_MUESTRAS_CUBO:
            lo[cubo] = float(np.quantile(en_cubo, percentil))
            hi[cubo] = float(np.quantile(en_cubo, 1.0 - percentil))

    validos = np.where(np.isfinite(lo) & np.isfinite(hi))[0]
    for cubo in range(n_cubos):
        if np.isfinite(lo[cubo]) and np.isfinite(hi[cubo]):
            continue
        if validos.size:
            cercano = int(validos[np.argmin(np.abs(validos - cubo))])
            lo[cubo] = lo[cercano]
            hi[cubo] = hi[cercano]
        else:
            lo[cubo] = lo_global
            hi[cubo] = hi_global

    if limite_min is not None:
        lo = np.maximum(lo, limite_min)
        hi = np.maximum(hi, limite_min)
    if limite_max is not None:
        lo = np.minimum(lo, limite_max)
        hi = np.minimum(hi, limite_max)

    ancho_minimo = np.maximum(np.abs(lo) * 1e-9, 1e-12)
    hi = np.maximum(hi, lo + ancho_minimo)
    if limite_max is not None:
        hi = np.minimum(hi, limite_max)
        lo = np.minimum(lo, hi)
    return lo.astype(np.float64), hi.astype(np.float64)


def _prop_taker_sell_original(df: pl.DataFrame, idx: int) -> float:
    if "taker_sell_volume" not in df.columns or "volume" not in df.columns:
        return 0.5
    volumen = float(df["volume"][idx])
    if volumen <= 0:
        return 0.5
    return min(max(float(df["taker_sell_volume"][idx]) / volumen, 0.0), 1.0)


def _array_float_opcional(df: pl.DataFrame, col: str, n: int) -> tuple[np.ndarray, bool]:
    if col in df.columns:
        return df[col].cast(pl.Float64).to_numpy().copy(), True
    return np.zeros(n, dtype=np.float64), False


def _array_int_opcional(df: pl.DataFrame, col: str, n: int) -> tuple[np.ndarray, bool]:
    if col in df.columns:
        return df[col].cast(pl.Int64).to_numpy().copy(), True
    return np.zeros(n, dtype=np.int64), False


def _requerir_kernel_numba():
    if _kernel_perturbacion is None:
        detalle = f" Detalle: {_NUMBA_IMPORT_ERROR}" if _NUMBA_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "[PERT] Las perturbaciones activas requieren numba. "
            "Instala dependencias con `pip install -r requirements.txt`."
            f"{detalle}"
        )
    return _kernel_perturbacion


def validar_kernel_numba() -> None:
    kernel = _requerir_kernel_numba()
    try:
        _smoke_kernel_numba(kernel)
    except Exception as exc:
        raise RuntimeError(f"[PERT] No se pudo compilar/ejecutar el kernel Numba: {exc}") from exc


def _smoke_kernel_numba(kernel) -> None:
    n = 2
    unos = np.ones(n, dtype=np.float64)
    ceros = np.zeros(n, dtype=np.float64)
    tabla = TablaCondicionales(
        bordes=_bordes_cubos(0.005),
        vol_min=np.full(11, 0.8, dtype=np.float64),
        vol_max=np.full(11, 1.2, dtype=np.float64),
        sell_min=np.full(11, 0.45, dtype=np.float64),
        sell_max=np.full(11, 0.55, dtype=np.float64),
    )
    kernel(
        unos,
        unos * 1.01,
        unos * 0.99,
        unos,
        unos * 10.0,
        unos * 10.0,
        unos,
        unos * 0.02,
        unos * 0.5,
        unos * 0.5,
        unos * 0.02,
        ceros.copy(),
        ceros.copy(),
        ceros.copy(),
        np.zeros(n, dtype=np.int64),
        True,
        True,
        True,
        True,
        tabla.bordes,
        tabla.vol_min,
        tabla.vol_max,
        tabla.sell_min,
        tabla.sell_max,
        0.5,
        0.15,
        0.1,
        0.5,
        0.05,
        0.08,
        0.3,
        ceros,
        ceros,
        unos * 0.5,
        unos * 0.5,
        ceros,
        ceros,
    )


def _kernel_perturbacion_impl(
    open_orig: np.ndarray,
    high_orig: np.ndarray,
    low_orig: np.ndarray,
    close_orig: np.ndarray,
    volume_orig: np.ndarray,
    media_volumen_orig: np.ndarray,
    midpoint_orig: np.ndarray,
    rango_rel_orig: np.ndarray,
    frac_open_orig: np.ndarray,
    frac_close_orig: np.ndarray,
    volatilidad_local: np.ndarray,
    taker_buy: np.ndarray,
    taker_sell: np.ndarray,
    vol_delta: np.ndarray,
    num_trades: np.ndarray,
    has_taker_buy: bool,
    has_taker_sell: bool,
    has_vol_delta: bool,
    has_num_trades: bool,
    bordes: np.ndarray,
    vol_min: np.ndarray,
    vol_max: np.ndarray,
    sell_min: np.ndarray,
    sell_max: np.ndarray,
    prop_sell_inicial: float,
    banda_max_precio: float,
    fuerza_amortiguacion: float,
    escala_volatilidad: float,
    sigma_rango_vela: float,
    ruido_posicion_ohlc: float,
    inercia_order_flow: float,
    ruido_midpoint: np.ndarray,
    ruido_rango: np.ndarray,
    ruido_open: np.ndarray,
    ruido_close: np.ndarray,
    ruido_volumen: np.ndarray,
    ruido_prop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = open_orig.shape[0]
    open_p = np.empty(n, dtype=np.float64)
    high_p = np.empty(n, dtype=np.float64)
    low_p = np.empty(n, dtype=np.float64)
    close_p = np.empty(n, dtype=np.float64)
    volume_p = np.empty(n, dtype=np.float64)

    open_p[0] = open_orig[0]
    high_p[0] = high_orig[0]
    low_p[0] = low_orig[0]
    close_p[0] = close_orig[0]
    volume_p[0] = volume_orig[0] if volume_orig[0] > 0.0 else 0.0

    midpoint_prev = midpoint_orig[0]
    prop_sell_prev = prop_sell_inicial
    amp_pos = _clip_scalar(ruido_posicion_ohlc, 0.0, 0.49)
    peso_inercia = _clip_scalar(inercia_order_flow, 0.0, 1.0)
    media_log_rango = -(sigma_rango_vela * sigma_rango_vela) * 0.5

    for i in range(1, n):
        media_vol_local = media_volumen_orig[i]
        if media_vol_local < 0.0:
            media_vol_local = 0.0

        paso_log = ruido_midpoint[i] * volatilidad_local[i] * escala_volatilidad
        midpoint = midpoint_prev * np.exp(paso_log)
        desviacion = midpoint / midpoint_orig[i] - 1.0
        if abs(desviacion) > banda_max_precio:
            limite = banda_max_precio if desviacion > 0.0 else -banda_max_precio
            exceso = desviacion - limite
            desviacion = limite + exceso * (1.0 - fuerza_amortiguacion)
            midpoint = midpoint_orig[i] * (1.0 + desviacion)
        minimo_midpoint = midpoint_orig[i] * 1e-6
        if midpoint < minimo_midpoint:
            midpoint = minimo_midpoint

        factor_rango = np.exp(media_log_rango + sigma_rango_vela * ruido_rango[i])
        rango_rel = rango_rel_orig[i] * factor_rango
        if rango_rel < 1e-9:
            rango_rel = 1e-9
        rango = midpoint * rango_rel
        max_rango = midpoint * 1.98
        if rango > max_rango:
            rango = max_rango

        low = midpoint - rango * 0.5
        high = midpoint + rango * 0.5

        min_open = frac_open_orig[i] - amp_pos
        if min_open < 0.01:
            min_open = 0.01
        max_open = frac_open_orig[i] + amp_pos
        if max_open > 0.99:
            max_open = 0.99
        frac_open = min_open + ruido_open[i] * (max_open - min_open)

        min_close = frac_close_orig[i] - amp_pos
        if min_close < 0.01:
            min_close = 0.01
        max_close = frac_close_orig[i] + amp_pos
        if max_close > 0.99:
            max_close = 0.99
        frac_close = min_close + ruido_close[i] * (max_close - min_close)

        open_p[i] = low + frac_open * rango
        high_p[i] = high
        low_p[i] = low
        close_p[i] = low + frac_close * rango

        retorno = close_p[i] / close_p[i - 1] - 1.0 if close_p[i - 1] > 0.0 else 0.0
        cubo = _buscar_cubo(retorno, bordes)

        vol_rel = vol_min[cubo] + ruido_volumen[i] * (vol_max[cubo] - vol_min[cubo])
        volume = media_vol_local * vol_rel
        if volume < 0.0:
            volume = 0.0
        volume_p[i] = volume

        sell_lo = sell_min[cubo]
        sell_hi = sell_max[cubo]
        sell_centro = (sell_lo + sell_hi) * 0.5
        prop_prev_clip = _clip_scalar(prop_sell_prev, 0.0, 1.0)
        prop_prev_en_cubo = sell_lo + (sell_hi - sell_lo) * prop_prev_clip
        ancla = peso_inercia * prop_prev_en_cubo + (1.0 - peso_inercia) * sell_centro
        z = ruido_prop[i]
        if z < 0.0:
            prop_sell = ancla + z * (ancla - sell_lo)
        else:
            prop_sell = ancla + z * (sell_hi - ancla)

        sell = volume * prop_sell
        buy = volume - sell
        if has_taker_sell:
            taker_sell[i] = sell
        if has_taker_buy:
            taker_buy[i] = buy
        if has_vol_delta:
            vol_delta[i] = buy - sell
        if has_num_trades:
            original_trades = float(num_trades[i])
            if volume_orig[i] > 0.0:
                estimado = original_trades * volume / volume_orig[i]
            else:
                estimado = original_trades
            num_trades[i] = int(estimado + 0.5) if estimado > 0.0 else 0

        midpoint_prev = midpoint
        prop_sell_prev = prop_sell

    return open_p, high_p, low_p, close_p, volume_p


def _clip_scalar(valor: float, minimo: float, maximo: float) -> float:
    if valor < minimo:
        return minimo
    if valor > maximo:
        return maximo
    return valor


def _buscar_cubo(retorno: float, bordes: np.ndarray) -> int:
    ultimo = bordes.shape[0] - 2
    for j in range(1, bordes.shape[0]):
        if retorno < bordes[j]:
            return j - 1
    return ultimo


if njit is None:
    _kernel_perturbacion = None
else:
    _clip_scalar = njit(cache=True)(_clip_scalar)
    _buscar_cubo = njit(cache=True)(_buscar_cubo)
    _kernel_perturbacion = njit(cache=True)(_kernel_perturbacion_impl)


def _validar_invariantes(df: pl.DataFrame) -> None:
    high = _f64(df, "high")
    low = _f64(df, "low")
    open_ = _f64(df, "open")
    close = _f64(df, "close")
    volume = _f64(df, "volume")

    if not (
        np.isfinite(high).all()
        and np.isfinite(low).all()
        and np.isfinite(open_).all()
        and np.isfinite(close).all()
        and np.isfinite(volume).all()
    ):
        raise ValueError("[PERT] La perturbacion genero NaN o infinitos en OHLCV.")
    if (low <= 0).any() or (open_ <= 0).any() or (high <= 0).any() or (close <= 0).any():
        raise ValueError("[PERT] La perturbacion genero precios no positivos.")
    if (high < low).any():
        raise ValueError("[PERT] La perturbacion genero high < low.")
    if ((open_ < low) | (open_ > high) | (close < low) | (close > high)).any():
        raise ValueError("[PERT] La perturbacion genero open/close fuera del rango.")
    if (volume < 0).any():
        raise ValueError("[PERT] La perturbacion genero volumen negativo.")

    if {"taker_buy_volume", "taker_sell_volume"}.issubset(df.columns):
        buy = _f64(df, "taker_buy_volume")
        sell = _f64(df, "taker_sell_volume")
        if not (np.isfinite(buy).all() and np.isfinite(sell).all()):
            raise ValueError("[PERT] La perturbacion genero NaN o infinitos en taker volume.")
        if (buy < -1e-10).any() or (sell < -1e-10).any():
            raise ValueError("[PERT] La perturbacion genero taker volume negativo.")
        if not np.allclose(buy + sell, volume, rtol=1e-9, atol=1e-9):
            raise ValueError("[PERT] taker_buy_volume + taker_sell_volume no coincide con volume.")

    if "vol_delta" in df.columns:
        delta = _f64(df, "vol_delta")
        if not np.isfinite(delta).all():
            raise ValueError("[PERT] La perturbacion genero NaN o infinitos en vol_delta.")
        if (np.abs(delta) > volume + 1e-9).any():
            raise ValueError("[PERT] La perturbacion genero |vol_delta| > volume.")

    for col in ("quote_volume", "taker_buy_quote_volume"):
        if col in df.columns and not np.isfinite(_f64(df, col)).all():
            raise ValueError(f"[PERT] La perturbacion genero NaN o infinitos en {col}.")

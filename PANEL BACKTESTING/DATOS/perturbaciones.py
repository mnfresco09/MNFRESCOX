"""Perturbaciones multivariantes en memoria, sin modificar HISTORICO.

La capa condicional usa priors monotónicos por cubo de retorno. Es deliberado:
no calcula estadisticas globales del periodo cargado, porque eso introduciria
lookahead en los primeros tramos del backtest. La calibracion empirica rolling
puede añadirse despues, siempre que cada fila consulte solo informacion pasada.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import polars as pl

try:
    from numba import njit
except ImportError:  # pragma: no cover - validado en runtime si se activan perturbaciones
    njit = None


_COLUMNAS_PRECIO = ("open", "high", "low", "close")


@dataclass(frozen=True)
class ConfiguracionPerturbaciones:
    activa: bool
    seed_global: int | None
    banda_max_precio: float
    fuerza_amortiguacion: float
    escala_volatilidad: float
    ventana_volatilidad: int
    sigma_rango_vela: float
    ruido_posicion_ohlc: float
    sigma_volumen: float
    granularidad_cubos: float
    inercia_order_flow: float
    ventana_media_volumen: int

    @classmethod
    def desde_config(cls, cfg: Any) -> "ConfiguracionPerturbaciones":
        return cls(
            activa=bool(getattr(cfg, "PERTURBACIONES_ACTIVAS", False)),
            seed_global=getattr(cfg, "PERTURBACIONES_SEED", None),
            banda_max_precio=float(getattr(cfg, "BANDA_MAX_PRECIO", 0.15)),
            fuerza_amortiguacion=float(getattr(cfg, "FUERZA_AMORTIGUACION", 0.10)),
            escala_volatilidad=float(getattr(cfg, "ESCALA_VOLATILIDAD", 0.50)),
            ventana_volatilidad=int(getattr(cfg, "VENTANA_VOLATILIDAD", 20)),
            sigma_rango_vela=float(getattr(cfg, "SIGMA_RANGO_VELA", 0.05)),
            ruido_posicion_ohlc=float(getattr(cfg, "RUIDO_POSICION_OHLC", 0.08)),
            sigma_volumen=float(getattr(cfg, "SIGMA_VOLUMEN", 0.10)),
            granularidad_cubos=float(getattr(cfg, "GRANULARIDAD_CUBOS", 0.005)),
            inercia_order_flow=float(getattr(cfg, "INERCIA_ORDER_FLOW", 0.30)),
            ventana_media_volumen=int(getattr(cfg, "VENTANA_MEDIA_VOLUMEN", 20)),
        )


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
    out = {col: df[col].to_numpy().copy() for col in df.columns if col != "timestamp"}

    open_orig = _f64(df, "open")
    high_orig = _f64(df, "high")
    low_orig = _f64(df, "low")
    close_orig = _f64(df, "close")
    volume_orig = _f64(df, "volume")

    n = df.height
    rango_orig = np.maximum(high_orig - low_orig, np.maximum(close_orig, 1.0) * 1e-9)
    midpoint_orig = (high_orig + low_orig) * 0.5
    midpoint_orig = np.maximum(midpoint_orig, 1e-9)
    rango_rel_orig = np.maximum(rango_orig / midpoint_orig, 1e-9)
    frac_open_orig = _fraccion_array(open_orig, low_orig, rango_orig)
    frac_close_orig = _fraccion_array(close_orig, low_orig, rango_orig)
    volatilidad_local = _media_rodante_pasada(rango_rel_orig, config.ventana_volatilidad)
    bordes, vol_min, vol_max, sell_min, sell_max = _arrays_cubos_condicionales(config.granularidad_cubos)

    taker_buy, has_taker_buy = _array_float_opcional(out, "taker_buy_volume", n)
    taker_sell, has_taker_sell = _array_float_opcional(out, "taker_sell_volume", n)
    vol_delta, has_vol_delta = _array_float_opcional(out, "vol_delta", n)
    num_trades, has_num_trades = _array_int_opcional(out, "num_trades", n)

    open_p, high_p, low_p, close_p, volume_p = kernel(
        open_orig,
        high_orig,
        low_orig,
        close_orig,
        volume_orig,
        midpoint_orig,
        rango_rel_orig,
        frac_open_orig,
        frac_close_orig,
        volatilidad_local,
        taker_buy,
        taker_sell,
        vol_delta,
        num_trades,
        bool(has_taker_buy),
        bool(has_taker_sell),
        bool(has_vol_delta),
        bool(has_num_trades),
        bordes,
        vol_min,
        vol_max,
        sell_min,
        sell_max,
        float(_prop_taker_sell_original(df, 0)),
        float(config.banda_max_precio),
        float(config.fuerza_amortiguacion),
        float(config.escala_volatilidad),
        int(config.ventana_media_volumen),
        float(config.sigma_rango_vela),
        float(config.ruido_posicion_ohlc),
        float(config.sigma_volumen),
        float(config.inercia_order_flow),
        rng.normal(0.0, 1.0, n).astype(np.float64),
        rng.normal(0.0, 1.0, n).astype(np.float64),
        rng.random(n).astype(np.float64),
        rng.random(n).astype(np.float64),
        rng.normal(0.0, 1.0, n).astype(np.float64),
        rng.uniform(-1.0, 1.0, n).astype(np.float64),
    )

    out["open"] = open_p
    out["high"] = high_p
    out["low"] = low_p
    out["close"] = close_p
    out["volume"] = volume_p

    if "quote_volume" in out:
        precio_medio = (high_p + low_p) * 0.5
        out["quote_volume"] = volume_p * precio_medio
    if "taker_buy_quote_volume" in out:
        precio_medio = (high_p + low_p) * 0.5
        out["taker_buy_quote_volume"] = taker_buy * precio_medio

    perturbado = df.with_columns([pl.Series(col, values) for col, values in out.items()])
    _validar_invariantes(perturbado)
    return perturbado


def _validar_columnas_minimas(df: pl.DataFrame) -> None:
    faltantes = [col for col in (*_COLUMNAS_PRECIO, "volume") if col not in df.columns]
    if faltantes:
        raise ValueError(f"[PERT] Faltan columnas obligatorias para perturbar: {faltantes}")


def _f64(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].cast(pl.Float64).to_numpy()


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


def _arrays_cubos_condicionales(
    granularidad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if granularidad <= 0:
        raise ValueError("[PERT] GRANULARIDAD_CUBOS debe ser mayor que 0.")
    g = float(granularidad)
    bordes = np.array(
        [-np.inf, -4*g, -2*g, -g, -0.4*g, -0.1*g, 0.1*g, 0.4*g, g, 2*g, 4*g, np.inf],
        dtype=np.float64,
    )
    vol_min = np.array([2.5, 1.7, 1.1, 0.8, 0.5, 0.4, 0.5, 0.8, 1.1, 1.7, 2.5], dtype=np.float64)
    vol_max = np.array([8.5, 5.2, 3.1, 2.0, 1.4, 1.2, 1.4, 2.0, 3.1, 5.2, 8.5], dtype=np.float64)
    sell_min = np.array([0.62, 0.58, 0.54, 0.51, 0.48, 0.46, 0.36, 0.30, 0.24, 0.20, 0.19], dtype=np.float64)
    sell_max = np.array([0.81, 0.76, 0.70, 0.64, 0.56, 0.54, 0.49, 0.46, 0.42, 0.38, 0.38], dtype=np.float64)
    return bordes, vol_min, vol_max, sell_min, sell_max


def _prop_taker_sell_original(df: pl.DataFrame, idx: int) -> float:
    if "taker_sell_volume" not in df.columns or "volume" not in df.columns:
        return 0.5
    volumen = float(df["volume"][idx])
    if volumen <= 0:
        return 0.5
    return min(max(float(df["taker_sell_volume"][idx]) / volumen, 0.0), 1.0)


def _array_float_opcional(out: dict[str, np.ndarray], col: str, n: int) -> tuple[np.ndarray, bool]:
    if col in out:
        return out[col].astype(np.float64, copy=False), True
    return np.zeros(n, dtype=np.float64), False


def _array_int_opcional(out: dict[str, np.ndarray], col: str, n: int) -> tuple[np.ndarray, bool]:
    if col in out:
        return out[col], True
    return np.zeros(n, dtype=np.int64), False


def _requerir_kernel_numba():
    if _kernel_perturbacion is None:
        raise RuntimeError(
            "[PERT] Las perturbaciones activas requieren numba. "
            "Instala dependencias con `pip install -r requirements.txt`."
        )
    return _kernel_perturbacion


def _kernel_perturbacion_impl(
    open_orig: np.ndarray,
    high_orig: np.ndarray,
    low_orig: np.ndarray,
    close_orig: np.ndarray,
    volume_orig: np.ndarray,
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
    ventana_media_volumen: int,
    sigma_rango_vela: float,
    ruido_posicion_ohlc: float,
    sigma_volumen: float,
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
    vol_sum = 0.0
    amp_pos = _clip_scalar(ruido_posicion_ohlc, 0.0, 0.49)
    peso_inercia = _clip_scalar(inercia_order_flow, 0.0, 1.0)
    media_log_rango = -(sigma_rango_vela * sigma_rango_vela) * 0.5
    sigma_vol = sigma_volumen if sigma_volumen > 1e-12 else 1e-12

    for i in range(1, n):
        vol_sum += volume_p[i - 1]
        fuera = i - ventana_media_volumen - 1
        if fuera >= 0:
            vol_sum -= volume_p[fuera]
        vol_count = i if i < ventana_media_volumen else ventana_media_volumen
        media_vol_local = vol_sum / vol_count if vol_count > 0 else volume_orig[i]
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

        vol_centro = (vol_min[cubo] + vol_max[cubo]) * 0.5
        vol_amp = (vol_max[cubo] - vol_min[cubo]) * 0.5
        vol_rel = vol_centro + np.tanh(ruido_volumen[i] * sigma_vol) * vol_amp
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
        if (buy < -1e-10).any() or (sell < -1e-10).any():
            raise ValueError("[PERT] La perturbacion genero taker volume negativo.")
        if not np.allclose(buy + sell, volume, rtol=1e-9, atol=1e-9):
            raise ValueError("[PERT] taker_buy_volume + taker_sell_volume no coincide con volume.")

    if "vol_delta" in df.columns:
        delta = _f64(df, "vol_delta")
        if (np.abs(delta) > volume + 1e-9).any():
            raise ValueError("[PERT] La perturbacion genero |vol_delta| > volume.")

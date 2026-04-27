"""Estrategia tendencial por cruce de EMAs.

Los indicadores se calculan **dentro de esta estrategia**: si en otra
estrategia quisieras una EMA con otra fórmula (adjust=True, otra
inicialización…), la implementarías allí sin afectar a ésta.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from NUCLEO.base_estrategia import BaseEstrategia


class EMATendencia(BaseEstrategia):
    ID = 2
    NOMBRE = "EMA Tendencia"
    COLUMNAS_REQUERIDAS = set()

    def parametros_por_defecto(self) -> dict:
        return {"ema_rapida": 21, "ema_lenta": 89}

    def espacio_busqueda(self, trial) -> dict:
        return {
            "ema_rapida": trial.suggest_int("ema_rapida", 8, 40),
            "ema_lenta": trial.suggest_int("ema_lenta", 50, 180),
        }

    # ── Indicador: EMA propia de esta estrategia ─────────────────────────
    #
    # Convención: alpha = 2/(periodo+1), sin "adjust". Se calcula con el
    # camino vectorial nativo de Polars (C, sin GIL extra). El cache de
    # combinación memoiza por (id del buffer, periodo) para que Optuna no
    # recalcule cuando repite el mismo entero.
    @staticmethod
    def _calcular_ema(close: np.ndarray, periodo: int) -> np.ndarray:
        alpha = 2.0 / (periodo + 1.0)
        return pl.Series(close).ewm_mean(alpha=alpha, adjust=False).to_numpy()

    def _ema(self, periodo: int) -> np.ndarray:
        return self.memo("ema", id(self.close), int(periodo),
                         calcular=lambda: self._calcular_ema(self.close, int(periodo)))

    def _emas(self, params: dict) -> tuple[np.ndarray, np.ndarray]:
        return self._ema(int(params["ema_rapida"])), self._ema(int(params["ema_lenta"]))

    # ── Señales y salidas ────────────────────────────────────────────────

    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        ema_r, ema_l = self._emas(params)
        diff = ema_r - ema_l
        diff_prev = self.shift(diff, 1)
        finitos = np.isfinite(diff_prev) & np.isfinite(diff)
        long_mask = finitos & (diff_prev <= 0) & (diff > 0)
        short_mask = finitos & (diff_prev >= 0) & (diff < 0)
        return self.serie_senales(df.height, long_mask, short_mask)

    def generar_salidas(self, df: pl.DataFrame, params: dict) -> pl.Series:
        ema_r, ema_l = self._emas(params)
        diff = ema_r - ema_l
        diff_prev = self.shift(diff, 1)
        finitos = np.isfinite(diff_prev) & np.isfinite(diff)
        long_exit = finitos & (diff_prev >= 0) & (diff < 0)
        short_exit = finitos & (diff_prev <= 0) & (diff > 0)
        return self.serie_senales(df.height, long_exit, short_exit)

    # ── Indicadores para el reporte HTML ─────────────────────────────────

    def indicadores_para_grafica(self, df: pl.DataFrame, params: dict) -> list[dict]:
        r = int(params.get("ema_rapida", 21))
        l = int(params.get("ema_lenta", 89))
        ema_r = self._ema(r)
        ema_l = self._ema(l)
        return [
            _serie_overlay(df, ema_r, "#f59e0b", f"EMA({r})"),
            _serie_overlay(df, ema_l, "#818cf8", f"EMA({l})"),
        ]


def _serie_overlay(df: pl.DataFrame, valores: np.ndarray, color: str, nombre: str) -> dict:
    timestamps_us = df["timestamp"].dt.epoch("us").to_numpy()
    finitos = np.isfinite(valores)
    ts_seg = (timestamps_us[finitos] // 1_000_000).astype(np.int64)
    vals = np.round(valores[finitos], 6)
    return {
        "nombre": nombre,
        "tipo": "overlay",
        "color": color,
        "data": [{"t": int(t), "v": float(v)} for t, v in zip(ts_seg, vals)],
    }

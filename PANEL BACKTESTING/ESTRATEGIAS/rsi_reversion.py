"""Estrategia de reversión a la media basada en RSI.

El RSI vive **dentro de esta estrategia**. Aquí usamos suavizado de
Wilder (alpha = 1/periodo). Otra estrategia podría usar otra fórmula
(EMA simple, SMA, etc.) sin afectar a ésta.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from NUCLEO.base_estrategia import BaseEstrategia


class RSIReversion(BaseEstrategia):
    ID = 1
    NOMBRE = "RSI Reversión"
    COLUMNAS_REQUERIDAS = set()

    def parametros_por_defecto(self) -> dict:
        return {"rsi_periodo": 14, "sobreventa": 30, "sobrecompra": 70}

    def espacio_busqueda(self, trial) -> dict:
        return {
            "rsi_periodo": trial.suggest_int("rsi_periodo", 7, 28),
            "sobreventa": trial.suggest_int("sobreventa", 20, 40),
            "sobrecompra": trial.suggest_int("sobrecompra", 60, 80),
        }

    # ── Indicador: RSI Wilder propio de esta estrategia ──────────────────
    @staticmethod
    def _calcular_rsi(close: np.ndarray, periodo: int) -> np.ndarray:
        serie = pl.Series(close)
        delta = serie.diff()
        ganancia = delta.clip(lower_bound=0)
        perdida = (-delta).clip(lower_bound=0)
        alpha = 1.0 / periodo
        media_gan = ganancia.ewm_mean(alpha=alpha, adjust=False)
        media_per = perdida.ewm_mean(alpha=alpha, adjust=False)
        rs = media_gan / media_per
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Las primeras `periodo` posiciones no son fiables: las marcamos NaN.
        out = rsi.to_numpy()
        if periodo > 0 and out.shape[0] > periodo:
            out[:periodo] = np.nan
        return out

    def _rsi(self, periodo: int) -> np.ndarray:
        return self.memo("rsi", id(self.close), int(periodo),
                         calcular=lambda: self._calcular_rsi(self.close, int(periodo)))

    # ── Señales y salidas ────────────────────────────────────────────────

    def generar_señales(self, df: pl.DataFrame, params: dict) -> pl.Series:
        periodo = int(params["rsi_periodo"])
        sobreventa = float(params["sobreventa"])
        sobrecompra = float(params["sobrecompra"])
        rsi = self._rsi(periodo)
        rsi_prev = self.shift(rsi, 1)
        finitos = np.isfinite(rsi_prev) & np.isfinite(rsi)
        long_mask = finitos & (rsi_prev < sobreventa) & (rsi >= sobreventa)
        short_mask = finitos & (rsi_prev > sobrecompra) & (rsi <= sobrecompra)
        return self.serie_senales(df.height, long_mask, short_mask)

    def generar_salidas(self, df: pl.DataFrame, params: dict) -> pl.Series:
        periodo = int(params["rsi_periodo"])
        rsi = self._rsi(periodo)
        rsi_prev = self.shift(rsi, 1)
        finitos = np.isfinite(rsi_prev) & np.isfinite(rsi)
        long_exit = finitos & (rsi_prev < 50) & (rsi >= 50)
        short_exit = finitos & (rsi_prev > 50) & (rsi <= 50)
        return self.serie_senales(df.height, long_exit, short_exit)

    # ── Indicadores para el reporte HTML ─────────────────────────────────

    def indicadores_para_grafica(self, df: pl.DataFrame, params: dict) -> list[dict]:
        periodo = int(params.get("rsi_periodo", 14))
        sobreventa = float(params.get("sobreventa", 30))
        sobrecompra = float(params.get("sobrecompra", 70))
        valores = self._rsi(periodo)
        timestamps_us = df["timestamp"].dt.epoch("us").to_numpy()
        finitos = np.isfinite(valores)
        ts_seg = (timestamps_us[finitos] // 1_000_000).astype(np.int64)
        vals = np.round(valores[finitos], 4)
        return [{
            "nombre": f"RSI({periodo})",
            "tipo": "pane",
            "color": "#818cf8",
            "data": [{"t": int(t), "v": float(v)} for t, v in zip(ts_seg, vals)],
            "niveles": [
                {"valor": sobrecompra, "color": "#ef535066"},
                {"valor": 50, "color": "#64748b88"},
                {"valor": sobreventa, "color": "#26a69a66"},
            ],
            "min": 0,
            "max": 100,
        }]

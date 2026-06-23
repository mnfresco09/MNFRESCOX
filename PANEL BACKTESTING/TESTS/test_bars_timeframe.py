from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import optuna
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from NUCLEO.contexto import crear_contexto  # noqa: E402
from OPTIMIZACION import runner  # noqa: E402
from OPTIMIZACION.runner import ExitConfig, _sim_config  # noqa: E402
from SALIDAS import velas as velas_cfg  # noqa: E402
from DATOS.resampleo import resamplear  # noqa: E402


class BarsTimeframeTest(unittest.TestCase):
    def test_exit_velas_bars_se_interpreta_en_timeframe_de_estrategia(self) -> None:
        df_base = _df_1m(30)
        df_tf = resamplear(df_base, "5m")
        ctx = crear_contexto(df_base=df_base, df_tf=df_tf, timeframe="5m")
        salida = ExitConfig(tipo="BARS", sl_pct=25.0, tp_pct=0.0, velas=3)

        sim_cfg = _sim_config(salida, ctx=ctx)

        self.assertEqual(salida.velas, 3)
        self.assertEqual(sim_cfg.exit_velas, 15)

    def test_exit_velas_bars_no_escala_si_el_timeframe_ya_es_base(self) -> None:
        df_base = _df_1m(30)
        ctx = crear_contexto(df_base=df_base, df_tf=df_base, timeframe="1m")
        salida = ExitConfig(tipo="BARS", sl_pct=25.0, tp_pct=0.0, velas=3)

        sim_cfg = _sim_config(salida, ctx=ctx)

        self.assertEqual(sim_cfg.exit_velas, 3)

    def test_exit_velas_bars_1h_convierte_a_barras_base_para_el_motor(self) -> None:
        df_base = _df_1m(180)
        df_tf = resamplear(df_base, "1h")
        ctx = crear_contexto(df_base=df_base, df_tf=df_tf, timeframe="1h")
        salida = ExitConfig(tipo="BARS", sl_pct=25.0, tp_pct=0.0, velas=20)

        sim_cfg = _sim_config(salida, ctx=ctx)

        self.assertEqual(salida.velas, 20)
        self.assertEqual(sim_cfg.exit_velas, 1200)

    def test_bars_permite_apagar_sl_de_emergencia(self) -> None:
        prev_exit_type = runner.cfg.EXIT_TYPE
        prev_usar_sl = getattr(velas_cfg, "USAR_SL_EMERGENCIA", None)
        prev_sl = velas_cfg.EXIT_SL_PCT
        try:
            runner.cfg.EXIT_TYPE = "BARS"
            velas_cfg.EXIT_SL_PCT = 50.0

            velas_cfg.USAR_SL_EMERGENCIA = False
            sin_sl = next(runner._salidas_a_ejecutar())
            self.assertFalse(sin_sl.sl_emergencia)
            self.assertEqual(sin_sl.sl_pct, 0.0)

            velas_cfg.USAR_SL_EMERGENCIA = True
            con_sl = next(runner._salidas_a_ejecutar())
            self.assertTrue(con_sl.sl_emergencia)
            self.assertEqual(con_sl.sl_pct, 50.0)
        finally:
            runner.cfg.EXIT_TYPE = prev_exit_type
            velas_cfg.EXIT_SL_PCT = prev_sl
            if prev_usar_sl is None:
                try:
                    delattr(velas_cfg, "USAR_SL_EMERGENCIA")
                except AttributeError:
                    pass
            else:
                velas_cfg.USAR_SL_EMERGENCIA = prev_usar_sl

    def test_bars_optimizado_no_busca_sl_si_emergencia_esta_apagado(self) -> None:
        salida = ExitConfig(
            tipo="BARS",
            sl_pct=0.0,
            tp_pct=0.0,
            velas=20,
            sl_emergencia=False,
            optimizar=True,
            sl_min=5,
            sl_max=50,
            velas_min=20,
            velas_max=20,
        )
        trial = optuna.trial.FixedTrial({"exit_velas": 20})

        salida_trial, params = runner._salida_para_trial(salida, trial)

        self.assertFalse(salida_trial.sl_emergencia)
        self.assertEqual(salida_trial.sl_pct, 0.0)
        self.assertEqual(params, {"exit_velas": 20})


def _df_1m(n: int) -> pl.DataFrame:
    idx = np.arange(n, dtype=np.float64)
    close = 100.0 + idx * 0.1
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=n - 1),
                interval="1m",
                eager=True,
            ),
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.full(n, 100.0, dtype=np.float64),
        }
    )


if __name__ == "__main__":
    unittest.main()

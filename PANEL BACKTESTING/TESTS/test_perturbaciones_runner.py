from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DATOS.perturbaciones import (  # noqa: E402
    ConfiguracionPerturbaciones,
    aplicar_perturbaciones,
    seed_para_trial,
    validar_kernel_numba,
)
from DATOS.resampleo import resamplear  # noqa: E402
from NUCLEO.contexto import crear_contexto  # noqa: E402
from OPTIMIZACION import runner  # noqa: E402


class PerturbacionesRunnerTest(unittest.TestCase):
    def test_jobs_perturbados_se_capar_por_memoria_sin_forzar_secuencial(self) -> None:
        self.assertEqual(runner._normalizar_jobs_perturbaciones(1, max_jobs=4), 1)
        self.assertEqual(runner._normalizar_jobs_perturbaciones(2, max_jobs=4), 2)
        self.assertEqual(runner._normalizar_jobs_perturbaciones(8, max_jobs=4), 4)

    def test_ctx_perturbado_reutiliza_plan_temporal_sin_resampleo_generico(self) -> None:
        try:
            validar_kernel_numba()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        df_base = _df_1m_order_flow(180)
        df_tf = resamplear(df_base, "5m")
        ctx = crear_contexto(df_base=df_base, df_tf=df_tf, timeframe="5m")
        config = ConfiguracionPerturbaciones(
            activa=True,
            seed_global=123,
            granularidad_cubos=0.005,
            percentil_tabla=0.10,
        ).con_tabla_desde(df_base)
        seed = seed_para_trial(
            config,
            trial_numero=3,
            activo="BTC",
            timeframe="5m",
            estrategia_id=5,
            salida_tipo="TRAILING",
        )
        esperado = resamplear(aplicar_perturbaciones(df_base, config, seed=seed), "5m")

        with (
            patch.object(runner, "resamplear", side_effect=AssertionError("resampleo generico")),
            patch.object(runner, "crear_contexto", side_effect=AssertionError("contexto generico")),
        ):
            ctx_trial = runner._ctx_para_trial(
                ctx=ctx,
                timeframe="5m",
                perturbaciones=config,
                seed=seed,
            )

        self.assertEqual(ctx_trial.df_base.height, df_base.height)
        self.assertEqual(ctx_trial.df_tf.height, esperado.height)
        self.assertFalse(ctx_trial.es_min_tf)
        np.testing.assert_array_equal(ctx_trial.tf_to_base_idx, ctx.tf_to_base_idx)
        for columna in esperado.columns:
            if columna == "timestamp":
                self.assertTrue(ctx_trial.df_tf[columna].equals(esperado[columna]))
            else:
                np.testing.assert_allclose(
                    ctx_trial.df_tf[columna].to_numpy(),
                    esperado[columna].to_numpy(),
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_validacion_de_invariantes_es_configurable_para_ruta_caliente(self) -> None:
        try:
            validar_kernel_numba()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        df_base = _df_1m_order_flow(180)
        config = ConfiguracionPerturbaciones(
            activa=True,
            seed_global=123,
            granularidad_cubos=0.005,
            percentil_tabla=0.10,
            validar_invariantes=False,
        ).con_tabla_desde(df_base)
        seed = seed_para_trial(
            config,
            trial_numero=4,
            activo="BTC",
            timeframe="5m",
            estrategia_id=5,
            salida_tipo="TRAILING",
        )

        with patch("DATOS.perturbaciones._validar_invariantes", Mock(side_effect=AssertionError)):
            perturbado = aplicar_perturbaciones(df_base, config, seed=seed)

        self.assertEqual(perturbado.height, df_base.height)


def _df_1m_order_flow(n: int) -> pl.DataFrame:
    idx = np.arange(n, dtype=np.float64)
    close = 100.0 * np.exp(np.cumsum(0.0008 * np.sin(idx / 9.0)))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    rango = close * (0.001 + 0.0004 * (np.cos(idx / 13.0) + 1.0))
    high = np.maximum(open_, close) + rango
    low = np.minimum(open_, close) - rango
    volume = 100.0 + 20.0 * (np.sin(idx / 17.0) + 1.0)
    sell_prop = np.clip(0.48 + 0.12 * np.cos(idx / 11.0), 0.05, 0.95)
    taker_sell = volume * sell_prop
    taker_buy = volume - taker_sell
    precio_medio = (high + low) * 0.5
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 1) + pl.duration(minutes=n - 1),
                interval="1m",
                eager=True,
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * precio_medio,
            "num_trades": (50 + volume).astype(np.int64),
            "taker_buy_volume": taker_buy,
            "taker_buy_quote_volume": taker_buy * precio_medio,
            "taker_sell_volume": taker_sell,
            "vol_delta": taker_buy - taker_sell,
        }
    )


if __name__ == "__main__":
    unittest.main()

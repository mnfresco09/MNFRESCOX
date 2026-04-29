from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MOTOR import MOTIVOS, simular_full  # noqa: E402
from NUCLEO.contexto import SimConfigMotor  # noqa: E402


class TrailingExitTest(unittest.TestCase):
    def test_trailing_cierra_con_motivo_y_precio_esperados(self) -> None:
        arrays = SimpleNamespace(
            timestamps=np.array([1, 2, 3, 4], dtype=np.int64),
            opens=np.array([100.0, 100.0, 102.5, 102.5], dtype=np.float64),
            highs=np.array([100.0, 103.0, 102.8, 103.0], dtype=np.float64),
            lows=np.array([100.0, 101.5, 101.8, 102.0], dtype=np.float64),
            closes=np.array([100.0, 102.5, 102.0, 102.8], dtype=np.float64),
            salidas_neutras=np.zeros(4, dtype=np.int8),
        )
        cfg = SimConfigMotor(
            saldo_inicial=10_000.0,
            saldo_por_trade=500.0,
            apalancamiento=10.0,
            saldo_minimo=1_000.0,
            comision_pct=0.0005,
            comision_lados=2,
            exit_type="TRAILING",
            exit_sl_pct=20.0,
            exit_tp_pct=0.0,
            exit_velas=0,
            exit_trail_act_pct=20.0,
            exit_trail_dist_pct=10.0,
        )

        result = simular_full(
            arrays,
            np.array([1, 0, 0, 0], dtype=np.int8),
            sim_cfg=cfg,
        )
        trades = result.take_trades()

        self.assertEqual(MOTIVOS[int(trades["motivo_salida"][0])], "TRAILING")
        self.assertEqual(int(trades["idx_salida"][0]), 2)
        self.assertAlmostEqual(float(trades["precio_salida"][0]), 102.0)


if __name__ == "__main__":
    unittest.main()

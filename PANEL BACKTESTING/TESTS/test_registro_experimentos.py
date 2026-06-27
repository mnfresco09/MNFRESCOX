"""Tests de COMUN/registro_experimentos.py (Fase 0).

Pura stdlib (sqlite3): no requieren Polars ni el motor.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from COMUN.registro_experimentos import RegistroExperimentos  # noqa: E402


def _trial(numero, score, sharpe, trades):
    return {
        "numero": numero,
        "score": score,
        "parametros": {"rsi": 14, "th": numero},
        "metricas": {"sharpe_ratio": sharpe, "total_trades": trades, "saldo_final": 10000 + numero},
    }


class TestRegistro(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.db = Path(self._dir.name) / "experimentos.db"

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_registro_y_conteo(self):
        with RegistroExperimentos(self.db) as r:
            run_id = r.registrar_run(
                activo="BTC", timeframe="1h", estrategia_id=1,
                estrategia_nombre="RSI", salida_tipo="BARS", modo="investigacion",
                n_trials=3, sampler="QMC", funcion_score="PSR",
                huella={"digest": "abc", "datos_hash": "d1", "git_commit": None},
            )
            n_ins = r.registrar_trials(run_id, [
                _trial(0, 0.9, 0.10, 120),
                _trial(1, 0.8, 0.05, 80),
                _trial(2, 0.7, 0.02, 40),
            ])
            self.assertEqual(n_ins, 3)
            self.assertEqual(r.contar_configuraciones("BTC"), 3)
            self.assertEqual(r.contar_configuraciones("BTC", estrategia_id=1), 3)
            self.assertEqual(r.contar_configuraciones("ETH"), 0)
            sharpes = r.sharpes_configuraciones("BTC")
            self.assertEqual(sorted(sharpes), [0.02, 0.05, 0.10])

    def test_acumula_entre_runs(self):
        # El N del DSR es ACUMULATIVO: suma todos los runs del activo.
        with RegistroExperimentos(self.db) as r:
            for run in range(2):
                rid = r.registrar_run(
                    activo="BTC", timeframe="1h", estrategia_id=1,
                    estrategia_nombre="RSI", salida_tipo="BARS", modo="investigacion",
                    n_trials=2,
                )
                r.registrar_trials(rid, [_trial(0, 0.5, 0.1, 50), _trial(1, 0.4, 0.05, 50)])
            self.assertEqual(r.contar_configuraciones("BTC"), 4)
            resumen = {x.activo: x for x in r.resumen_por_activo()}
            self.assertEqual(resumen["BTC"].n_runs, 2)
            self.assertEqual(resumen["BTC"].n_configuraciones, 4)

    def test_persiste_en_disco(self):
        with RegistroExperimentos(self.db) as r:
            rid = r.registrar_run(
                activo="GOLD", timeframe="4h", estrategia_id=2,
                estrategia_nombre="EMA", salida_tipo="FIXED", modo="investigacion",
                n_trials=1,
            )
            r.registrar_trials(rid, [_trial(0, 1.0, 0.2, 200)])
        # Reabrir en otra conexión: los datos siguen ahí.
        with RegistroExperimentos(self.db) as r2:
            self.assertEqual(r2.contar_configuraciones("GOLD"), 1)


if __name__ == "__main__":
    unittest.main()

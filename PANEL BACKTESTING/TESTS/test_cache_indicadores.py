from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from NUCLEO.base_estrategia import CacheIndicadores  # noqa: E402


class CacheIndicadoresTest(unittest.TestCase):
    def test_limita_crecimiento_con_miles_de_claves_unicas(self) -> None:
        cache = CacheIndicadores(max_entries=32)

        for idx in range(1_000):
            cache.put(("indicador", idx), object())

        self.assertEqual(len(cache), 32)
        self.assertIsNone(cache.get(("indicador", 0)))
        self.assertIsNotNone(cache.get(("indicador", 999)))

    def test_evicta_la_clave_menos_reciente(self) -> None:
        cache = CacheIndicadores(max_entries=3)
        cache.put(("k", 1), "uno")
        cache.put(("k", 2), "dos")
        cache.put(("k", 3), "tres")

        self.assertEqual(cache.get(("k", 1)), "uno")
        cache.put(("k", 4), "cuatro")

        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.get(("k", 2)))
        self.assertEqual(cache.get(("k", 1)), "uno")
        self.assertEqual(cache.get(("k", 3)), "tres")
        self.assertEqual(cache.get(("k", 4)), "cuatro")

    def test_limita_memoria_estimada_de_arrays_numpy(self) -> None:
        cache = CacheIndicadores(max_entries=None, max_bytes=80)

        cache.put(("serie", 1), np.ones(5, dtype=np.float64))
        cache.put(("serie", 2), np.ones(5, dtype=np.float64))
        cache.put(("serie", 3), np.ones(5, dtype=np.float64))

        self.assertLessEqual(cache.bytes_estimados, 80)
        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get(("serie", 1)))
        self.assertIsNotNone(cache.get(("serie", 2)))
        self.assertIsNotNone(cache.get(("serie", 3)))


if __name__ == "__main__":
    unittest.main()

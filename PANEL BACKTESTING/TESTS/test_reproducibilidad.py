"""Tests de COMUN/reproducibilidad.py (Fase 0).

Pura stdlib: no requieren Polars ni el motor.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from COMUN import reproducibilidad as repro  # noqa: E402


def _cfg_falso(tmp_data: Path, **extra) -> types.SimpleNamespace:
    base = dict(
        FECHA_INICIO="2021-01-01",
        FECHA_FIN="2024-12-31",
        HOLDOUT_INICIO="2024-01-01",
        MODO="investigacion",
        N_TRIALS=300,
        APALANCAMIENTO=8,
        RAIZ=tmp_data.parent,
        CARPETA_HISTORICO=tmp_data.parent,
    )
    base.update(extra)
    return types.SimpleNamespace(**base)


class TestHuella(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name) / "datos.bin"
        self.tmp.write_bytes(b"datos de prueba deterministas" * 100)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_huella_determinista(self):
        cfg = _cfg_falso(self.tmp)
        h1 = repro.calcular_huella(cfg, self.tmp)
        h2 = repro.calcular_huella(cfg, self.tmp)
        self.assertEqual(h1.digest, h2.digest)
        self.assertEqual(len(h1.digest), 64)  # SHA-256 hex

    def test_cambio_config_cambia_huella(self):
        cfg_a = _cfg_falso(self.tmp, N_TRIALS=300)
        cfg_b = _cfg_falso(self.tmp, N_TRIALS=301)
        self.assertNotEqual(
            repro.calcular_huella(cfg_a, self.tmp).digest,
            repro.calcular_huella(cfg_b, self.tmp).digest,
        )

    def test_cambio_datos_cambia_huella(self):
        cfg = _cfg_falso(self.tmp)
        h1 = repro.calcular_huella(cfg, self.tmp)
        self.tmp.write_bytes(b"otros datos distintos")
        h2 = repro.calcular_huella(cfg, self.tmp)
        self.assertNotEqual(h1.datos_hash, h2.datos_hash)
        self.assertNotEqual(h1.digest, h2.digest)

    def test_hash_fichero_coincide_con_sha256_conocido(self):
        import hashlib

        esperado = hashlib.sha256(self.tmp.read_bytes()).hexdigest()
        self.assertEqual(repro.hash_fichero(self.tmp), esperado)

    def test_git_opcional_no_rompe(self):
        # En un directorio sin repo git, info_git devuelve (None, None) sin error.
        commit, sucio = repro.info_git(self.tmp.parent)
        self.assertTrue(commit is None or isinstance(commit, str))
        self.assertTrue(sucio is None or isinstance(sucio, bool))

    def test_como_dict_serializable(self):
        import json

        cfg = _cfg_falso(self.tmp)
        d = repro.calcular_huella(cfg, self.tmp).como_dict()
        json.dumps(d)  # no debe lanzar
        self.assertIn("digest", d)
        self.assertIn("datos_hash", d)


if __name__ == "__main__":
    unittest.main()

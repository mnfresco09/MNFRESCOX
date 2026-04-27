"""
Verificación SHA-256 de ZIPs contra el archivo CHECKSUM de Binance Vision.
Lee en bloques de 8 192 bytes para no cargar el ZIP completo en memoria.
"""
import hashlib
from pathlib import Path

from .utils import get_logger

log = get_logger(__name__)


def verificar_sha256(zip_path: Path, checksum_path: Path) -> bool:
    """
    Retorna True si el hash SHA-256 del ZIP coincide con el CHECKSUM.
    Formato del .CHECKSUM: '<hash64>  <nombre_archivo>'
    """
    texto = checksum_path.read_text(encoding="utf-8").strip()
    hash_esperado = texto.split()[0].lower()

    sha256 = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for bloque in iter(lambda: f.read(8192), b""):
            sha256.update(bloque)
    hash_calculado = sha256.hexdigest()

    if hash_calculado != hash_esperado:
        log.error(f"SHA-256 incorrecto para {zip_path.name}")
        log.error(f"  esperado:   {hash_esperado}")
        log.error(f"  calculado:  {hash_calculado}")
        return False
    return True

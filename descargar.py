#!/usr/bin/env python3
"""
Script de descarga de datos de mercado desde Binance Vision.
Descarga klines, markPrice, indexPrice, premiumIndex y metrics
para BTCUSDT y GOLDUSDT (Futuros USD-M, 1m) y los guarda en
PANEL BACKTESTING/HISTORICO/ como Parquet snappy.

Uso:
    python descargar.py                   # descarga BTCUSDT y GOLDUSDT
    python descargar.py BTCUSDT           # solo BTC
    python descargar.py BTCUSDT GOLDUSDT  # ambos (explícito)
"""
import sys
from pathlib import Path

# Asegurar que el paquete DESCARGADOR sea localizable desde la raíz
sys.path.insert(0, str(Path(__file__).parent))

from DESCARGADOR.descargador import ejecutar

if __name__ == "__main__":
    activos = sys.argv[1:] or None
    try:
        ejecutar(activos)
    except (ValueError, RuntimeError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

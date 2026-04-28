from pathlib import Path
from datetime import date

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
HISTORICO = RAIZ_PROYECTO / "PANEL BACKTESTING" / "HISTORICO"
TMP = Path(__file__).resolve().parent / "tmp"

BASE_URL = "https://data.binance.vision/data"
MERCADO = "um"
INTERVALO = "1m"
MAX_REINTENTOS = 3
TIMEOUT_SEGUNDOS = 120
CHUNK_SIZE = 65_536

ACTIVOS = {
    "BTCUSDT": {
        "klines":             date(2019, 9, 8),
        "markPriceKlines":    date(2019, 9, 8),
        "indexPriceKlines":   date(2019, 9, 8),
        "premiumIndexKlines": date(2019, 9, 8),
    },
    "GOLDUSDT": {
        "klines":             date(2020, 1, 1),
        "markPriceKlines":    date(2020, 1, 1),
        "indexPriceKlines":   date(2020, 1, 1),
        "premiumIndexKlines": date(2020, 1, 1),
    },
}

TIPOS = ["klines", "markPriceKlines", "indexPriceKlines", "premiumIndexKlines"]

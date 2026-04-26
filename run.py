from pathlib import Path
import os
import sys


RAIZ = Path(__file__).resolve().parent
VENV_PYTHON = RAIZ / ".venv" / "bin" / "python"

if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

PANEL_DIR = RAIZ / "PANEL BACKTESTING"
sys.path.insert(0, str(PANEL_DIR))

from NUCLEO.runner_fijo import main


if __name__ == "__main__":
    main()

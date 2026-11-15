"""Train only gpr. Trains all four cases and writes to this folder's results/ and saved_models/.

    python gpr/train.py
    python gpr/train.py --cases nmos_off
    python gpr/train.py --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.training import main

if __name__ == "__main__":
    main("gpr", Path(__file__).resolve().parent)

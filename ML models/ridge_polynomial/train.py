"""Train only the ridge polynomial. Trains all four cases and writes to this folder's results/ and saved_models/.

    python ridge_polynomial/train.py
    python ridge_polynomial/train.py --cases nmos_off
    python ridge_polynomial/train.py --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.training import main

if __name__ == "__main__":
    main("ridge", Path(__file__).resolve().parent)

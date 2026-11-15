"""Train only the Random Forest. Trains all four cases and writes to this folder's results/ and saved_models/.

    python random_forest/train.py
    python random_forest/train.py --cases nmos_off
    python random_forest/train.py --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.training import main

if __name__ == "__main__":
    main("rf", Path(__file__).resolve().parent)

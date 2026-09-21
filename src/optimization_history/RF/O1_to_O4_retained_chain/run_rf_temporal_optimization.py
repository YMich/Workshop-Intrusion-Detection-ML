import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from rf_temporal_optimization import run_temporal_optimization  # noqa: E402


def main() -> None:
    run_temporal_optimization()


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from rf_static_optimization_chain import run_static_optimization  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled RF Section 2.2 chain: "
            "57 fixed baseline -> 10 fixed -> 12 fixed -> shared tuning on 12."
        )
    )
    parser.add_argument(
        "--grid",
        choices=("focused", "full"),
        default="full",
        help=(
            "O3 grid. 'full' is the report-quality 2x2x2x2 search; "
            "'focused' is a faster smoke-test search."
        ),
    )
    args = parser.parse_args()
    run_static_optimization(grid_mode=args.grid)


if __name__ == "__main__":
    main()

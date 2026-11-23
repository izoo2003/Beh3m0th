"""CLI entry for the offline golden suite and the optional live judge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beh3m0th.evaluation.suite import run_suite  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Beh3m0th golden trajectories.")
    parser.add_argument(
        "--mode",
        choices=("deterministic", "live"),
        default="deterministic",
        help="deterministic is offline; live calls the free-tier judge.",
    )
    args = parser.parse_args()
    raise SystemExit(run_suite(args.mode))


if __name__ == "__main__":
    main()

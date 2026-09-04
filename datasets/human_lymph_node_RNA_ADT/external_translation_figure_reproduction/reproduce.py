"""One-command reproduction of lymph-node Figure 4d-f and Figure S3."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from workflow import reproduce


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]


def default_data_root() -> Path:
    value = os.environ.get("SPADIVA_DATA_ROOT")
    if value:
        return Path(value)
    return SCRIPT_DIR.parents[2] / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the manuscript-exact lymph-node translation panels "
            "from deposited Source Data, Zenodo inputs, and the frozen "
            "totalVI prediction."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Root of the extracted Zenodo data package (or set SPADIVA_DATA_ROOT).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results",
        help="Output directory (default: repository-level results/ directory).",
    )
    parser.add_argument(
        "--retrain-totalvi",
        action="store_true",
        help="Retrain totalVI on the validated fixed split instead of using the frozen prediction.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional totalVI retraining.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = reproduce(
        data_root=args.data_root,
        artifact_root=SCRIPT_DIR,
        results_root=args.results_dir,
        retrain=args.retrain_totalvi,
        device=args.device,
    )
    observed = validation["observed"]
    print("Validation passed.")
    print(f"n_features = {observed['n_features']}")
    print(f"Spearman r = {observed['spearman_r']:.12f}")
    print(f"one-sided Mann-Whitney p = {observed['mannwhitney_p']:.12g}")
    print(f"Results: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()

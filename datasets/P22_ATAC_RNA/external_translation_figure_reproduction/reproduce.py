"""Single entry point for P22 Figure 3f-h reproduction.

Examples
--------
Exact, offline figure reproduction from deposited compact source data::

    python datasets/P22_ATAC_RNA/external_translation_figure_reproduction/reproduce.py

From-scratch external-translation analysis (requires the assets reported by
``--preflight-only``)::

    python datasets/P22_ATAC_RNA/external_translation_figure_reproduction/reproduce.py \
        --mode full --data-root /path/to/zenodo-data
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from _plotting import SOURCE_FILES, render_panels


WORKFLOW_ROOT = Path(__file__).resolve().parent


def find_repository_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "datasets").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the spaDIVA_reproducibility root from this script."
    )


REPOSITORY_ROOT = find_repository_root(WORKFLOW_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the P22 external-translation panels in spaDIVA Figure 3. "
            "Compact mode is the offline, exact default."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("compact", "full"),
        default="compact",
        help="compact redraws deposited source data; full reruns MultiVI and analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: repository results/Figure_3).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root of the extracted Zenodo data package, required by full mode.",
    )
    parser.add_argument(
        "--p22-ga",
        type=Path,
        default=None,
        help="Path to the deposited 3.67 GB P22_GA.txt used by full mode.",
    )
    parser.add_argument(
        "--gmt-dir",
        type=Path,
        default=None,
        help="Directory containing the three frozen paper-version GMT files.",
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="scvi-tools accelerator for full-mode MultiVI training.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate full-mode external assets, then exit.",
    )
    return parser


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "anndata",
        "scanpy",
        "scvi-tools",
        "mudata",
        "torch",
        "gseapy",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_output_dir(value: Path | None) -> Path:
    if value is None:
        return REPOSITORY_ROOT / "results" / "Figure_3"
    value = value.expanduser()
    if not value.is_absolute():
        value = REPOSITORY_ROOT / value
    return value.resolve()


def _resolve_data_root(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    environment_value = os.environ.get("SPADIVA_DATA_ROOT")
    if environment_value:
        return Path(environment_value).expanduser().resolve()
    return (REPOSITORY_ROOT / "data").resolve()


def _full_preflight(args: argparse.Namespace, data_root: Path) -> Any:
    from _full_pipeline import preflight_full

    return preflight_full(
        workflow_root=WORKFLOW_ROOT,
        data_root=data_root,
        public_ga_path=args.p22_ga,
        gmt_dir=args.gmt_dir,
    )


def main() -> int:
    args = build_parser().parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = _resolve_data_root(args.data_root)

    try:
        if args.preflight_only:
            if args.mode != "full":
                raise ValueError("--preflight-only is only meaningful with --mode full.")
            inputs = _full_preflight(args, data_root)
            print("P22 full-mode preflight passed.")
            print(f"P22_GA.txt: {inputs.public_ga_path}")
            print("GMT snapshots:")
            for path in inputs.gmt_paths:
                print(f"  - {path}")
            return 0

        run_context: dict[str, Any]
        if args.mode == "compact":
            source_dir = WORKFLOW_ROOT / "source_data"
            run_context = {
                "mode": "compact",
                "description": (
                    "Exact offline redraw from deposited paper source data; "
                    "no model training or GSEA database access."
                ),
                "z_source": "Z_poe",
            }
        else:
            from _full_pipeline import run_full_pipeline

            source_dir, run_context = run_full_pipeline(
                workflow_root=WORKFLOW_ROOT,
                data_root=data_root,
                output_dir=output_dir,
                public_ga_path=args.p22_ga,
                gmt_dir=args.gmt_dir,
                accelerator=args.accelerator,
            )

        validation = render_panels(
            source_dir,
            output_dir,
            strict_reference=args.mode == "compact",
        )
        validation["mode"] = args.mode
        _write_json(output_dir / "translation_validation.json", validation)

        source_records = [
            _file_record(source_dir / filename)
            for filename in SOURCE_FILES.values()
        ]
        output_records = [
            _file_record(output_dir / name)
            for name in ("panel_f.pdf", "panel_g.pdf", "panel_h.pdf")
        ]
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "command": " ".join(sys.argv),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": _package_versions(),
            "repository_root": _display_path(REPOSITORY_ROOT),
            "workflow_root": _display_path(WORKFLOW_ROOT),
            "output_dir": _display_path(output_dir),
            "run_context": run_context,
            "source_files": source_records,
            "output_files": output_records,
            "validation_file": "translation_validation.json",
        }
        _write_json(output_dir / "translation_run_manifest.json", manifest)

        print(f"P22 Figure 3 reproduction passed in {args.mode!r} mode.")
        print(f"Output directory: {output_dir}")
        print(
            "Figure 3f: "
            f"n={validation['figure3f']['n_genes']:,}, "
            f"Spearman r={validation['figure3f']['spearman_r']:.12f}"
        )
        for name in ("panel_f.pdf", "panel_g.pdf", "panel_h.pdf"):
            print(f"  - {output_dir / name}")
        print(f"  - {output_dir / 'translation_validation.json'}")
        print(f"  - {output_dir / 'translation_run_manifest.json'}")
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

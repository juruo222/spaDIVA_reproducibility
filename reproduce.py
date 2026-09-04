"""Redraw supported manuscript panels from saved results.

Model runs are independent entry points documented in MODEL_RUNS.md. This
command executes the compact plotting profiles, writes generated outputs to
the selected results directory, and retains the versioned inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.check_reproducibility_setup import check_replot_setup, file_sha256
from scripts.figure_workflows import resolve_data_root, selected_workflows


REPO_ROOT = Path(__file__).resolve().parent


def plotting_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "anndata", "h5py"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not recorded"
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "2", "3", "4", "5"), default="all")
    parser.add_argument("--data-root", type=Path, help="Extracted data root, or set SPADIVA_DATA_ROOT.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results",
                        help="Generated result root (default: results/); source-data directories are reserved for inputs.")
    parser.add_argument("--list", action="store_true", help="List supported panels, scripts and output files, then exit.")
    parser.add_argument("--check-only", action="store_true", help="Validate required saved inputs and packages, then exit.")
    parser.add_argument("--import-check", action="store_true", help="Actually import plotting dependencies during preflight.")
    return parser.parse_args()


def resolve_output(value: Path) -> Path:
    path = value.expanduser()
    path = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    protected = [REPO_ROOT / name for name in ("datasets", "spaDIVA", "scripts", "environments", "tests")]
    if path == REPO_ROOT or any(path == item or item in path.parents for item in protected):
        raise ValueError("Generated results require results/ or a separate output directory; repository and input/code directories are reserved.")
    return path


def main() -> int:
    args = parse_args()
    workflows = selected_workflows(args.figure)
    if args.list:
        for workflow in workflows:
            print(f"{workflow.key}: {', '.join(workflow.panels)}")
            print(f"  code: {workflow.script}")
            for output in workflow.outputs:
                print(f"  output: results/{output}")
        print("\nUnlisted panels are outside this saved-result entry. See README.md.")
        return 0

    data_root = resolve_data_root(REPO_ROOT, args.data_root)
    status = check_replot_setup(REPO_ROOT, workflows, data_root, import_check=args.import_check)
    if status or args.check_only:
        return status

    output_root = resolve_output(args.output_dir)
    if data_root is not None and (output_root == data_root or data_root in output_root.parents):
        raise ValueError("Generated figures must be outside the extracted input-data directory.")
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(output_root / ".cache" / "matplotlib")
    if data_root is not None:
        env["SPADIVA_DATA_ROOT"] = str(data_root)

    report = {
        "mode": "saved_results", "model_training": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "packages": plotting_versions(),
        "manifest_sha256": file_sha256(REPO_ROOT / "MANIFEST.tsv"),
        "workflows": [], "status": "running",
    }
    report_path = output_root / f"replot_run_figure_{args.figure}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        for workflow in workflows:
            command = [sys.executable, str(REPO_ROOT / workflow.script), *workflow.arguments,
                       workflow.output_option, str(output_root / workflow.output_subdir)]
            print("\n> " + " ".join(command), flush=True)
            started_ns = time.time_ns()
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
            paths = [output_root / relative for relative in workflow.outputs]
            missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
            if missing:
                raise RuntimeError(f"Workflow {workflow.key} did not generate expected PDFs: {missing}")
            for path in paths:
                with path.open("rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise RuntimeError(f"Output is not a PDF: {path}")
                if path.stat().st_mtime_ns < started_ns - 1_000_000_000:
                    raise RuntimeError(f"Output was not refreshed by this run: {path}")
            report["workflows"].append({
                "key": workflow.key, "panels": list(workflow.panels), "command": command,
                "code": {relative: file_sha256(REPO_ROOT / relative) for relative in workflow.code_files},
                "outputs": {relative: file_sha256(output_root / relative) for relative in workflow.outputs},
                "sources": {relative: file_sha256(REPO_ROOT / relative) for relative in workflow.source_files},
                "external_inputs": {relative: file_sha256(data_root / relative) for relative in workflow.data_files},
            })
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    count = sum(len(workflow.outputs) for workflow in workflows)
    print(f"\nCompleted {count} supported panel PDFs from saved results. No models were trained.")
    print(f"Outputs: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

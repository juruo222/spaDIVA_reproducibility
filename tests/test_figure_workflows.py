"""Standard-library tests for the saved-result workflow contracts.

Run from the repository root with:
    python -B -m unittest discover -s tests -p 'test_figure_workflows.py' -v
These standard-library tests validate workflow metadata and paths independently
of plotting and training dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.figure_workflows import (  # noqa: E402
    WORKFLOWS,
    required_files,
    resolve_data_root,
    selected_workflows,
)


class WorkflowContractTests(unittest.TestCase):
    def test_six_workflows_and_31_unique_panels_and_outputs(self):
        self.assertEqual(len(WORKFLOWS), 6)
        self.assertEqual(len({workflow.key for workflow in WORKFLOWS}), 6)
        panels = [panel for workflow in WORKFLOWS for panel in workflow.panels]
        outputs = [output for workflow in WORKFLOWS for output in workflow.outputs]
        self.assertEqual(len(panels), 31)
        self.assertEqual(len(set(panels)), 31)
        self.assertEqual(len(outputs), 31)
        self.assertEqual(len(set(outputs)), 31)
        for workflow in WORKFLOWS:
            self.assertEqual(len(workflow.panels), len(workflow.outputs), workflow.key)

    def test_figure_selection_counts(self):
        for figure, workflows, panels in (("2", 1, 5), ("3", 2, 10), ("4", 2, 9), ("5", 1, 7)):
            with self.subTest(figure=figure):
                selection = selected_workflows(figure)
                self.assertEqual(len(selection), workflows)
                self.assertEqual(sum(len(workflow.panels) for workflow in selection), panels)
                self.assertTrue(all(workflow.figure == figure for workflow in selection))
        self.assertEqual(selected_workflows(), WORKFLOWS)
        with self.assertRaises(ValueError):
            selected_workflows("1")

    def test_defaults_never_request_training(self):
        for workflow in WORKFLOWS:
            with self.subTest(workflow=workflow.key):
                self.assertNotIn("full", workflow.arguments)
                self.assertFalse(any("retrain" in argument for argument in workflow.arguments))
                if "--mode" in workflow.arguments:
                    index = workflow.arguments.index("--mode")
                    self.assertIn(workflow.arguments[index + 1], ("plot", "compact"))
                self.assertIn(workflow.script, workflow.code_files)

    def test_only_actual_repository_assets_are_required(self):
        for workflow in WORKFLOWS:
            for relative in (*workflow.code_files, *workflow.source_files):
                with self.subTest(path=relative):
                    path = REPO_ROOT / relative
                    self.assertTrue(path.is_file(), relative)
                    self.assertGreater(path.stat().st_size, 0, relative)
            self.assertFalse(any("expected/metrics.json" in path for path in workflow.source_files))
        p22 = next(workflow for workflow in WORKFLOWS if workflow.key == "p22_domains")
        self.assertTrue(any(path.endswith("figure3b_spadiva_cluster_annotations.csv") for path in p22.source_files))

    def test_zenodo_input_counts(self):
        for figure, expected in (("2", 1), ("3", 0), ("4", 3), ("5", 0)):
            observed = {path for workflow in selected_workflows(figure) for path in workflow.data_files}
            self.assertEqual(len(observed), expected, figure)

    def test_paths_are_relative_and_contained(self):
        for workflow in WORKFLOWS:
            for relative in (*workflow.code_files, *workflow.source_files, *workflow.data_files, *workflow.outputs):
                with self.subTest(path=relative):
                    self.assertFalse(Path(relative).is_absolute())
                    self.assertNotIn("..", Path(relative).parts)
            self.assertIn(workflow.output_option, ("--output-dir", "--results-root", "--results-dir"))
            self.assertIn(workflow.output_subdir, ("", "Figure_3", "Figure_4", "Figure_5"))
            self.assertTrue(all(path.endswith(".pdf") for path in workflow.outputs))

    def test_repository_paths_match_case_sensitive_filesystems(self):
        paths = {p for workflow in WORKFLOWS for p in (*workflow.code_files, *workflow.source_files)}
        for relative in paths:
            current = REPO_ROOT
            for part in Path(relative).parts:
                with self.subTest(path=relative, component=part):
                    self.assertIn(part, {child.name for child in current.iterdir()})
                current = current / part

    def test_home_relative_data_root(self):
        expected = (Path.home() / "spaDIVA data").resolve()
        with patch.dict(os.environ, {"SPADIVA_DATA_ROOT": "~/spaDIVA data"}):
            self.assertEqual(resolve_data_root(REPO_ROOT), expected)
            self.assertEqual(resolve_data_root(REPO_ROOT, Path("~/spaDIVA data")), expected)

    def test_explicit_data_root_overrides_environment_even_when_missing(self):
        explicit = REPO_ROOT / "nonexistent-explicit-input"
        with patch.dict(os.environ, {"SPADIVA_DATA_ROOT": str(REPO_ROOT / "other-input")}):
            self.assertEqual(resolve_data_root(REPO_ROOT, explicit), explicit.resolve())

    def test_environment_overrides_default_directories(self):
        selected = REPO_ROOT / "nonexistent-environment-input"
        with patch.dict(os.environ, {"SPADIVA_DATA_ROOT": str(selected)}):
            with patch.object(Path, "is_dir", return_value=True):
                self.assertEqual(resolve_data_root(REPO_ROOT), selected.resolve())

    def test_standard_data_is_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_data_root(REPO_ROOT), REPO_ROOT / "data")

    def test_missing_defaults_report_standard_data_root(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "is_dir", return_value=False):
                self.assertEqual(resolve_data_root(REPO_ROOT), REPO_ROOT / "data")

    def test_required_files_deduplicates_and_uses_selected_roots(self):
        workflow = selected_workflows("2")[0]
        data_root = REPO_ROOT / "example-data-root"
        files = required_files((workflow, workflow), REPO_ROOT, data_root)
        self.assertEqual(len(files), len(set(files)))
        self.assertEqual(len(files), len(workflow.code_files) + len(workflow.source_files) + len(workflow.data_files))
        self.assertIn(data_root / workflow.data_files[0], files)
        self.assertIn(REPO_ROOT / workflow.script, files)
        self.assertEqual(required_files((), REPO_ROOT, None), [])


if __name__ == "__main__":
    unittest.main()

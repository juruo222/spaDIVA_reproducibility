"""Safety tests for saved-result preflight and the default entry point."""

import contextlib
import csv
from dataclasses import replace
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import check_reproducibility_setup as setup
from scripts.figure_workflows import selected_workflows

spec = importlib.util.spec_from_file_location("replot_entry", ROOT / "reproduce.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ReplotPreflightTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / "results" / ".test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp.name)
        (self.root / "plot.py").write_text("# test fixture\n", encoding="utf-8")
        source = self.root / "values.csv"
        source.write_text("value\n1\n", encoding="utf-8")
        with (self.root / "MANIFEST.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["relative_path", "sha256"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"relative_path": "values.csv", "sha256": setup.file_sha256(source)})
        self.workflow = replace(selected_workflows("3")[0], script="plot.py", code_files=("plot.py",),
                                source_files=("values.csv",), data_files=())

    def tearDown(self):
        self.temp.cleanup()

    def check(self, workflow=None, data_root=None):
        output = io.StringIO()
        with patch.object(setup, "check_packages", return_value=[]) as packages, contextlib.redirect_stdout(output):
            result = setup.check_replot_setup(self.root, (workflow or self.workflow,), data_root)
        packages.assert_called_once_with(setup.REPLOT_IMPORTS, False)
        return result, output.getvalue()

    def test_replot_does_not_require_unrelated_raw_data(self):
        self.assertEqual(self.check(data_root=self.root / "does-not-exist")[0], 0)

    def test_modified_source_is_rejected(self):
        (self.root / "values.csv").write_text("value\n2\n", encoding="utf-8")
        result, output = self.check()
        self.assertNotEqual(result, 0)
        self.assertIn("differs from MANIFEST.tsv", output)

    def test_missing_plot_input_is_reported(self):
        workflow = replace(self.workflow, data_files=("datasets/lymph/annotation.csv",))
        result, output = self.check(workflow, self.root)
        self.assertNotEqual(result, 0)
        self.assertIn("annotation.csv", output)

    def test_missing_plot_code_is_reported(self):
        (self.root / "plot.py").unlink()
        self.assertNotEqual(self.check()[0], 0)

    def test_missing_manifest_is_rejected(self):
        (self.root / "MANIFEST.tsv").unlink()
        self.assertNotEqual(self.check()[0], 0)

    def test_no_model_dependency_in_replot_checks(self):
        self.assertFalse({"torch", "scvi", "rpy2", "hnswlib", "spaDIVA"} & set(setup.REPLOT_IMPORTS))


class EntryTests(unittest.TestCase):
    def test_list_never_runs_or_checks_packages(self):
        with patch.object(sys, "argv", ["reproduce.py", "--list"]), \
             patch.object(entry, "check_replot_setup") as preflight, \
             patch.object(entry.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(), 0)
        preflight.assert_not_called()
        run.assert_not_called()

    def test_preflight_failure_never_starts_plotters(self):
        with patch.object(sys, "argv", ["reproduce.py", "--figure", "3"]), \
             patch.object(entry, "check_replot_setup", return_value=1), \
             patch.object(entry.subprocess, "run") as run:
            self.assertEqual(entry.main(), 1)
        run.assert_not_called()

    def test_check_only_never_starts_plotters(self):
        with patch.object(sys, "argv", ["reproduce.py", "--check-only"]), \
             patch.object(entry, "check_replot_setup", return_value=0), \
             patch.object(entry.subprocess, "run") as run:
            self.assertEqual(entry.main(), 0)
        run.assert_not_called()

    def test_sources_cannot_be_used_as_output(self):
        for path in (ROOT, ROOT / "datasets" / "simulation", ROOT / "spaDIVA"):
            with self.assertRaises(ValueError):
                entry.resolve_output(path)


if __name__ == "__main__":
    unittest.main()

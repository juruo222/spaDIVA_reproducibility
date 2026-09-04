"""Full-mode GSEA selection and plotting; frozen reference checks stay strict."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "datasets/P22_ATAC_RNA/external_translation_figure_reproduction"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, WORKFLOW / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


full = _load_module("_p22_gsea_full_test", "_full_pipeline.py")
plotting = _load_module("_p22_gsea_plotting_test", "_plotting.py")


def _results(nes, fdr):
    return pd.DataFrame({
        "Term": [f"library__pathway_{index}" for index in range(len(nes))],
        "NES": nes,
        "FDR q-val": fdr,
    })


class GseaSelectionTests(unittest.TestCase):
    def test_fewer_than_ten_significant_pathways(self):
        displayed = full._select_displayed_gsea(
            _results([1.4, 2.0, -1.6, -2.0], [0.02, 0.01, 0.1, 0.3])
        )
        self.assertEqual(displayed["NES"].tolist(), [2.0, 1.4, -1.6])
        self.assertEqual(displayed["plot_order_top_to_bottom"].tolist(), [1, 2, 3])
        self.assertTrue((displayed["FDR_q_value"] < 0.25).all())

    def test_at_most_five_per_direction_selected_by_fdr(self):
        displayed = full._select_displayed_gsea(
            _results(list(range(1, 8)) + list(range(-1, -8, -1)),
                     [0.01 * index for index in range(1, 8)] * 2)
        )
        self.assertEqual(displayed["NES"].tolist(), [5, 4, 3, 2, 1, -1, -2, -3, -4, -5])

    def test_no_significant_pathways_returns_empty_schema(self):
        displayed = full._select_displayed_gsea(_results([1.1, -1.2], [0.25, 0.9]))
        self.assertTrue(displayed.empty)
        self.assertIn("plot_order_top_to_bottom", displayed)
        self.assertIn("FDR_q_value", displayed)

    def test_one_direction_is_allowed(self):
        displayed = full._select_displayed_gsea(_results([-2.0, -1.0], [0.1, 0.2]))
        self.assertEqual(displayed["NES"].tolist(), [-1.0, -2.0])

    def test_invalid_statistics_are_not_displayed(self):
        displayed = full._select_displayed_gsea(
            _results([np.inf, 1.0, -1.0, 0.0, 2.0], [0.1, np.inf, -0.1, 0.1, 0.2])
        )
        self.assertEqual(displayed["NES"].tolist(), [2.0])

    def test_full_table_is_saved_when_no_pathways_are_significant(self):
        result = _results([1.1, -1.2], [0.25, 0.9])
        fake_gp = SimpleNamespace(prerank=lambda **kwargs: SimpleNamespace(res2d=result))
        ranking = pd.DataFrame({"gene": ["A", "B"], "unique_factor_ratio": [0.1, 0.2]})
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(sys.modules, {"gseapy": fake_gp}), \
             patch.object(full, "_read_local_gmts", return_value={"set": ["A", "B"]}):
            output = Path(directory) / "gsea_all_results.csv"
            displayed = full._run_gsea(ranking, (), output)
            self.assertTrue(displayed.empty)
            pd.testing.assert_frame_equal(pd.read_csv(output), result)


class GseaPlottingTests(unittest.TestCase):
    def test_full_validation_accepts_variable_and_empty_tables(self):
        for result in (_results([1.0, -1.0], [0.1, 0.2]), _results([1.0], [0.3])):
            with self.subTest(result=result):
                displayed = full._select_displayed_gsea(result)
                validated = plotting._validate_gsea_sources(displayed, strict_reference=False)
                self.assertEqual(len(validated), len(displayed))

    def test_full_validation_rejects_non_significant_and_non_finite_rows(self):
        displayed = full._select_displayed_gsea(_results([1.0], [0.1]))
        for column, value in (("FDR_q_value", 0.25), ("FDR_q_value", np.nan),
                              ("FDR_q_value", -0.1), ("NES", np.inf), ("NES", 0)):
            with self.subTest(column=column, value=value):
                invalid = displayed.copy()
                invalid.loc[0, column] = value
                with self.assertRaises(ValueError):
                    plotting._validate_gsea_sources(invalid, strict_reference=False)

    def test_frozen_reference_pathways_remain_required(self):
        source = pd.read_csv(WORKFLOW / "source_data/figure3g_gsea_displayed.csv")
        validated = plotting._validate_gsea_sources(source, strict_reference=True)
        self.assertEqual(tuple(validated["pathway"]), plotting.EXPECTED_GSEA_PATHWAYS)
        for changed in (source.iloc[:3], source.iloc[:0]):
            with self.assertRaises(AssertionError):
                plotting._validate_gsea_sources(changed, strict_reference=True)

    def test_empty_plot_explains_no_significant_result(self):
        displayed = full._select_displayed_gsea(_results([1.0], [0.3]))
        with patch.object(plotting, "_save_pdf") as save:
            plotting.plot_panel_g(displayed, Path("unused.pdf"))
        figure = save.call_args.args[0]
        try:
            self.assertIn("No pathways with FDR < 0.25", [text.get_text() for text in figure.axes[0].texts])
            self.assertEqual(len(figure.axes[0].patches), 0)
        finally:
            plotting.plt.close(figure)

    def test_short_plot_has_one_bar_per_significant_pathway(self):
        displayed = full._select_displayed_gsea(_results([1.0, -1.0], [0.1, 0.2]))
        with patch.object(plotting, "_save_pdf") as save:
            plotting.plot_panel_g(displayed, Path("unused.pdf"))
        figure = save.call_args.args[0]
        try:
            self.assertEqual(len(figure.axes[0].patches), 2)
        finally:
            plotting.plt.close(figure)


if __name__ == "__main__":
    unittest.main()

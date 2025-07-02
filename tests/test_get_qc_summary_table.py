import unittest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from quality_control import add_qc_metrics
from quality_control import get_qc_summary_table

class TestGetQCSummaryTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    def create_test_adata(self):
        X = np.array([
            [1, 0, 3, 0],
            [0, 2, 0, 4],
            [5, 0, 0, 6]
        ])
        var_names = ["MT-CO1", "MT-CO2", "GeneA", "GeneB"]
        obs_names = ["cell1", "cell2", "cell3"]
        adata = AnnData(X=X)
        adata.var_names = var_names
        adata.obs_names = obs_names
        add_qc_metrics(adata)
        return adata

    def test_qc_summary_table_basic(self):
        adata = self.create_test_adata()
        get_qc_summary_table(adata)
        self.assertIn("qc_summary_table", adata.uns)
        summary = adata.uns["qc_summary_table"]
        self.assertTrue(isinstance(summary, pd.DataFrame))
        self.assertIn("mean", summary.columns)
        self.assertIn("median", summary.columns)
        self.assertIn("upper_mad", summary.columns)
        self.assertIn("lower_mad", summary.columns)
        self.assertIn("upper_quantile", summary.columns)
        self.assertIn("lower_quantile", summary.columns)
        self.assertIn("Sample", summary.columns)
        self.assertEqual(set(summary["metric_name"]), {"nFeature", "nCount", "percent.mt"})
        self.assertEqual(summary["Sample"].iloc[0], "All")

    def test_qc_summary_table_non_numeric(self):
        adata = self.create_test_adata()
        adata.obs["non_numeric"] = ["a", "b", "c"]
        with self.assertRaises(TypeError):
            get_qc_summary_table(adata, stat_columns_list=["nFeature", "non_numeric"]) 

    def test_qc_summary_table_statistics(self):
        adata = self.create_test_adata()
        get_qc_summary_table(adata)
        summary = adata.uns["qc_summary_table"]
        # Check mean, median, quantiles for nFeature
        nfeature_row = summary[summary["metric_name"] == "nFeature"].iloc[0]
        # nFeature is [2, 2, 2]
        self.assertEqual(nfeature_row["mean"], 2)
        self.assertEqual(nfeature_row["median"], 2)
        self.assertEqual(nfeature_row["upper_mad"], 2)
        self.assertEqual(nfeature_row["lower_mad"], 2)
        self.assertEqual(nfeature_row["upper_quantile"], 2)
        self.assertEqual(nfeature_row["lower_quantile"], 2)
        # Check mean, median, quantiles for nCount
        ncount_row = summary[summary["metric_name"] == "nCount"].iloc[0]
        expected_mean = np.mean([4, 6, 11])
        expected_median = np.median([4, 6, 11])
        expected_upper = np.percentile([4, 6, 11], 95)
        expected_lower = np.percentile([4, 6, 11], 5)
        self.assertAlmostEqual(ncount_row["mean"], expected_mean)
        self.assertAlmostEqual(ncount_row["median"], expected_median)
        self.assertAlmostEqual(ncount_row["upper_quantile"], expected_upper)
        self.assertAlmostEqual(ncount_row["lower_quantile"], expected_lower)

if __name__ == "__main__":
    unittest.main()

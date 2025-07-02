import scanpy as sc
import pandas as pd
import os 
import numpy as np
from scipy.sparse import issparse, csr_matrix
from quality_control import add_qc_metrics
from scipy.stats import median_abs_deviation
from anndata import AnnData

os.chdir("/Users/bombina2/github/Reg_Ax/ST_workflow")

#adata = sc.read_h5ad("/Users/bombina2/github/Reg_Ax/CCBR/output/vizium_test.h5ad")
adata = sc.read_h5ad("/Users/bombina2/github/Reg_Ax/CCBR/output/example/lung_atlas.h5ad")

def get_qc_summary_table(
    adata: AnnData,
    n_mad: int = 5,
    upper_quantile: float = 0.95,
    lower_quantile: float = 0.05,
    stat_columns_list: list = ["nFeature", "nCount", "percent.mt"],
    sample_column: str = None
) -> None:
    """
    Compute summary statistics for quality control metrics in an AnnData object 
    and store the result in adata.uns['qc_summary_table'].

    Parameters:
        adata (AnnData): The AnnData object containing the data.
        n_mad (int): Number of MADs to use for upper/lower thresholds.
        upper_quantile (float): Upper quantile to compute (e.g., 0.95).
        lower_quantile (float): Lower quantile to compute (e.g., 0.05).
        stat_columns_list (list): List of column names to compute statistics for.
        sample_column (str, optional): Column name to group by sample. 
        If None, computes for all data.

    Returns:
        None. The summary table is stored in adata.uns['qc_summary_table'].
    """
    def compute_stats(df):
        stat_vals = []
        for col_name in stat_columns_list:
            if col_name in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    raise TypeError(f"Column '{col_name}' must be numeric to compute statistics.")
                median = df[col_name].median()
                mad = median_abs_deviation(df[col_name], nan_policy='omit')
                col_stats = [
                    col_name,
                    df[col_name].mean(),
                    median,
                    median + n_mad * mad,
                    median - n_mad * mad,
                    df[col_name].quantile(upper_quantile),
                    df[col_name].quantile(lower_quantile)
                ]
                stat_vals.append(col_stats)
            else:
                raise ValueError(f"Column '{col_name}' not found in adata.obs")
        return pd.DataFrame(
            stat_vals,
            columns=[
                "metric_name", "mean", "median", 
                "upper_mad", "lower_mad", 
                "upper_quantile", "lower_quantile"
            ]
        )

    obs_df = adata.obs
    summary_table = pd.DataFrame()
    if sample_column is None:
        stat_df = compute_stats(obs_df)
        stat_df["Sample"] = "All"
        summary_table = stat_df
    else:
        samples_list = pd.unique(obs_df[sample_column])
        for current_sample in samples_list:
            sample_df = obs_df[obs_df[sample_column] == current_sample]
            stat_df = compute_stats(sample_df)
            stat_df["Sample"] = current_sample
            summary_table = pd.concat([summary_table, stat_df])
    summary_table = summary_table.reset_index(drop=True)
    adata.uns["qc_summary_table"] = summary_table


add_qc_metrics(adata, organism="hs")

get_qc_summary_table(adata, sample_column=None)



sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_in_top_50_genes"],
    jitter=0.4,
    multi_panel=True,
)
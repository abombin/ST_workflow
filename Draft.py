import scanpy as sc
import pandas as pd
import os 
import numpy as np
from scipy.sparse import issparse, csr_matrix

os.chdir("/Users/bombina2/github/Reg_Ax/ST_workflow")

adata = sc.read_h5ad("/Users/bombina2/github/Reg_Ax/CCBR/output/vizium_test.h5ad")
atlas_adata = sc.read_h5ad("/Users/bombina2/github/Reg_Ax/CCBR/output/example/lung_atlas.h5ad")

def get_qc_summary_table(
    adata, 
    n_mad=5, 
    upper_quantile=0.95, 
    lower_quantile=0.05, 
    stat_columns_list=["nFeature", "nCount", "percent.mt"],
    sample_column=None
):
    def compute_stats(df):
        stat_vals = []
        for col_name in stat_columns_list:
            if col_name in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    raise TypeError(f"Column '{col_name}' must be numeric to compute statistics.")
                median = df[col_name].median()
                mad = df[col_name].mad()
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
    return summary_table


sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_in_top_50_genes"],
    jitter=0.4,
    multi_panel=True,
)
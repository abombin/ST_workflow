import scanpy as sc
import pandas as pd
import os 
import numpy as np
from scipy.sparse import issparse, csr_matrix
from quality_control import add_qc_metrics
from scipy.stats import median_abs_deviation
from anndata import AnnData
from quality_control import get_qc_summary_table
import matplotlib.pyplot as plt

os.chdir("/Users/bombina2/github/ST_workflow")

#adata = sc.read_h5ad("/Users/bombina2/github/Reg_Ax/CCBR/output/vizium_test.h5ad")
adata = sc.read_h5ad("/Users/bombina2/github/Reg_Ax/CCBR/output/example/lung_atlas.h5ad")


add_qc_metrics(adata, organism="hs")

get_qc_summary_table(adata, sample_column=None)

list(adata.obs)

def plot_qc_metrics(
    adata, 
    metric_name=["nCount", "nFeature", "percent.mt"], 
    group_column=None, 
    log=False, 
    size=1, 
    layer=None,
    rotation=None
):
    """
    Generate violin plots for quality control (QC) metrics from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix (from Scanpy).
    metric_name : list of str, optional
        List of metric names to plot (default: ["nCount", "nFeature", "percent.mt"]).
    group_column : str or None, optional
        Column name in adata.obs to group the data by (default: None).
    log : bool, optional
        Whether to log-transform the data (default: False).
    size : float, optional
        Size of the points in the violin plot (default: 1).
    layer : str or None, optional
        Layer in adata to use for plotting (default: None).
    rotation :  float or None (default: None) Rotation of xtick labels.

    Returns
    -------
    dict
        If group_column is None, returns a dictionary with keys 'figure' and 'axes' for the whole dataset.
        If group_column is provided, returns a dictionary mapping each group to its own {'figure', 'axes'} dict for the subsetted AnnData.
    """
    if group_column is not None:
        results = {}
        for group in adata.obs[group_column].unique():
            adata_subset = adata[adata.obs[group_column] == group]
            violin_plot = sc.pl.violin(
                adata_subset,
                metric_name,
                size=size,
                groupby=None, 
                log=log,
                layer=layer,
                rotation=rotation,
                jitter=0.4,
                multi_panel=True,
                show=False,
                use_raw=False,
            )
            results[group] = {
                "figure": violin_plot.figure,
                "axes": violin_plot.axes
            }
        return results
    else:
        violin_plot = sc.pl.violin(
            adata,
            metric_name,
            size=size,
            groupby=None,
            log=log,
            layer=layer,
            rotation=rotation,
            jitter=0.4,
            multi_panel=True,
            show=False,
            use_raw=False,
        )
        return {
            "figure": violin_plot.figure,
            "axes": violin_plot.axes
        }

plots_res = plot_qc_metrics(adata, group_column="batch", metric_name=["nCount", "nFeature"])

plots_res['A1']['figure'].show()  # Display the plot

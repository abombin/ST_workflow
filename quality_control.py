import scanpy as sc
import pandas as pd
import os 
import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.stats import median_abs_deviation
from anndata import AnnData

# add QC metrics to AnnData object
def add_qc_metrics(adata, 
                   organism="hs", 
                   mt_match_pattern=None, 
                   layer=None):
    """
    Adds quality control (QC) metrics to the AnnData object.

    Parameters:
    -----------
    adata : AnnData
        The AnnData object containing single-cell or spatial 
        transcriptomics data.
    organism : str, optional
        The organism type. Default is "hs" (human). Use "mm" for mouse.
        Determines the mitochondrial gene prefix 
        ("MT-" for human, "mt-" for mouse).
    mt_match_pattern : str, optional
        A custom pattern to identify mitochondrial genes. If None, it defaults
        to "MT-" for human or "mt-" for mouse based on the `organism` parameter.
        Takes precedence over the default patterns.
        If provided, it should match the prefix of mitochondrial gene names in 
        `adata.var_names`.
    layer : str, optional
        The name of the layer in `adata.layers` to use for calculations. 
        If None, the default `adata.X` matrix is used.

    Modifies:
    ---------
    adata.obs : pandas.DataFrame
        Adds the following QC metrics as new columns:
        - "nFeature": Number of genes with non-zero expression for each cell.
        - "nCount": Total counts (sum of all gene expression values) 
            for each cell.
        - "nCount_mt": Total counts for mitochondrial genes for each cell.
        - "percent.mt": Percentage of counts in mitochondrial genes 
            for each cell.

    Raises:
    -------
    ValueError
        If the specified `layer` is not found in `adata.layers`.

    Notes:
    ------
    - If the input matrix (`adata.X` or the specified layer) is dense, 
        it is converted to a sparse matrix for efficient computation.
    - Mitochondrial genes are identified based on the `mt_match_pattern`.

    Example:
    --------
    >>> add_qc_metrics(adata, organism="hs")
    >>> print(adata.obs[["nFeature", "nCount", "nCount_mt", "percent.mt"]])
    """
    # identify mitochondrial genes pattern
    if mt_match_pattern is None:
        if organism == "hs":
            mt_match_pattern = "MT-"
        elif organism == "mm":
            mt_match_pattern = "mt-"
        else:
            raise ValueError(f"Unsupported organism '{organism}'. Supported values are 'hs' and 'mm'.")

    if layer is None:
        test_matrix = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        test_matrix = adata.layers[layer]
    
    # Check if adata.X is sparse, and convert if necessary
    if not issparse(test_matrix):
        test_matrix = csr_matrix(test_matrix)

    # Calculate total number of genes with values > 0 for each cell
    adata.obs["nFeature"] = np.array((test_matrix > 0).sum(axis=1)).flatten()
    # Calculate the sum of counts for all genes for each cell
    adata.obs["nCount"] = np.array(test_matrix.sum(axis=1)).flatten()
    # Identify mitochondrial genes based on the match pattern
    mt_genes = adata.var_names.str.startswith(mt_match_pattern)
    # Calculate the sum of counts for mitochondrial genes for each cell
    adata.obs["nCount_mt"] = np.array(test_matrix[:, mt_genes]
                                      .sum(axis=1)).flatten()
    # Calculate the percentage of counts in mitochondrial genes for each cell
    adata.obs["percent.mt"] = (adata.obs["nCount_mt"] / 
                               adata.obs["nCount"]) * 100
    # Handle NaN values in percent.mt
    adata.obs["percent.mt"] = adata.obs["percent.mt"].fillna(0)
    # Ensure percent.mt is stored as a float
    adata.obs["percent.mt"] = adata.obs["percent.mt"].astype(float)

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
            # Check if the column exists in the DataFrame
            if col_name in df.columns:
                # Ensure the column is numeric
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    raise TypeError(f"Column '{col_name}' must be numeric to compute statistics.")
                # Compute median and MAD (median absolute deviation)
                median = df[col_name].median()
                mad = median_abs_deviation(df[col_name], nan_policy='omit')
                # Collect statistics for this column
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
                # Raise error if column is missing
                raise ValueError(f"Column '{col_name}' not found in adata.obs")
        # Return DataFrame with statistics for all columns
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
    # If no sample_column, compute stats for all data
    if sample_column is None:
        stat_df = compute_stats(obs_df)
        stat_df["Sample"] = "All"
        summary_table = stat_df
    else:
        # Otherwise, compute stats for each sample group
        samples_list = pd.unique(obs_df[sample_column])
        for current_sample in samples_list:
            sample_df = obs_df[obs_df[sample_column] == current_sample]
            stat_df = compute_stats(sample_df)
            stat_df["Sample"] = current_sample
            summary_table = pd.concat([summary_table, stat_df])
    # Reset index and store in adata.uns
    summary_table = summary_table.reset_index(drop=True)
    adata.uns["qc_summary_table"] = summary_table

# make violin plots for QC metrics

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
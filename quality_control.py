import scanpy as sc
import pandas as pd
import os 
import numpy as np
from scipy.sparse import issparse, csr_matrix

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


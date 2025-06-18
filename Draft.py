import scanpy as sc
import pandas as pd
import os 
import numpy as np
from scipy.sparse import issparse, csr_matrix

os.chdir("/Users/bombina2/github/Reg_Ax/ST_workflow")

adata = sc.read_h5ad("/Users/bombina2/github/Reg_Ax/CCBR/output/vizium_test.h5ad")

def add_qc_metrics_0(adata, organism="hs", mt_match_pattern=None, layer=None,
                   log1p=False):
    # identify mitochondrial genes pattern
    if mt_match_pattern is None:
        if organism == "hs":
            mt_match_pattern = "MT-"
        elif organism == "mm":
            mt_match_pattern = "mt-"
        else:
            raise ValueError("Unknown organism")
    adata.var["mt"] = adata.var_names.str.startswith(mt_match_pattern)
    # calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, 
        log1p=False, percent_top=None, layer=layer
    )



    
    


sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_in_top_50_genes"],
    jitter=0.4,
    multi_panel=True,
)
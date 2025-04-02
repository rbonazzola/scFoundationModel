import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import os

def select_highly_variable_genes(h5ad_path, top_n=2000, output_csv=None):
    """ 
    Select the top N highly variable genes using Scanpy's method (adjusted dispersion).
    
    Args:
        h5ad_path (str): Path to the input H5AD file.
        top_n (int): Number of genes to select.
        output_csv (str or None): Path to the output CSV file. If None, a default will be generated.
    
    Returns:
        DataFrame with selected genes ordered by normalized dispersion.
    """
    # Auto-generate output name if not provided
    if output_csv is None:
        base = os.path.splitext(os.path.basename(h5ad_path))[0]
        output_csv = f"HVG_{base}.csv"

    adata = sc.read_h5ad(h5ad_path)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=top_n)
    variable_genes = adata.var[adata.var['highly_variable']]

    genes_df = pd.DataFrame({
        "gene_name": variable_genes.index,
        "position": np.arange(len(variable_genes)),
        "variability": variable_genes["variances_norm"]
    })  

    genes_df = genes_df.sort_values("variability", ascending=False)
    genes_df.to_csv(output_csv, index=False)
    
    print(f"Saved {top_n} highly variable genes to {output_csv}")
    return genes_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select highly variable genes from an H5AD file.")
    parser.add_argument("h5ad_path", help="Path to the input .h5ad file")
    parser.add_argument("--top_n", type=int, default=2000, help="Number of genes to select (default: 2000)")
    parser.add_argument("--output_csv", default=None, help="Path to output CSV file (default: auto-generated)")

    args = parser.parse_args()

    select_highly_variable_genes(
        h5ad_path=args.h5ad_path,
        top_n=args.top_n,
        output_csv=args.output_csv
    )
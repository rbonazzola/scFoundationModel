import scanpy as sc
import pandas as pd
import numpy as np

def select_highly_variable_genes(h5ad_path, top_n=2000, output_csv="highly_variable_genes.csv"):
    """
    Selecciona los N genes más variables utilizando el método de Scanpy (dispersion ajustada).
    
    Args:
        h5ad_path (str): path to a H5AD file.
        top_n (int): number of genes to select
        output_csv (str): path to output file
    
    Returns:
        DataFrame with selected genes ordered by dispersion measure
    """

    adata = sc.read_h5ad(h5ad_path)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=top_n)
    variable_genes = adata.var[adata.var['highly_variable']]

    print(variable_genes.head())
    genes_df = pd.DataFrame({
        "gene_name": variable_genes.index,
        "position": np.arange(len(variable_genes)),
        "variability": variable_genes["variances_norm"]
    })

    genes_df = genes_df.sort_values("variability", ascending=False)
    genes_df.to_csv(output_csv, index=False)
    
    print(f"Saved {top_n} highly variable genes to {output_csv}")
    return genes_df

# Ejemplo de uso
h5ad_file = "./transforms/CRA004476_transformed.h5ad"
select_highly_variable_genes(h5ad_file, top_n=20000, output_csv="highly_variable_genes.csv")

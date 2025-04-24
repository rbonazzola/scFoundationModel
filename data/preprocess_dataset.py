import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import argparse
import scanpy as sc
import anndata as ad

DATASET_METADATA_SPREADSHEET = "./scRNA-Seq_datasets.csv"


def select_datasets(regex="[Rr]oot"):

    df = pd.read_csv(DATASET_METADATA_SPREADSHEET)
    regex = re.compile(regex)
    root_datasets = df.loc[df.tissue.apply(lambda x: bool(regex.match(str(x))))]["dataset ID"].to_list()
    return [f"transforms/HVG_{dataset}.csv" for dataset in root_datasets]


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


def subset_h5ad_by_gene_list(h5ad_path, gene_list, output_folder):
    adata = sc.read_h5ad(h5ad_path)

    # Filter only available genes and respect the order that was given
    genes_in_adata = set(adata.var_names)
    genes_to_keep = [gene for gene in gene_list if gene in genes_in_adata]

    adata_subset = adata[:, genes_to_keep].copy()

    base_name = os.path.splitext(os.path.basename(h5ad_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_subset.h5ad")
    adata_subset.write(output_path)
    print(f"✅ Saved subset with {len(genes_to_keep)} genes to {output_path}")


def compute_consensus_hvg(csv_paths, top_n=1000):
    """
    Computes the consensus list of HVGs from multiple CSV files.
    
    Args:
        csv_paths (list of str): paths to CSVs containing HVGs.
        top_n (int): number of most frequent genes to return.
        
    Returns:
        list of str: top N genes by frequency.
    """
    
    counter = Counter()

    for path in csv_paths:
        try:
            df = pd.read_csv(path).head(4000)
            print(f"processing {path}...")
        except FileNotFoundError:
            print(f"{path} does not exist")
            continue
        genes = df.iloc[:, 0].tolist()  # assume first column contains gene names
        counter.update(genes)

    most_common_genes = [gene for gene, _ in counter.most_common(top_n)]
    return most_common_genes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute consensus HVGs from per-sample CSVs.")
    
    # parser.add_argument("csvs", nargs="+", help="Paths to HVG CSVs")
    parser.add_argument("h5ad_path", help="Path to the input .h5ad file")
    parser.add_argument("--input_folder", default="transforms")
    parser.add_argument("--top_n", type=int, default=4000, help="Number of consensus genes to return")
    parser.add_argument("--output", default="consensus_hvg.csv", help="Output CSV for consensus gene list")

    args = parser.parse_args()

    select_highly_variable_genes(
        h5ad_path=args.h5ad_path,
        top_n=args.top_n,
        output_csv=args.output_csv
    )

    # args.csvs = [ f"{args.input_folder}/{x}" for x in args.csvs ]
    # print(args.csvs)

    csvs = select_datasets()
    
    consensus_genes = compute_consensus_hvg(csvs, args.top_n)
    pd.Series(consensus_genes, name="gene_name").to_csv(args.output, index=False)
    print(f"✅ Consensus HVG list saved to {args.output}")

import scanpy as sc
import pandas as pd
import argparse
import os


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subset .h5ad files by a gene list in specified order.")
    parser.add_argument("gene_list_csv", help="CSV file with genes (one per line or in column 'gene_name')")
    parser.add_argument("h5ad_files", nargs="+", help="Input .h5ad files to process")
    parser.add_argument("--output_folder", default=".", help="Folder to save the output files")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Leer lista de genes (respeta orden)
    genes_df = pd.read_csv(args.gene_list_csv)
    gene_list = genes_df.iloc[:, 0].tolist()

    for h5ad_path in args.h5ad_files:
        subset_h5ad_by_gene_list(h5ad_path, gene_list, args.output_folder)

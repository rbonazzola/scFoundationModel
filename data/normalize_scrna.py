import scanpy as sc
import argparse
import os

def normalize_and_log(input_path, output_path=None, output_folder="."):
    """Normalize and log-transform an AnnData object."""
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Auto-generate output_path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}_transformed.h5ad")

    print(f"📂 Reading: {input_path}")
    adata = sc.read_h5ad(input_path)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata, base=2)

    print(f"💾 Writing: {output_path}")
    adata.write(output_path)
    print("✅ Done.")



def select_highly_variable_genes(h5ad_path, top_n=20000, output_csv=None):
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

    parser = argparse.ArgumentParser(description="Normalize and log-transform an H5AD file.")
    parser.add_argument("input_path", help="Path to the input .h5ad file")
    parser.add_argument("--output_path", default=None, help="Full path for output .h5ad file (optional)")
    parser.add_argument("--output_folder", default=".", help="Folder where the output file will be saved (default: current directory)")

    args = parser.parse_args()
    normalize_and_log(args.input_path, args.output_path, args.output_folder)

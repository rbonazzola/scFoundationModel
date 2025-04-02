import pandas as pd
from collections import Counter
import argparse
import os
import re
import pandas as pd


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


def select_datasets(regex="[Rr]oot"):

    df = pd.read_csv("~/Descargas/scRNA-Seq_datasets.csv")
    regex = re.compile(regex)
    root_datasets = df.loc[df.tissue.apply(lambda x: bool(regex.match(str(x))))]["dataset ID"].to_list()
    return [f"transforms/HVG_{dataset}.csv" for dataset in root_datasets]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute consensus HVGs from per-sample CSVs.")
    # parser.add_argument("csvs", nargs="+", help="Paths to HVG CSVs")
    parser.add_argument("--input_folder", default="transforms")
    parser.add_argument("--top_n", type=int, default=4000, help="Number of consensus genes to return")
    parser.add_argument("--output", default="consensus_hvg.csv", help="Output CSV for consensus gene list")

    args = parser.parse_args()

    # args.csvs = [ f"{args.input_folder}/{x}" for x in args.csvs ]
    # print(args.csvs)
    csvs = select_datasets()
    print(csvs)
    consensus_genes = compute_consensus_hvg(csvs, args.top_n)
    pd.Series(consensus_genes, name="gene_name").to_csv(args.output, index=False)
    print(f"✅ Consensus HVG list saved to {args.output}")

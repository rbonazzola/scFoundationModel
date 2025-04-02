import scanpy as sc
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random
import numpy as np

N_CLASSES = 7

class MultiScRNADataset(Dataset):
    def __init__(self, folder_path, gene_csv=None, top_n_genes=None):
        super().__init__()

        self.adata_list = []
        self.lengths = []
        self.cumulative_lengths = []

        # Load all .h5ad files in the folder
        h5ad_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5ad')])
        for filename in h5ad_files:
            path = os.path.join(folder_path, filename)
            adata = sc.read_h5ad(path)

            # Subset genes if needed
            if gene_csv is not None and top_n_genes is not None:
                selected_genes = pd.read_csv(gene_csv).sort_values("variability", ascending=False).head(top_n_genes)
                selected_positions = selected_genes['position'].values
                adata = adata[:, selected_positions]

            self.adata_list.append(adata)
            self.lengths.append(adata.shape[0])

        # Compute cumulative lengths for index mapping
        self.cumulative_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        
        file_idx = np.searchsorted(self.cumulative_lengths, index, side='right') - 1
        local_idx = index - self.cumulative_lengths[file_idx]
    
        adata = self.adata_list[file_idx]
        x = adata.X[local_idx]
        data = x.toarray()[0] if hasattr(x, "toarray") else x.A1 if hasattr(x, "A1") else x
    
        data[data > (N_CLASSES - 2)] = N_CLASSES - 2
        seq = torch.from_numpy(data).long()
        seq = torch.cat((seq, torch.tensor([0])))  # <EOS>
        return seq

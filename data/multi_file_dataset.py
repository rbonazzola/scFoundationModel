import scanpy as sc
import torch
from torch.utils.data import Dataset
import os
import numpy as np

N_CLASSES = 7

class MultiScRNADataset(Dataset):
    def __init__(self, folder_path, top_n_genes=None):
        super().__init__()

        self.tensor_list = []
        self.lengths = []
        self.N_CLASSES = N_CLASSES

        h5ad_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5ad')])
        self.file_paths = [os.path.join(folder_path, f) for f in h5ad_files]

        # Find common genes
        gene_sets = []
        for path in self.file_paths:
            adata = sc.read_h5ad(path, backed='r')
            gene_sets.append(set(adata.var_names))
            adata.file.close()
        common_genes = set.intersection(*gene_sets)

        # Use order from first file
        first_adata = sc.read_h5ad(self.file_paths[0], backed='r')
        self.selected_genes = [g for g in first_adata.var_names if g in common_genes][:top_n_genes]
        first_adata.file.close()

        for path in self.file_paths:
            adata = sc.read_h5ad(path)
            adata = adata[:, self.selected_genes]

            # Extract matrix and convert
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            elif hasattr(X, "A"):
                X = X.A
            else:
                X = np.asarray(X)

            # Clip and cast to smallest possible int type
            X = np.clip(X, a_min=0, a_max=N_CLASSES - 2).astype(np.int64)  # values 0-5 (or 6)

            # Add <EOS> token at the end of each sequence
            eos_column = np.zeros((X.shape[0], 1), dtype=np.int64)
            X = np.concatenate([X, eos_column], axis=1)

            tensor_data = torch.from_numpy(X)  # dtype=torch.uint8
            self.tensor_list.append(tensor_data)
            self.lengths.append(tensor_data.shape[0])

            del adata, X  # explicit cleanup

        self.cumulative_lengths = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.cumulative_lengths, index, side='right') - 1
        local_idx = index - self.cumulative_lengths[file_idx]
        return self.tensor_list[file_idx][local_idx].long()

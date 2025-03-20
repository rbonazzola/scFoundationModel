import scanpy as sc
from torch.utils.data import Dataset
import random
import torch

N_CLASSES = 7

class scRNADataset(Dataset):
    def __init__(self, data, gene_csv=None, top_n_genes=None):
        super().__init__()       
        self.data = data
        self.top_n_genes = top_n_genes
        
        if gene_csv is not None and top_n_genes is not None:
            selected_genes = pd.read_csv(gene_csv).sort_values("variability", ascending=False).head(self.top_n_genes)
            selected_positions = selected_genes['position'].values
            self.data = data[:, selected_positions]
        else:
            self.data = data


        self.N_CLASSES = N_CLASSES

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (N_CLASSES - 2)] = N_CLASSES - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))) # .to(device)
        return full_seq

    def __len__(self):
        return self.data.shape[0]

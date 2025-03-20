import scanpy as sc
from torch.utils.data import Dataset
import random
import torch

N_CLASSES = 7

class scRNADataset(Dataset):
    def __init__(self, data):
        super().__init__()       
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

import os
import gc
import argparse
import json
import math

import random
from functools import reduce

import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq

    def __len__(self):
        return self.data.shape[0]
    

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--batch_size", type=int, default=8, help='Number of batch size.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default=f"{os.getenv('HOME')}/data/scrna_seq_arabidopsis/scPlantDB/CRA004476.h5ad.gz", 
                    help='Path of data for pretraining.')
args = parser.parse_args()


CLASS = args.bin_num + 2
BATCH_SIZE = args.batch_size
POS_EMBED = args.pos_embed
SEED = 42

if args.data_path.endswith('.h5ad.gz'):
    import gzip
    with gzip.open(args.data_path, "rb") as f:
        data = ad.read_h5ad(f).X
elif args.data_path.endswith('.h5ad'):
    data = sc.read_h5ad(args.data_path).X

print("Data loaded")

data_train, data_val = train_test_split(data, test_size=0.05,random_state=SEED)
print(data_train.shape, data_val.shape)

train_dataset = SCDataset(data_train)

val_dataset = SCDataset(data_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

from IPython import embed; embed()
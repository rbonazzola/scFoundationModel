import os
import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
input_dir = "/home/home01/scrb/nobackup/data/scrna/subsetted"
output_dirs = {
    "train": os.path.join(input_dir, "train"),
    "val":   os.path.join(input_dir, "val"),
    "test":  os.path.join(input_dir, "test"),
}

for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

files = sorted([f for f in os.listdir(input_dir) if f.endswith(".h5ad")])

for file in files:
    path = os.path.join(input_dir, file)
    adata = sc.read_h5ad(path)

    indices = np.arange(adata.n_obs)
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42)

    # Subset
    adata_train = adata[train_idx].copy()
    adata_val = adata[val_idx].copy()
    adata_test = adata[test_idx].copy()

    # Base name (without .h5ad)
    base = os.path.splitext(file)[0]

    adata_train.write(os.path.join(output_dirs["train"], f"{base}_train.h5ad"))
    adata_val.write(os.path.join(output_dirs["val"], f"{base}_val.h5ad"))
    adata_test.write(os.path.join(output_dirs["test"], f"{base}_test.h5ad"))

    print(f"✔ Split done for {file}")

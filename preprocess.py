import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
from tqdm import tqdm

ref_data = sc.read_h5ad('./data/transforms/CRA004476.h5ad')
data = sc.read_h5ad('./data/transforms/CRA004476.h5ad')

counts = sparse.lil_matrix((data.X.shape[0], ref_data.X.shape[1]),dtype=np.float32)
ref = ref_data.var_names.tolist()

obj = data.var_names.tolist()
obj_set = set(obj)

for i in tqdm(range(len(ref))):
    # if ref[i] in obj_set:    
        loc = obj.index(ref[i])
        counts[:,i] = data.X[:,loc]

counts = counts.tocsr()
new = ad.AnnData(X=counts)
new.var_names = ref
new.obs_names = data.obs_names
new.obs = data.obs
new.uns = ref_data.uns

sc.pp.filter_cells(new, min_genes=200)
sc.pp.normalize_total(new, target_sum=1e4)
sc.pp.log1p(new, base=2)
new.write('./data/CRA004476_transformed.h5ad')
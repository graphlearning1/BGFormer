import numpy as np 
import h5py
import scipy as sp
import scanpy as sc
import pandas as pd
import torch
import torch.utils.data as Data
import os
from scipy import sparse


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)

def dict_from_group(group):
    d = dotdict()
    for key in group:
        if isinstance(group[key],h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = group[key][...]
            if value.dtype.type is np.bytes_:
                value = decode(value)
            if value.size == 1:
                value = value.flat[0]
        d[key]=value 
    return d 


def load_dataset(data_path):
    adata = sc.read_h5ad(data_path)
    if 'highly_variable' not in adata.var_keys():
        adata = normalize(adata, copy=True, highly_genes=1500, size_factors=True, normalize_input=True, logtrans_input=True)
    sf = np.array(adata.obs['size_factors'])
    highly_variable = adata.var['highly_variable']
    raw = adata.raw[:, highly_variable].X
    if isinstance(raw, sparse.csr_matrix) or isinstance(raw, sparse.csc_matrix):
        raw = raw.toarray()
    adata = adata[:, highly_variable]
    sc.pp.scale(adata, max_value=10)
    x = adata.X
    if isinstance(x, sparse.csr_matrix) or isinstance(x, sparse.csc_matrix):
        x = x.toarray()
    x = torch.from_numpy(x).to(torch.float)
    sf = torch.from_numpy(sf).to(torch.float)
    if data_path.split('/')[-2] == 'Allages':
        raw = raw.astype(np.float32)
    raw = torch.from_numpy(raw).to(torch.float)

    return x, sf, raw, adata


def load_h5(data_dir):
    #adata = sc.read_h5ad(cfg['data_dir'])
    with h5py.File(data_dir, "r") as f:
        dict_from_group(f['obs'])
        dict_from_group(f['var'])
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        exprs_handle = f["exprs"]
        if isinstance(exprs_handle, h5py.Group):
             mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
        else:
            mat = exprs_handle[...].astype(np.float32)

        if isinstance(mat, np.ndarray):
            X = np.array(mat)
        else:
            X = np.array(mat.toarray())
        cell_name = np.array(obs["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        return X, cell_label, cell_type


def load_h5ad(data_dir):
    adata = sc.read_h5ad(data_dir)
    return adata


def load_data(data_path):
    x = np.load(data_path+'/x.npy')
    sf = np.load(data_path+'/sf.npy')
    raw = np.load(data_path+'/raw.npy')
    adata = sc.read_h5ad(data_path+'/adata.h5ad')
    return x, sf, raw, adata


def preprocess(adata):
    if isinstance(adata.X, sparse.csr_matrix) or isinstance(adata.X, sparse.csc_matrix):
        adata.X = adata.X.toarray()
    raw = adata.X.copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
    sc.pp.log1p(adata)
    adata_ = adata.copy()
    if adata.shape[1] < 5000:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    else:
        sc.pp.highly_variable_genes(adata)
    hvg_index = adata.var["highly_variable"].values
    raw = raw[:, hvg_index]
    adata = adata[:, hvg_index]

    sc.pp.scale(adata, max_value=10)
    x = adata.X

    return x, sf, raw, adata_


def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    # raw = adata.X.copy()
    # sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
    
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


class MyDataset(Data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.Y = labels

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):

        return self.features[idx], self.Y[idx], idx
    

class IndexDataset(Data.Dataset):
    def __init__(self, index_mapping):
        self.index_mapping = index_mapping
        self.keys = list(index_mapping.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.index_mapping[self.keys[idx]]
    
if __name__ == "__main__":
    mapping = {i: np.random.permutation(10) for i in range(10)}
    mapping[0] = np.array([0, 1, 2, 3])
    print(mapping)
    data = IndexDataset(mapping)
    dataset = Data.TensorDataset(torch.tensor(list(mapping.keys())))
    loader = Data.DataLoader(dataset, batch_size=2, shuffle=True)
    for i, index in enumerate(loader):
        print(index[0])
        print(data[list(index[0])])
        print('-'*10)

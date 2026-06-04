import numpy as np
import h5py
import scipy as sp
import scanpy as sc
import pandas as pd


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
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = group[key][...]
            if value.dtype.type is np.bytes_:
                value = decode(value)
            if value.size == 1:
                value = value.flat[0]
        d[key] = value
    return d


def load_dataset(data_path):
    X, Y = load_data(data_path)
    X = np.ceil(X).astype(np.float64)
    cluster = int(max(Y) - min(Y) + 1)
    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    adata = normalize(adata, copy=True, highly_genes=1500, size_factors=True, normalize_input=True, logtrans_input=True)
    count = adata.X
    return count, Y


def load_data(data_path):
    # adata = sc.read_h5ad(cfg['data_dir'])
    with h5py.File(data_path, "r") as f:
        dict_from_group(f['obs'])
        dict_from_group(f['var'])
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        exprs_handle = f["exprs"]
        if isinstance(exprs_handle, h5py.Group):
            mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                        exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
        else:
            mat = exprs_handle[...].astype(np.float32)

        if isinstance(mat, np.ndarray):
            X = np.array(mat)
        else:
            X = np.array(mat.toarray())
        cell_name = np.array(obs["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        # print(f"{type(f['obs'])},{type(f['var'])}")
        return X, cell_label


def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True, size_factors=True, normalize_input=True,
              logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
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
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

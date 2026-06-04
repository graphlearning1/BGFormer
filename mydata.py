

from config import get_config, parse_args
import numpy as np

import scanpy as sc
import pandas as pd
from scipy import sparse




if __name__ == '__main__':
    args = parse_args()
    # args.device = 0
    args.dataset = 'HRCA'
    # args.device = 2
    device = args.device
    cfg_path = args.cfg_path + f'/{args.dataset}.yml'
    cfg = get_config(cfg_path)
    seed = args.seed



    for key, value in vars(args).items():
        cfg[key] = value



    data_path = cfg['data_dir']
    data_path = "/home/user/data_zyx/datasets/single-cell/Ratmap-scp/raw/ratmap_scp.h5ad"
    adata = sc.read_h5ad(data_path)
    sf = np.array(adata.obs['size_factors'])
    highly_variable = adata.var['highly_variable']
    raw = adata.raw[:, highly_variable].X
    if isinstance(raw, sparse.csr_matrix) or isinstance(raw, sparse.csc_matrix):
        raw = raw.toarray()
    adata = adata[:, highly_variable]
    sc.pp.scale(adata, max_value=10)
    x = adata.X


    # adata.X: (n_cells, n_genes)
    df = pd.DataFrame(
        adata.X,
        index=adata.obs_names,  # cell IDs
        columns=adata.var_names  # gene names
    )
    df.insert(0, 'cell', df.index)
    # 导出为 csv.gz
    df.to_csv("{}.csv.gz".format(args.dataset), compression="gzip")

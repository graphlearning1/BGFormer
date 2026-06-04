dataset_name: 'MRCA'
data_dir: '/root/data/lzm/single-cell/MRCA/adata_preprocessed.h5ad'
type_key: 'cell_type'
num_classes: 27

lr: 0.0001
seed: 3
num_vq: 32
n_heads: 1

batch_size: 10000
fine_batch_size: 16
pretraining_epoch: 30
fine_tune_epoch: 200
hops: 15
k: 4 
pe_dim: 3


pred_dim: 128
hidden_dim: 512
k_dim: 512
v_dim: 512
ffn_dim: 512
emb_dim: 512
n_layers: 1
n_buckets: 12
n_hashes: 3

prob_feature: 0.1
prob_edge: 0.5
tau: 0.8
alpha: 0.1
beta: 0.5
dropout_rate: 0.1
attention_dropout_rate: 0.1
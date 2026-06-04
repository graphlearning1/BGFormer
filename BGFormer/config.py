import yaml
from easydict import EasyDict
import argparse
import warnings

def get_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()
    # Copy
    for k, v in config.items():
        cfg[k] = v
    return cfg


def parse_args():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kv_lr", type=float, default=0.0001)
    parser.add_argument("--kl_ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--start_seed", type=int, default=2)
    parser.add_argument("--end_seed", type=int, default=5)
    # parser.add_argument("--num_vq", type=int, nargs="+", default=[1024, 64])
    parser.add_argument("--num_vq", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--save_model_path", type=str, default='./saved_models')
    parser.add_argument("--save_embedding_path", type=str, default='./save')
    parser.add_argument("--cfg_path", type=str, default='./configs')
    parser.add_argument("--dataset", type=str, default='Bach')  #'HRCA', "MRCA", 'Ratmap", "Allages", 'Chen'
    parser.add_argument("--data_path", type=str, default='/root/data/lzm/single-cell/')
    parser.add_argument("--device", type=int, default=2, help='Device cuda id')
    parser.add_argument("--wandb", action='store_true', default=False, help="Use wandb or not")
    parser.add_argument("--pretrain", action='store_true', default=False, help="Use wandb or not")
    parser.add_argument("--finetune", action='store_true', default=False, help="Use wandb or not")
    parser.add_argument("--save", action='store_true', default=False, help="Save results or not")
    args = parser.parse_args()
    return args

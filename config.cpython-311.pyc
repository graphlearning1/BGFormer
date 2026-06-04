import os
from datetime import datetime
from os.path import exists
from easydict import EasyDict
import psutil
# from scanpy import AnnData
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import gc
from config import get_config, parse_args
from data import normalize, load_dataset, MyDataset
from scRNA_workflow import *
from my_models_codebook_multi import BGFormer
from utils import setup_seed, evaluate, kmeans
import utils
import time
import faiss
import numpy as np
import argparse
import warnings



def faiss_kmeans(x, kmeans, niter=20):
    x_np = x.detach().cpu().numpy().astype('float32')
    faiss.normalize_L2(x_np)

    kmeans.train(x_np)
    _, I = kmeans.index.search(x_np, 1)
    centers = torch.tensor(kmeans.centroids)
    y = I.squeeze()

    del kmeans.index
    del kmeans
    torch.cuda.empty_cache()

    return y, centers


def get_result(z, c):

    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - c) ** 2, dim=2) / 1)
    q = q ** ((1 + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, dim=1)).t()

    return q

def get_res_env(y, p):
    y_pred = torch.argmax(p, dim=-1)
    acc, f1, nmi, ari, homo, comp = evaluate(y, y_pred.cpu().numpy(), cfg.num_classes)
    print(f'ACC: {acc:.6f}, F1: {f1:.6f}, NMI: {nmi:.6f}, ARI: {ari:.6f}, Homo: {homo:.6f}, Comp: {comp:.6f}')

def pretraining(cfg, X, sf, raw, Y, model, model_kv, cluster_method, device):
    print(f'Pretraining Hashformer model for {cfg.pretraining_epoch} epochs... for dataset::::{cfg.dataset_name}')
    type_key = cfg['type_key']
    save_model_path = cfg.save_model_path


    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    min_loss = np.inf
    best_epoch = 0
    idx = torch.arange(0, X.shape[0]).to(device)
    train_dataset = Data.TensorDataset(X, sf, raw, Y, idx)
    train_loader = Data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    epoch = cfg.pretraining_epoch


    with torch.no_grad():
        z = []
        for i, (xbatch, sfbatch, rawbatch, y, _) in enumerate(train_loader):
            xbatch = xbatch.to(device)
            rep = model.encoder_x(xbatch)
            z.append(rep.detach().cpu())
        z = torch.cat(z, dim=0)
        z = z.detach().cpu().numpy().astype('float32')
    model.vq.init_codebook(z, 'Kmeans')

    with torch.no_grad():
        emb = []
        y_true = []
        k = model.vq.return_Q()
        for i, (xbatch, sfbatch, rawbatch, y, _) in enumerate(train_loader):
            xbatch = xbatch.to(device)
            z = model.embedding(xbatch, k, k)
            emb.append(z.detach().cpu())
            y_true.append(y)
        emb = torch.cat(emb, dim=0)
        y_true = torch.cat(y_true, dim=0)

    y_pred, centers = faiss_kmeans(emb, cluster_method)
    acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred, cfg.num_classes)
    print(f'ACC: {acc:.6f}, F1: {f1:.6f}, NMI: {nmi:.6f}, ARI: {ari:.6f}, Homo: {homo:.6f}, Comp: {comp:.6f}')

    dis = get_result(emb, centers)
    z_p_all = dis.detach()
    y_p = torch.argmax(dis, dim=-1)
    acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_p.numpy(), cfg.num_classes)
    model.set_centers(centers)
    print(f'ACC: {acc:.6f}, F1: {f1:.6f}, NMI: {nmi:.6f}, ARI: {ari:.6f}, Homo: {homo:.6f}, Comp: {comp:.6f}')

    time_var = []
    ####train model and clus
    best_acc = 0
    peak_all = 0
    for epoch_id in range(epoch):

        model.train()
        train_loss = 0


        for i, (xbatch, sfbatch, rawbatch, y, idx_i) in enumerate(train_loader):
            xbatch = xbatch.to(device)
            sfbatch = sfbatch.to(device)
            rawbatch = rawbatch.to(device)

            x_rep = model.encoder_x(xbatch)
            hidden_q, loss_c = model.vq(x_rep, return_assignment=False)


            K = model.vq.return_Q()

            if not cfg.beta == 0:
                meanbatch, dispbatch, pibatch = model.decoder_z(hidden_q)
                loss_rec_q = model.loss_rec(sfbatch, rawbatch, meanbatch, dispbatch, pibatch)

            emb_inchange, meanbatch_inchange, dispbatch_inchange, pibatch_inchange = model(xbatch, K, K)
            loss_rec = model.loss(emb_inchange, sfbatch, rawbatch, meanbatch_inchange, dispbatch_inchange, pibatch_inchange)

            if not cfg.beta == 0:
                loss = 1*loss_rec + (loss_rec_q + loss_c)*cfg.beta
            else:
                loss = loss_rec + (loss_rec_q + loss_c)*cfg.beta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach().item()


        with torch.no_grad():
            y_pred = []
            y_true = []
            z_p_all = []
            for i, (xbatch, sfbatch, rawbatch, y, _) in enumerate(train_loader):
                xbatch = xbatch.to(device)
                # z_inchange, z_change, mask = model.masker(xbatch, K, V, returnmask=True)
                K = model.vq.return_Q()
                emb = model.embedding(xbatch, K, K)
                dis = get_result(emb, model.centers)
                z_p = dis.detach()
                y_p = torch.argmax(z_p, dim=-1)
                z_p_all.append(z_p)
                y_true.append(y)
                y_pred.append(y_p)
                del emb, dis, z_p
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)


        acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred.cpu().numpy(), cfg.num_classes)
        print(f'ACC: {acc}, F1: {f1}, NMI: {nmi}, ARI: {ari}, Homo: {homo}, Comp: {comp}')

        if best_acc < acc:
            path = os.path.join('model', args.dataset)
            if not os.path.exists(path):
                os.makedirs(path)

            best_acc = acc
            best_epoch = epoch_id
            path_save = os.path.join('model_vis', args.dataset, "model_multi_{}.pth".format(args.dataset))
            torch.save(model.state_dict(), path_save)


    path_save = os.path.join('model_vis', args.dataset, "model_multi_{}.pth".format(args.dataset))
    state_dict = torch.load(path_save)
    model.load_state_dict(state_dict)

    print(time_var)
    print("time mean: ", torch.tensor(time_var).mean().item(), torch.tensor(time_var).var().item())
    print(f"Overall peak GPU memory usage: {peak_all:.2f} MB")

    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []
        z_p_all = []
        for i, (xbatch, sfbatch, rawbatch, y, _) in enumerate(train_loader):
            xbatch = xbatch.to(device)
            K = model.vq.return_Q()
            emb = model.embedding(xbatch, K, K)
            dis = get_result(emb, model.centers)
            z_p = dis.detach()
            y_p = torch.argmax(z_p, dim=-1)
            z_p_all.append(z_p)
            y_true.append(y)
            y_pred.append(y_p)
            del emb, dis, z_p
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred.cpu().numpy(), cfg.num_classes)
        print(f"best at epoch: {best_epoch}")
        print(f'best ACC: {acc}, F1: {f1}, NMI: {nmi}, ARI: {ari}, Homo: {homo}, Comp: {comp}')

    return acc, f1, nmi, ari, homo, comp


if __name__ == '__main__':
    args = parse_args()
    device = args.device
    cfg_path = args.cfg_path + f'/{args.dataset}.yml'
    cfg = get_config(cfg_path)
    seed = cfg.seed





    for key, value in vars(args).items():
        cfg[key] = value
    if cfg.wandb:
        if not os.path.exists("./wandb/"):
            os.makedirs("./wandb")
        wandb.init(config=cfg,
                   project="SGFormer",
                   name="scHashFormer_{}".format(cfg.dataset_name),
                   dir="./wandb/",
                   job_type="training",
                   reinit=True)

    X, sf, raw, adata = load_dataset(cfg['data_dir'])
    print(X.shape)
    type_key = cfg.type_key
    cell_name = np.array(adata.obs[type_key])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    adata.obs['Group'] = cell_label
    Y = torch.from_numpy(np.array(adata.obs['Group'])).to(torch.long)

    input_dim = X.shape[1]

    res = []
    setup_seed(seed)

    print(cfg)

    model = BGFormer(cfg, input_dim).to(device)


    pretrain_dataset = Data.TensorDataset(X, sf, raw, Y)
    pretrain_loader = Data.DataLoader(pretrain_dataset, batch_size=cfg.batch_size, shuffle=True)


    cluster_method = faiss.Kmeans(cfg.hidden_dim, cfg.num_classes, niter=20, gpu=True)

    result = pretraining(cfg, X, sf, raw, Y, model, None, cluster_method, device)
    res.append(result)
    print("*****train end********")
    print(torch.tensor(res))
    res = torch.tensor(res).mean(0)
    acc, f1, nmi, ari, homo, comp = tuple(res)
    acc, f1, nmi, ari, homo, comp = acc.item(), f1.item(), nmi.item(), ari.item(), homo.item(), comp.item()
    print(f'mean ACC: {acc:.6f}, F1: {f1:.6f}, NMI: {nmi:.6f}, ARI: {ari:.6f}, Homo: {homo:.6f}, Comp: {comp:.6f}')
    print(args)
    print('\n')
    print('\n')
    print('\n')
    print('\n')


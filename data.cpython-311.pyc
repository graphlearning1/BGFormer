from einops import rearrange
import numpy as np
import torch
import math
from torch import nn
from torch.nn import functional as F
import copy
from torch_geometric.nn.conv import GCNConv
from utils import dis_fun


class Quantizer(nn.Module):
    def __init__(self, entry_num, entry_dim):
        super().__init__()

        self.entry_num = entry_num
        self.entry_dim = entry_dim
        self.decay = 0.9
        self.entry = nn.Embedding(self.entry_num, self.entry_dim)
        self.register_buffer("entry_prob", torch.zeros(self.entry_num))

    def init_codebook(self, z, method):
        if method == "Random":
            self.entry.weight.data.uniform_(-1.0 / self.entry_num, 1.0 / self.entry_num)
        elif method == "Kmeans":
            import faiss

            d = z.shape[1]
            kmeans = faiss.Kmeans(d, self.entry_num, spherical=True, gpu=True)
            kmeans.train(z)
            D, I = kmeans.index.search(z, 1)
            assignments = I.reshape(-1)
            centers = np.zeros((self.entry_num, d))
            for i in range(self.entry_num):
                centers[i] = z[assignments == i].mean(axis=0)
            self.entry.weight.data.copy_(torch.from_numpy(centers))
        elif method == "Geometric":
            from geosketch import gs

            sketch_index = gs(z, self.entry_num, replace=False)
            self.entry.weight.data.copy_(torch.from_numpy(z[sketch_index]))

    def return_Q(self):
        return self.entry.weight.data

    def forward(self, e, return_assignment):
        # cosine similarity
        normed_e = F.normalize(e, dim=1).detach()
        normed_c = F.normalize(self.entry.weight, dim=1)
        sim = torch.einsum("bd,dn->bn", normed_e, rearrange(normed_c, "n d -> d n"))

        # entry assignment
        assignment_indices = torch.argmax(sim, dim=1)
        assignments = torch.zeros(
            assignment_indices.unsqueeze(1).shape[0], self.entry_num, device=e.device
        )
        assignments.scatter_(1, assignment_indices.unsqueeze(1), 1)
        avg_probs = torch.mean(assignments, dim=0)

        # quantize
        e_q = torch.matmul(assignments, self.entry.weight)
        # L_C
        # loss = torch.mean((e_q - e.detach()) ** 2)
        loss = torch.mean((e_q - e) ** 2)

        if self.training:
            # update the entry usage
            self.entry_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)

            # deal with small entries
            norm_distance = F.softmax(1 - sim, dim=1)
            norm_distance = torch.max(norm_distance, dim=1).values
            dis_indices = torch.multinomial(
                norm_distance, num_samples=self.entry_num, replacement=True
            ).view(-1)
            random_feat = e.detach()[dis_indices]
            beta_s = (
                torch.exp(-self.entry_prob * self.entry_num * 100 - 1e-3)
                .unsqueeze(1)
                .repeat(1, self.entry_dim)
            )
            self.entry.weight.data = (
                self.entry.weight.data * (1 - beta_s) + random_feat * beta_s
            )

            # deal with large entries
            if self.entry_prob.sum() + 1e-4 >= 1:
                sim_t = sim.t()
                median_distance = torch.median(sim_t, dim=1).values
                median_distance = torch.abs(sim_t - median_distance[:, None])
                dis_indices = torch.multinomial(
                    F.softmax(-median_distance, dim=1), num_samples=1
                ).view(-1)
                random_feat = e.detach()[dis_indices]
                beta_l = (
                    torch.exp(-self.entry_prob.mean() / self.entry_prob * 10 - 1e-3)
                    .unsqueeze(1)
                    .repeat(1, self.entry_dim)
                )
                self.entry.weight.data = (
                    self.entry.weight.data * (1 - beta_l) + random_feat * beta_l
                )

        if return_assignment:
            top2_sim = torch.topk(sim, 2, dim=1).values
            delta_conf = top2_sim[:, 0] - top2_sim[:, 1]
            loss_c_sum = torch.sum((e_q - e.detach()) ** 2)

            return assignment_indices, delta_conf, loss_c_sum
        else:
            return e_q, loss


class Encoder(nn.Module):
    def __init__(self, input_dim, entry_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, entry_dim),
        )

    def forward(self, input):
        return self.encoder(input)


def buildNetwork(layers, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)



class BGFormer(nn.Module):
    def __init__(self, cfg, input_dim, pe_dim=0):
        super(BGFormer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = cfg.hidden_dim
        self.ffn_dim = cfg.ffn_dim
        self.prob_feature = cfg.prob_feature
        self.prob_edge = cfg.prob_edge
        self.tau = cfg.tau
        self.hops = cfg.hops
        dropout_rate = cfg.dropout_rate
        attention_dropout_rate = cfg.attention_dropout_rate
        self.alpha = cfg.alpha


        self.pro_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pro_v = nn.Linear(self.hidden_dim, self.hidden_dim)


        self.vq = Quantizer(cfg.num_vq, self.hidden_dim)
        self.encoder_x = Encoder(input_dim, self.hidden_dim)


        if not cfg.beta == 0:
            self.decoder_q = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self._dec_mean_q = nn.Sequential(nn.Linear(self.hidden_dim*2, input_dim), MeanAct())
            self._dec_disp_q = nn.Sequential(nn.Linear(self.hidden_dim*2, input_dim), DispAct())
            self._dec_pi_q = nn.Sequential(nn.Linear(self.hidden_dim*2, input_dim), nn.Sigmoid())

        # MLP layer for input features

        self.encoder = TransformerModel(input_dim, self.hidden_dim, self.ffn_dim, cfg.hops, cfg.n_layers, cfg.num_vq,
                                            cfg.n_heads, dropout_rate, attention_dropout_rate)
        
        self.final_ln = nn.LayerNorm(self.hidden_dim)
        self.centers = nn.Parameter(torch.empty(cfg.num_classes, cfg.hidden_dim))


        self.decoder = nn.Linear(self.hidden_dim, self.hidden_dim*2)
        self._dec_mean = nn.Sequential(nn.Linear(self.hidden_dim*2, input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(self.hidden_dim*2, input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(self.hidden_dim*2, input_dim), nn.Sigmoid())

        self.criterion_kl = torch.nn.KLDivLoss(size_average=False)

        self.fc1 = nn.Linear(cfg.hidden_dim, 256)


    def forward(self, x, K, V):
        K = self.pro_k(K)
        V = self.pro_v(V)
        h = self.final_ln(self.encoder(x+torch.randn_like(x), K, V))
        h = self.decoder(h)  

        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)
        
        z = self.final_ln(self.encoder(x, K, V))
        return z, _mean, _disp, _pi


    def embedding(self, x, K, V):
        K = self.pro_k(K)
        V = self.pro_v(V)
        z = self.final_ln(self.encoder(x, K, V))

        return z

    def decoder_z(self, h):
        h = self.decoder_q(h)

        _mean = self._dec_mean_q(h)
        _disp = self._dec_disp_q(h)
        _pi = self._dec_pi_q(h)

        return _mean, _disp, _pi

    def set_centers(self, centers):
        if isinstance(centers, torch.Tensor):
            centers = centers.to(self.centers.device)
        elif isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers).to(self.centers.device)
        self.centers.data = centers.data

    def soft_assign(self, z):
        # norm_squared = torch.sum((z.unsqueeze(1) - self.centers) ** 2, 2)
        # numerator = 1.0 / (1.0 + (norm_squared / self.tau))
        # numerator = numerator ** 2
        # q = (numerator.t() / torch.sum(numerator, 1)).t()
        # weight = (q ** 2) / torch.sum(q, 0)
        # p = (weight.t() / torch.sum(weight, 1)).t()
        # return q, p.detach()

        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.centers)**2, dim=2) / 1)
        q = q**((1+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        p = q**2 / q.sum(0)
        return q, (p.t() / p.sum(1)).t().data
        
    def sim_center(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def class_center_loss(self, z: torch.Tensor):
        z = self.fc1(z)
        f = lambda x: torch.exp(x / 0.5)
        pos_same = f(self.sim_center(z, z))
        loss =  -torch.log(pos_same.diag() / (pos_same.sum(1) - pos_same.diag()))
        return loss.mean()

    def class_node_loss(self, z: torch.Tensor, p, center):
        z = self.fc1(z)
        center = self.fc1(center)
        P = p.detach()
        tau = 1
        # 稳定版 softmax
        # logits = p / tau
        # logits = logits - logits.max(dim=1, keepdim=True).values
        # P = torch.softmax(logits, dim=1)
        value, indices = P.max(1)
        raw = torch.arange(0, z.shape[0]).to(z.device)
        f = lambda x: torch.exp(x / 0.5)
        pos_same = f(self.sim_center(z, center))
        loss =  -torch.log(pos_same[raw, indices] / (pos_same.sum(1) - pos_same[raw, indices]))
        loss = loss * value
        return loss.mean()

    def loss_rec(self, sf, raw, mean, disp, pi):
        re_loss = negative_binomial_loss(raw, mean, disp, pi, sf)
        return re_loss

    def loss(self, emb, sf, raw, mean, disp, pi):

        # re_loss = F.mse_loss(x, x_hat)
        re_loss = negative_binomial_loss(raw, mean, disp, pi, sf)

        # loss_center = self.class_center_loss(center)
        # loss_node_center = self.class_node_loss(emb, z_p, center)
        # loss_clu = (loss_node_center + loss_center) / 2

        q, p = self.soft_assign(emb)
        kl_loss = torch.mean(torch.sum(p*torch.log(p/(q+1e-6)), dim=-1))
        loss_clu = kl_loss

        # a = 0.5
        # loss = a * re_loss + (1-a) * loss_clu
        loss = re_loss + self.alpha * loss_clu
        return loss

    def pretrain_loss(self, sf, raw, mean, disp, pi):

        # re_loss = F.mse_loss(x, x_hat)
        re_loss = negative_binomial_loss(raw, mean, disp, pi, sf)
        return re_loss

    def encodeBatch(self, x, batch_size=256, device='cuda'):
        encoded = []
        num = x.shape[0]
        num_batch = int(math.ceil(1.0*x.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = x[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            xbatch = xbatch.to(device)
            z, _, _, _ = self.forward(xbatch)
            encoded.append(z.data.detach().cpu())

        encoded = torch.cat(encoded, dim=0).to(device)
        return encoded

    def clustering(self, x, n_clusters=None):
        if n_clusters is None:
            q, _ = self.soft_assign(x)
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            centers = self.centers.data.cpu().numpy()
        else:
            if type(x) == torch.Tensor:
                x = x.cpu().detach().numpy()
            from utils import kmeans
            y_pred, centers = kmeans(x, n_clusters, centers=None)

        if type(y_pred) == torch.Tensor:
            y_pred = y_pred.cpu().detach().numpy()
        if type(centers) == torch.Tensor:
            centers = centers.cpu().detach().numpy()
        return y_pred, centers
    
    def kmeans_loss(self, z):
        dist1 = self.tau*torch.sum(torch.square(z.unsqueeze(1) - self.centers), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        q = torch.exp(-temp_dist1)
        q = (q.t() / torch.sum(q, dim=1)).t()
        q = torch.pow(q, 2)
        q = (q.t() / torch.sum(q, dim=1)).t()
        dist2 = dist1 * q
        return torch.mean(torch.sum(dist2, dim=1))




class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, ffn_dim=64, hops=3, n_layers=1, n_class=6, num_heads=8, dropout_rate=0.0, attention_dropout_rate=0.1):
        super().__init__()
        self.seq_len = hops + 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.is_projection = True
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.input_layer = nn.Linear(input_dim, hidden_dim)


        encoders = [
            ExtEncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads, n_class)
            for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.hidden_dim)
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)

    def forward_mask(self, batched_data, K, V):
        # transformer encoder
        batched_data = self.input_layer(batched_data)
        for enc_layer in self.layers:
            z1 = enc_layer(batched_data, K, V)  # hidden_dim -> emb_dim
        output = self.final_ln(z1)
        emb = self.Linear1(torch.relu(self.out_proj(output)))
        return F.normalize(emb)  # emb_dim

    def forward(self, batched_data, K, V):
        # transformer encoder
        batched_data = self.input_layer(batched_data)
        for enc_layer in self.layers:
            z1 = enc_layer(batched_data, K, V)  # hidden_dim -> emb_dim
        output = self.final_ln(z1)
        emb = self.Linear1(torch.relu(self.out_proj(output)))
        return F.normalize(emb+batched_data)  # emb_dim

    def projection(self, x):
        z1 = self.att_embeddings_nope(x)
        self.is_projection = False
        return z1




class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, n_class):
        super(EncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, W_Q, W_V, attn_bias=None):
        norm_x = self.self_attention_norm(x)  # hidden_size
        y = self.self_attention(norm_x, W_Q, W_V, attn_bias)  # hidden_size -> hidden_size
        y = self.self_attention_dropout(y)
        z1 = norm_x + y  # hidden_size
        y = self.ffn_norm(z1)
        y = self.ffn(y)  # hidden_size -> hidden_size
        y = self.ffn_dropout(y)
        z = z1 + y
        return z

class ExtEncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, n_class):
        super(ExtEncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = ExternalAttention_mutli(hidden_size, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, W_Q, W_V):
        norm_x = self.self_attention_norm(x)  # hidden_size
        y = self.self_attention(norm_x, W_Q, W_V)  # hidden_size -> hidden_size
        y = self.self_attention_dropout(y)
        z1 = norm_x + y  # hidden_size
        y = self.ffn_norm(z1)
        y = self.ffn(y)  # hidden_size -> hidden_size
        y = self.ffn_dropout(y)
        z = z1 + y
        return z



class ExternalAttention_mutli(nn.Module):
    def __init__(self, d_model, numhead, attention_dropout_rate=0.9):

        super().__init__()
        # self.mk = nn.Linear(d_model, S, bias=False)   # Q -> external memory keys
        # self.mv = nn.Linear(S, d_model, bias=False)   # memory values -> output
        self.softmax = nn.Softmax(dim=-1)
        # if numhead == 1:



        assert d_model % numhead == 0

        self.att_size = att_size = d_model // numhead
        self.num_heads = numhead

        self.linear_q = nn.Identity()
        self.linear_k = nn.Identity()
        self.linear_v = nn.Identity()

        # if numhead == 0:
        #
        # else:
        #     self.linear_q = nn.Linear(d_model, numhead * att_size)
        #     self.linear_k = nn.Linear(d_model, numhead * att_size)
        #     self.linear_v = nn.Linear(d_model, numhead * att_size)


    def forward(self, x, mk, mv, attn_return=False):
        """
        x: (B, N, C)
        output: (B, N, C)
        """
        d_k = self.att_size
        d_v = self.att_size

        x = self.linear_q(x).view(-1, self.num_heads, d_k)
        mk = self.linear_k(mk).view(-1, self.num_heads, d_v)
        mv = self.linear_v(mv).view(-1, self.num_heads, d_v)

        x = x.transpose(0, 1)  # [n_heads, b, d_k]
        mk = mk.transpose(0, 1)  # [n_heads, c, d_v]
        mv = mv.transpose(0, 1)  # [n_heads, c, d_k]   [n_heads, b, c]

        # x = x * self.scale

        # Step 1: compute attention weights (without Q·K^T)
        # B, N, S
        attn = torch.einsum('bij,bkj->bik', x, mk)

        # Step 2: normalize across tokens dim (N)
        # ensures sum(attn_i) = 1  -> prevents collapse
        attn = self.softmax(attn)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))

        # attn = self.att_dropout(attn)

        # Step 4: aggregate memory values
        out = torch.einsum('bij, bjk->bik', attn, mv)
        out = out.view(-1, self.num_heads * d_v)
        if attn_return:
            return out, attn
        else:
            return out

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        n = q.size(0)
        d_k = self.att_size
        d_v = self.att_size

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(-1, self.num_heads, d_k)  # [b, n_heads, d_k]
        k = self.linear_k(k).view(-1, self.num_heads, d_k)  # [c, n_heads, d_k]
        v = self.linear_v(v).view(-1, self.num_heads, d_v)  # [c, n_heads, d_v]

        q = q.transpose(0, 1)  # [n_heads, b, d_k]
        v = v.transpose(0, 1)  # [n_heads, c, d_v]
        k = k.transpose(0, 1).transpose(1, 2)  # [n_heads, d_k, c]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        prob = torch.matmul(q, k)  # [n_heads, b, c]

        # x = torch.softmax(x, dim=-1)
        prob = self.att_dropout(prob)
        x = prob.matmul(v)  # [n_heads, b, d_v]
        x = x.transpose(0, 1).contiguous()  # [b, n_heads, d_v]
        x = x.view(-1, self.num_heads * d_v)  # [b, dim]
        x = self.output_layer(x)

        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(hidden_size, ffn_size), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(ffn_size, hidden_size), nn.ReLU())
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def final_cl_loss(alpha1, alpha2, z, z_aug, adj, adj_aug, tau, hidden_norm=True):
    loss = alpha1 * cl_loss(z, z_aug, adj, tau, hidden_norm) + alpha2 * cl_loss(z_aug, z, adj_aug, tau, hidden_norm)
    return loss


def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def cl_loss(z, z_aug, adj, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss) / adj_count

    return -torch.mean(loss, 0)

def contrastive_loss(e, index_mapping, device, tau=0.5, hidden_norm=True):
    if hidden_norm:
        e = F.normalize(e, dim=-1)
    b = [torch.mean(e[index_mapping[key]], dim=0) for key in index_mapping.keys()]
    b = torch.stack(b, dim=0).to(device)
    f = lambda x: torch.exp(x / tau)
    con_loss = torch.Tensor([0]).to(device)
    for i, key in enumerate(index_mapping.keys()):
        pos_index = index_mapping[key]
        pos_e = e[pos_index].to(device)
        sim = f(pos_e.matmul(b.T))

        numerator = sim[:,i]

        denominator = sim.sum(1) + 1e-5

        loss = -torch.log(numerator/denominator)
        loss = loss.mean()
        con_loss = con_loss + loss
    return con_loss/len(index_mapping)


class Decoder(nn.Module):
    def __init__(self, entry_dim, output_dim):
        super().__init__()
        self.docoder = nn.Sequential(
            nn.Linear(entry_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.decoder_mean = nn.Linear(512, output_dim)
        self.decoder_disp = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.Softplus(),
        )

    def forward(self, input):
        decode = self.docoder(input)
        mean = torch.clamp(torch.exp(self.decoder_mean(decode)), 1e-5, 1e6)
        disp = torch.clamp(self.decoder_disp(decode), 1e-4, 1e4)
        return mean, disp
    

def negative_binomial_loss(x, mean, disp, pi, scale_factor=1.0):
    # eps = 1e-12
    # scale_factor = scale_factor[:, None]
    # mean = mean * scale_factor

    # t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    # t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
    #     x * (torch.log(disp + eps) - torch.log(mean + eps))
    # )
    # nb_final = t1 + t2
    # result = nb_final
    eps = 1e-10
    scale_factor = scale_factor[:, None]
    mean = mean * scale_factor
        
    t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
    t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0-pi+eps)
    zero_nb = torch.pow(disp/(disp+mean+eps), disp)
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
    result = torch.mean(result)
    return result
    
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
    

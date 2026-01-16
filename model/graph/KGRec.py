import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from data.kg_loader import KGDataLoader
from torch_geometric.utils import softmax as scatter_softmax, scatter
import math
import random


def scatter_sum(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')


def scatter_mean(src, index, dim=0, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


class Contrast(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.7):
        super(Contrast, self).__init__()
        self.tau: float = tau
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def self_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return (z1 * z2).sum(1)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.self_sim(z1, z2))
        rand_item = torch.randperm(z1.shape[0])
        neg_sim = f(self.self_sim(z1, z2[rand_item])) + f(self.self_sim(z2, z1[rand_item]))
        return -torch.log(between_sim / (between_sim + between_sim + neg_sim))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.mlp1(z1)
        h2 = self.mlp2(z2)
        loss = self.loss(h1, h2).mean()
        return loss


class AttnHGCN(nn.Module):
    def __init__(self, channel, n_hops, n_users, n_relations, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(AttnHGCN, self).__init__()
        self.no_attn_convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))
        self.relation_emb = nn.Parameter(relation_emb)
        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))
        self.n_heads = 2
        self.d_k = channel // self.n_heads
        nn.init.xavier_uniform_(self.W_Q)
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        relation_emb = relation_emb[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * relation_emb
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)
        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads * self.d_k)
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def forward(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, mess_dropout=True, item_attn=None):
        if item_attn is not None:
            item_attn = item_attn[inter_edge[1, :]]
            item_attn = scatter_softmax(item_attn, inter_edge[0, :])
            norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
            norm = torch.index_select(norm, 0, inter_edge[0, :])
            item_attn = item_attn * norm
            inter_edge_w = inter_edge_w * item_attn
        entity_res_emb = entity_emb
        user_res_emb = user_emb
        for _ in range(self.n_hops):
            entity_emb, user_emb = self.shared_layer_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, self.relation_emb)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        return entity_res_emb, user_res_emb

    def forward_ui(self, user_emb, item_emb, inter_edge, inter_edge_w, mess_dropout=True):
        item_res_emb = item_emb
        for _ in range(self.n_hops):
            user_emb, item_emb = self.ui_agg(user_emb, item_emb, inter_edge, inter_edge_w)
            if mess_dropout:
                item_emb = self.dropout(item_emb)
                user_emb = self.dropout(user_emb)
            item_emb = F.normalize(item_emb)
            user_emb = F.normalize(user_emb)
            item_res_emb = torch.add(item_res_emb, item_emb)
        return item_res_emb

    def forward_kg(self, entity_emb, edge_index, edge_type, mess_dropout=True):
        entity_res_emb = entity_emb
        for _ in range(self.n_hops):
            entity_emb = self.kg_agg(entity_emb, edge_index, edge_type)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
        return entity_res_emb

    def ui_agg(self, user_emb, item_emb, inter_edge, inter_edge_w):
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        return user_agg, item_agg

    def kg_agg(self, entity_emb, edge_index, edge_type):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_emb[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        return entity_agg

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None, print=False, return_logits=False):
        head, tail = edge_index
        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        if edge_type is not None:
            key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score


class KGRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(KGRec, self).__init__(conf, training_set, test_set)
        args = self.config['KGRec'] if self.config.contain('KGRec') else {}
        self.context_hops = int(args.get('context_hops', 2))
        self.node_dropout_rate = float(args.get('node_dropout_rate', 0.5))
        self.mess_dropout_rate = float(args.get('mess_dropout_rate', 0.1))
        self.cl_coef = float(args.get('cl_coef', 0.01))
        self.cl_tau = float(args.get('cl_tau', 0.5))
        self.cl_drop_ratio = float(args.get('cl_drop_ratio', 0.5))
        self.mae_coef = float(args.get('mae_coef', 0.1))
        self.mae_msize = int(args.get('mae_msize', 256))
        self.kg_loader = KGDataLoader(self.config, self.data)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_structures()
        dataset_name = os.path.basename(os.path.dirname(self.config['training.set']))
        self.checkpoint_dir = os.path.join(self.output, f"{self.model_name}_{dataset_name}_checkpoints")
        self.start_epoch = 0
        self._resume_checkpoint = None
        if self.config.contain('KGRec') and 'resume' in self.config['KGRec']:
            resume_path = self.config['KGRec']['resume']
            if resume_path and os.path.exists(resume_path):
                try:
                    checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(resume_path, map_location=self.device)
                self._resume_checkpoint = checkpoint
                sd = checkpoint.get('model_state_dict', {})
                if 'all_embed' in sd:
                    self.all_embed.data = sd['all_embed'].to(self.device)
                if 'gcn' in sd:
                    self.gcn.load_state_dict(sd['gcn'])
                if 'contrast_fn' in sd:
                    self.contrast_fn.load_state_dict(sd['contrast_fn'])
                self.start_epoch = int(checkpoint.get('epoch', 0))
                if 'bestPerformance' in checkpoint:
                    self.bestPerformance = checkpoint['bestPerformance']
                if 'best_user_emb' in checkpoint and 'best_item_emb' in checkpoint:
                    self.best_user_emb = checkpoint['best_user_emb']
                    self.best_item_emb = checkpoint['best_item_emb']

    def _build_structures(self):
        n_users = self.data.user_num
        n_items = self.data.item_num
        n_nodes = self.kg_loader.n_entities
        n_entities = n_nodes - n_users
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_nodes
        self.n_entities = n_entities
        kg_df = self.kg_loader.kg_triples
        head = torch.LongTensor(kg_df['h'].values - n_users)
        tail = torch.LongTensor(kg_df['t'].values - n_users)
        edge_index = torch.stack([head, tail], dim=0)
        edge_type_raw = torch.LongTensor(kg_df['r'].values)
        edge_type = edge_type_raw - 1
        self.edge_index = edge_index.to(self.device)
        self.edge_type = edge_type.to(self.device)
        unique_mapped_rels = torch.unique(edge_type).numel()
        self.n_relations = int(unique_mapped_rels + 1)
        inter = self.data.normalize_graph_mat(self.data.interaction_mat).tocoo()
        inter_i = torch.LongTensor(np.vstack([inter.row, inter.col])).to(self.device)
        inter_v = torch.from_numpy(inter.data).float().to(self.device)
        self.inter_edge = inter_i
        self.inter_edge_w = inter_v
        initializer = nn.init.xavier_uniform_
        init_tensor = initializer(torch.empty(self.n_nodes, self.emb_size, device=self.device))
        self.all_embed = nn.Parameter(init_tensor)
        self.gcn = AttnHGCN(channel=self.emb_size,
                            n_hops=self.context_hops,
                            n_users=self.n_users,
                            n_relations=self.n_relations,
                            node_dropout_rate=self.node_dropout_rate,
                            mess_dropout_rate=self.mess_dropout_rate).to(self.device)
        self.contrast_fn = Contrast(self.emb_size, tau=self.cl_tau).to(self.device)

    def build(self):
        pass

    def _model_state_dict(self):
        return {
            'all_embed': self.all_embed.data.detach().clone().to('cpu'),
            'gcn': self.gcn.state_dict(),
            'contrast_fn': self.contrast_fn.state_dict(),
        }

    def _load_rng_states(self, payload):
        try:
            if 'torch_rng_state' in payload:
                torch.set_rng_state(payload['torch_rng_state'])
            if torch.cuda.is_available() and 'cuda_rng_state_all' in payload:
                torch.cuda.set_rng_state_all(payload['cuda_rng_state_all'])
        except Exception:
            pass
        try:
            if 'numpy_rng_state' in payload:
                np.random.set_state(payload['numpy_rng_state'])
        except Exception:
            pass
        try:
            if 'python_rng_state' in payload:
                random.setstate(payload['python_rng_state'])
        except Exception:
            pass

    def _save_checkpoint(self, optimizer, epoch, path):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        payload = {
            'model_state_dict': self._model_state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': int(epoch),
            'bestPerformance': self.bestPerformance,
        }
        if hasattr(self, 'best_user_emb') and hasattr(self, 'best_item_emb'):
            payload['best_user_emb'] = self.best_user_emb
            payload['best_item_emb'] = self.best_item_emb
        try:
            payload['torch_rng_state'] = torch.get_rng_state()
            if torch.cuda.is_available():
                payload['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
        try:
            payload['numpy_rng_state'] = np.random.get_state()
        except Exception:
            pass
        try:
            payload['python_rng_state'] = random.getstate()
        except Exception:
            pass
        torch.save(payload, path)

    def _create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.reg * regularizer / batch_size
        return mf_loss + emb_loss, mf_loss, emb_loss

    def _create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
        scores = -torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores

    def _relation_aware_edge_sampling(self, edge_index, edge_type, n_relations, samp_rate=0.5):
        cat_index, cat_type = [], []
        for i in range(1, n_relations):
            mask = (edge_type == i)
            ei = edge_index[:, mask]
            et = edge_type[mask]
            n_edges = ei.shape[1]
            if n_edges == 0:
                continue
            keep = max(1, int(n_edges * samp_rate))
            perm = torch.randperm(n_edges, device=ei.device)[:keep]
            cat_index.append(ei[:, perm])
            cat_type.append(et[perm])
        edge_index_sampled = torch.cat(cat_index, dim=1) if len(cat_index) > 0 else edge_index
        edge_type_sampled = torch.cat(cat_type, dim=0) if len(cat_type) > 0 else edge_type
        return edge_index_sampled, edge_type_sampled

    def _adaptive_kg_drop_cl(self, edge_index, edge_type, edge_attn_score, keep_rate):
        n_keep = int(keep_rate * edge_attn_score.shape[0])
        n_keep = max(1, n_keep)
        sampled_edge_idx = torch.topk(edge_attn_score, n_keep, sorted=False).indices
        cl_kg_edge = edge_index[:, sampled_edge_idx]
        cl_kg_type = edge_type[sampled_edge_idx]
        return cl_kg_edge, cl_kg_type

    def _sparse_dropout(self, i, v, keep_rate=0.5):
        noise_shape = i.shape[1]
        random_tensor = keep_rate + torch.rand(noise_shape, device=i.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = i[:, dropout_mask]
        v = v[dropout_mask] / keep_rate
        return i, v

    def _forward_batch(self, batch, epoch_start=False):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        edge_index, edge_type = self._relation_aware_edge_sampling(self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)
        edge_attn_score, _ = self.gcn.norm_attn_computer(entity_emb, edge_index, edge_type, print=epoch_start, return_logits=True)
        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities)
        item_attn_mean_1[item_attn_mean_1 == 0.] = 1.
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean_2[item_attn_mean_2 == 0.] = 1.
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score_noisy = edge_attn_score + noise
        topk_v, topk_attn_edge_id = torch.topk(edge_attn_score_noisy, min(self.mae_msize, edge_attn_score_noisy.shape[0]), sorted=False)
        masked_edge_index = edge_index[:, topk_attn_edge_id]
        masked_edge_type = edge_type[topk_attn_edge_id]
        enc_edge_index = edge_index
        enc_edge_type = edge_type
        inter_edge, inter_edge_w = self._sparse_dropout(self.inter_edge, self.inter_edge_w, self.node_dropout_rate)
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb, entity_emb, enc_edge_index, enc_edge_type, inter_edge, inter_edge_w, mess_dropout=True)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        loss, rec_loss, reg_loss = self._create_bpr_loss(u_e, pos_e, neg_e)
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type - 1]
        mae_loss = self.mae_coef * self._create_mae_loss(node_pair_emb, masked_edge_emb)
        cl_kg_edge, cl_kg_type = self._adaptive_kg_drop_cl(edge_index, edge_type, edge_attn_score, keep_rate=1 - self.cl_drop_ratio)
        item_agg_ui = self.gcn.forward_ui(user_emb, entity_emb[:self.n_items], inter_edge, inter_edge_w)
        item_agg_kg = self.gcn.forward_kg(entity_emb, cl_kg_edge, cl_kg_type)[:self.n_items]
        cl_loss = self.cl_coef * self.contrast_fn(item_agg_ui, item_agg_kg)
        total = loss + mae_loss + cl_loss
        return total, {'rec_loss': loss.item(), 'mae_loss': mae_loss.item(), 'cl_loss': cl_loss.item()}

    @torch.no_grad()
    def _generate_embeddings(self):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb, entity_emb, self.edge_index, self.edge_type, self.inter_edge, self.inter_edge_w, mess_dropout=False)
        item_emb = entity_gcn_emb[:self.n_items]
        return user_gcn_emb, item_emb

    def train(self):
        optim_params = [{'params': [self.all_embed], 'lr': self.lRate},
                        {'params': self.gcn.parameters(), 'lr': self.lRate},
                        {'params': self.contrast_fn.parameters(), 'lr': self.lRate}]
        optimizer = torch.optim.Adam(optim_params, lr=self.lRate)
        if self._resume_checkpoint is not None and 'optimizer_state_dict' in self._resume_checkpoint:
            try:
                optimizer.load_state_dict(self._resume_checkpoint['optimizer_state_dict'])
            except Exception:
                pass
            self._load_rng_states(self._resume_checkpoint)
        for epoch in range(self.start_epoch, self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                u_idx, i_idx, j_idx = batch
                feed = {
                    'users': torch.LongTensor(u_idx).to(self.device),
                    'pos_items': torch.LongTensor(i_idx).to(self.device),
                    'neg_items': torch.LongTensor(j_idx).to(self.device),
                    'batch_start': 1 if n == 0 else 0
                }
                loss, loss_dict = self._forward_batch(feed, epoch_start=(n == 0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                msg = f"training: epoch {epoch + 1} batch {n} total_loss {loss.item():.6f} rec_loss {loss_dict['rec_loss']:.6f} mae_loss {loss_dict['mae_loss']:.6f} cl_loss {loss_dict['cl_loss']:.6f}"
                print(msg)
                self.model_log.add(msg)
            with torch.no_grad():
                self.user_emb, self.item_emb = self._generate_embeddings()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
            checkpoint_path = os.path.join(self.checkpoint_dir, "last.pth")
            self._save_checkpoint(optimizer, epoch + 1, checkpoint_path)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self._generate_embeddings()
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_epoch = self.bestPerformance[0] if self.bestPerformance else 0
        torch.save(
            {
                'model_state_dict': self._model_state_dict(),
                'epoch': int(best_epoch),
                'bestPerformance': self.bestPerformance,
            },
            best_model_path
        )

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

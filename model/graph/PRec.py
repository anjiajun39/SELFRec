import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
from util.sampler import next_batch_pairwise


class PRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(PRec, self).__init__(conf, training_set, test_set)
        args = self.config['PRec']
        self.n_layers = int(args['n_layer'])
        self.hyper_dim = int(args['hyper_dim'])
        self.p = float(args['p'])
        self.drop_rate = float(args['drop_rate'])
        self.early_stopping = int(args.get('early_stopping', 10))
        self.model = HGNNModel(self.data, self.emb_size, self.hyper_dim, self.n_layers, self.p, self.drop_rate)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        best_ndcg = None
        epochs_no_improve = 0
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_emb, item_emb = model()
                anchor_emb = user_emb[user_idx]
                pos_emb = item_emb[pos_idx]
                neg_emb = item_emb[neg_idx]
                rec_loss = bpr_loss(anchor_emb, pos_emb, neg_emb)
                reg_loss = l2_reg_loss(self.reg, anchor_emb, pos_emb, neg_emb) / self.batch_size
                batch_loss = rec_loss + reg_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            measure = self.fast_evaluation(epoch)
            ndcg = self._get_ndcg_from_measure(measure)
            if best_ndcg is None or ndcg > best_ndcg:
                best_ndcg = ndcg
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if self.early_stopping > 0 and epochs_no_improve >= self.early_stopping:
                print('Early stopping triggered at epoch', epoch + 1, 'best NDCG:', best_ndcg)
                break
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def _get_ndcg_from_measure(self, measure):
        for line in measure:
            line = line.strip()
            if line.startswith('NDCG:'):
                return float(line.split(':')[1])
        return 0.0

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class HGNNModel(nn.Module):
    def __init__(self, data, emb_size, hyper_dim, n_layers, p, drop_rate):
        super(HGNNModel, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.hyper_dim = hyper_dim
        self.n_layers = n_layers
        self.p = p
        self.drop_rate = drop_rate
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).cuda()
        self.embedding_dict = self._init_model()
        self.hgnn_layer_cf = SelfAwareEncoder(self.data, self.emb_size, self.hyper_dim, self.n_layers, self.p, self.drop_rate)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_entity_emb': nn.Parameter(initializer(torch.empty(self.data.user_num + self.data.item_num, self.hyper_dim))),
        })
        return embedding_dict

    def calculate_cf_embeddings(self, keep_rate: float = 1):
        ego_embeddings = self.embedding_dict['user_entity_emb']
        sparse_norm_adj = SpAdjDropEdge()(self.sparse_norm_adj, keep_rate)
        user_all_embeddings, item_all_embeddings = self.hgnn_layer_cf(ego_embeddings, sparse_norm_adj)
        return user_all_embeddings, item_all_embeddings

    def forward(self, keep_rate=1):
        user_embed, item_embed = self.calculate_cf_embeddings(keep_rate=keep_rate)
        return user_embed, item_embed

class SelfAwareEncoder(nn.Module):
    def __init__(self, data, emb_size, hyper_size, n_layers, leaky, drop_rate, use_self_att=False):
        super(SelfAwareEncoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.hyper_size = hyper_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.relu = nn.ReLU()
        self.act = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(drop_rate)
        self.edgeDropper = SpAdjDropEdge()
        
        self.use_self_att = use_self_att

        self.hgnn_layers = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()

        for i in range(self.layers):
            self.hgnn_layers.append(HGCNConv(leaky=leaky))
            self.lns.append(torch.nn.LayerNorm(hyper_size))

    def forward(self, ego_embeddings, sparse_norm_adj):
        # ego_embeddings: [n_users + n_items, hyper_size]
        # sparse_norm_adj: [n_users + n_items, n_users + n_items] (稀疏矩阵)
        res = ego_embeddings     # 残差初始化
        all_embeddings = []      
        for k in range(self.layers):
            if k != self.layers - 1:
                ego_embeddings = self.lns[k](self.hgnn_layers[k](sparse_norm_adj, ego_embeddings)) + res
            else:
                ego_embeddings = self.lns[k](self.hgnn_layers[k](sparse_norm_adj, ego_embeddings, act=False)) + res
            all_embeddings += [ego_embeddings]
        
        user_all_embeddings = all_embeddings[-1][:self.data.user_num]
        item_all_embeddings = all_embeddings[-1][self.data.user_num:self.data.user_num + self.data.item_num]
        return user_all_embeddings, item_all_embeddings


class HGCNConv(nn.Module):
    def __init__(self, leaky):
        super(HGCNConv, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, adj, embs, act=True):
        if act:
            # torch.sparse.mm(adj.t(), embs)  [n_nodes, n_nodes] @ [n_nodes, embed_dim] → [n_nodes, embed_dim] 消息传递机制
            # adj.t(): 邻接矩阵的转置，形状为 [n_nodes, n_nodes]
            # embs: 节点嵌入，形状为 [n_nodes, embed_dim]
            # 结果: 聚合邻居节点的信息
            # 外层torch.sparse.mm 进行消息聚合
            # [n_nodes, n_nodes] @ [n_nodes, embed_dim] → [n_nodes, embed_dim]
            return self.act(torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs)))
        else:
            return torch.sparse.mm(adj, torch.sparse.mm(adj.t(), embs))

class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return torch.sparse_coo_tensor(newIdxs, newVals, adj.shape)
    

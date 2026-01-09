import numpy as np
import torch
import torch.nn as nn

from base.graph_recommender import GraphRecommender
from base.torch_interface import TorchGraphInterface
from data.augmentor import GraphAugmentor
from util.loss_torch import InfoNCE, bpr_loss, l2_reg_loss
from util.sampler import next_batch_pairwise


class HCCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(HCCF, self).__init__(conf, training_set, test_set)
        args = self.config['HCCF']
        self.n_layers = int(args['n_layer'])
        self.cl_rate = float(args['lambda'])
        self.drop_rate = float(args['drop_rate'])
        self.aug_type = int(args['aug_type'])
        self.temp = float(args['tau'])
        self.model = HCCF_Encoder(self.data, self.emb_size, self.n_layers, self.drop_rate, self.aug_type, self.temp)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            hg_view_1 = model.hypergraph_reconstruction()
            hg_view_2 = model.hypergraph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * model.cal_cl_loss([user_idx, pos_idx], hg_view_1, hg_view_2)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) + cl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class HCCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, drop_rate, aug_type, temp):
        super(HCCF_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.aug_type = aug_type
        self.temp = temp

        self.embedding_dict = self._init_model()
        self.base_hg = self._build_hg_inputs(self.data.interaction_mat)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        return nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })

    def _build_hg_inputs(self, interaction_mat):
        r = interaction_mat.tocsr()
        sparse_r = TorchGraphInterface.convert_sparse_mat_to_tensor(r).cuda()
        sparse_rt = TorchGraphInterface.convert_sparse_mat_to_tensor(r.transpose().tocsr()).cuda()

        user_degree = np.asarray(r.sum(axis=1)).reshape(-1)
        item_degree = np.asarray(r.sum(axis=0)).reshape(-1)
        user_degree[user_degree == 0] = 1.0
        item_degree[item_degree == 0] = 1.0

        d_u_inv = np.power(user_degree, -1.0)
        d_i_inv = np.power(item_degree, -1.0)
        d_u_inv_sqrt = np.power(user_degree, -0.5)
        d_i_inv_sqrt = np.power(item_degree, -0.5)

        return {
            'r': sparse_r,
            'rt': sparse_rt,
            'd_u_inv': torch.tensor(d_u_inv, dtype=torch.float32).cuda().view(-1, 1),
            'd_i_inv': torch.tensor(d_i_inv, dtype=torch.float32).cuda().view(-1, 1),
            'd_u_inv_sqrt': torch.tensor(d_u_inv_sqrt, dtype=torch.float32).cuda().view(-1, 1),
            'd_i_inv_sqrt': torch.tensor(d_i_inv_sqrt, dtype=torch.float32).cuda().view(-1, 1),
        }

    def hypergraph_reconstruction(self):
        if self.aug_type == 0:
            dropped = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        else:
            dropped = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        return self._build_hg_inputs(dropped)

    def _hg_forward(self, hg_inputs, user_init, item_init):
        user_emb = user_init
        item_emb = item_init

        all_user = [user_emb]
        all_item = [item_emb]

        r = hg_inputs['r']
        rt = hg_inputs['rt']
        d_u_inv = hg_inputs['d_u_inv']
        d_i_inv = hg_inputs['d_i_inv']
        d_u_inv_sqrt = hg_inputs['d_u_inv_sqrt']
        d_i_inv_sqrt = hg_inputs['d_i_inv_sqrt']

        for _ in range(self.n_layers):
            tmp_u = user_emb * d_u_inv_sqrt
            tmp_i = torch.sparse.mm(rt, tmp_u)
            tmp_i = tmp_i * d_i_inv
            tmp_u = torch.sparse.mm(r, tmp_i)
            user_emb = tmp_u * d_u_inv_sqrt
            all_user.append(user_emb)

            tmp_i2 = item_emb * d_i_inv_sqrt
            tmp_u2 = torch.sparse.mm(r, tmp_i2)
            tmp_u2 = tmp_u2 * d_u_inv
            tmp_i2 = torch.sparse.mm(rt, tmp_u2)
            item_emb = tmp_i2 * d_i_inv_sqrt
            all_item.append(item_emb)

        user_out = torch.mean(torch.stack(all_user, dim=1), dim=1)
        item_out = torch.mean(torch.stack(all_item, dim=1), dim=1)
        return user_out, item_out

    def cal_cl_loss(self, idx, hg_view_1, hg_view_2):
        u_idx = torch.unique(torch.tensor(idx[0], dtype=torch.long, device='cuda'))
        i_idx = torch.unique(torch.tensor(idx[1], dtype=torch.long, device='cuda'))
        user_init = self.embedding_dict['user_emb']
        item_init = self.embedding_dict['item_emb']
        user_v1, item_v1 = self._hg_forward(hg_view_1, user_init, item_init)
        user_v2, item_v2 = self._hg_forward(hg_view_2, user_init, item_init)
        user_cl = InfoNCE(user_v1[u_idx], user_v2[u_idx], self.temp)
        item_cl = InfoNCE(item_v1[i_idx], item_v2[i_idx], self.temp)
        return user_cl + item_cl

    def forward(self):
        user_init = self.embedding_dict['user_emb']
        item_init = self.embedding_dict['item_emb']
        return self._hg_forward(self.base_hg, user_init, item_init)


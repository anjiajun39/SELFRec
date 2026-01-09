import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
import numpy as np

# Paper: Dual Channel Hypergraph Collaborative Filtering. KDD'20

class DHCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DHCF, self).__init__(conf, training_set, test_set)
        args = self.config['DHCF']
        self.n_layers = int(args['n_layer'])
        self.model = DHCF_Encoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx], model.embedding_dict['item_emb'][pos_idx], model.embedding_dict['item_emb'][neg_idx])/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class DHCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(DHCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        # Hypergraph Channel Matrices
        # R: Interaction Matrix (User x Item)
        self.R = data.interaction_mat
        self.sparse_R = TorchGraphInterface.convert_sparse_mat_to_tensor(self.R).cuda()
        self.sparse_R_T = TorchGraphInterface.convert_sparse_mat_to_tensor(self.R.transpose()).cuda()

        # Calculate Degrees
        user_degree = np.array(self.R.sum(axis=1)).squeeze()
        item_degree = np.array(self.R.sum(axis=0)).squeeze()

        # Avoid division by zero
        user_degree[user_degree == 0] = 1
        item_degree[item_degree == 0] = 1

        d_u_inv = np.power(user_degree, -1)
        d_i_inv = np.power(item_degree, -1)
        d_u_inv_sqrt = np.power(user_degree, -0.5)
        d_i_inv_sqrt = np.power(item_degree, -0.5)

        # Convert to tensors
        self.d_u_inv = torch.tensor(d_u_inv, dtype=torch.float32).cuda().view(-1, 1)
        self.d_i_inv = torch.tensor(d_i_inv, dtype=torch.float32).cuda().view(-1, 1)
        self.d_u_inv_sqrt = torch.tensor(d_u_inv_sqrt, dtype=torch.float32).cuda().view(-1, 1)
        self.d_i_inv_sqrt = torch.tensor(d_i_inv_sqrt, dtype=torch.float32).cuda().view(-1, 1)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        user_emb = self.embedding_dict['user_emb']
        item_emb = self.embedding_dict['item_emb']
        
        # --- Graph Channel (LightGCN) ---
        ego_embeddings = torch.cat([user_emb, item_emb], 0)
        all_embeddings_gcn = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings_gcn.append(ego_embeddings)
        
        # Mean Pooling for GCN
        all_embeddings_gcn = torch.stack(all_embeddings_gcn, dim=1)
        all_embeddings_gcn = torch.mean(all_embeddings_gcn, dim=1)
        user_emb_gcn = all_embeddings_gcn[:self.data.user_num]
        item_emb_gcn = all_embeddings_gcn[self.data.user_num:]
        
        # --- Hypergraph Channel ---
        # Initialize
        user_emb_hg = user_emb
        item_emb_hg = item_emb
        all_user_hg = [user_emb_hg]
        all_item_hg = [item_emb_hg]
        
        for k in range(self.layers):
            # User Hypergraph Propagation
            # X_u_new = D_u^{-0.5} * R * D_i^{-1} * R^T * D_u^{-0.5} * X_u
            # 1. D_u^{-0.5} * X_u
            temp_u = user_emb_hg * self.d_u_inv_sqrt
            # 2. R^T * temp_u
            temp_i_agg = torch.sparse.mm(self.sparse_R_T, temp_u)
            # 3. D_i^{-1} * temp_i_agg
            temp_i_agg = temp_i_agg * self.d_i_inv
            # 4. R * temp_i_agg
            temp_u_agg = torch.sparse.mm(self.sparse_R, temp_i_agg)
            # 5. D_u^{-0.5} * temp_u_agg
            user_emb_hg = temp_u_agg * self.d_u_inv_sqrt
            all_user_hg.append(user_emb_hg)

            # Item Hypergraph Propagation
            # X_i_new = D_i^{-0.5} * R^T * D_u^{-1} * R * D_i^{-0.5} * X_i
            # 1. D_i^{-0.5} * X_i
            temp_i = item_emb_hg * self.d_i_inv_sqrt
            # 2. R * temp_i
            temp_u_agg = torch.sparse.mm(self.sparse_R, temp_i)
            # 3. D_u^{-1} * temp_u_agg
            temp_u_agg = temp_u_agg * self.d_u_inv
            # 4. R^T * temp_u_agg
            temp_i_agg = torch.sparse.mm(self.sparse_R_T, temp_u_agg)
            # 5. D_i^{-0.5} * temp_i_agg
            item_emb_hg = temp_i_agg * self.d_i_inv_sqrt
            all_item_hg.append(item_emb_hg)

        # Mean Pooling for HG
        user_emb_hg = torch.stack(all_user_hg, dim=1)
        user_emb_hg = torch.mean(user_emb_hg, dim=1)
        item_emb_hg = torch.stack(all_item_hg, dim=1)
        item_emb_hg = torch.mean(item_emb_hg, dim=1)
        
        # --- Combination ---
        # Summing Graph and Hypergraph channels
        final_user_emb = user_emb_gcn + user_emb_hg
        final_item_emb = item_emb_gcn + item_emb_hg
        
        return final_user_emb, final_item_emb

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from data.kg_processor import KGData
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss
import numpy as np
import random

class KGAT(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(KGAT, self).__init__(conf, training_set, test_set)
        self.kg_data = KGData(conf, self.data)
        
        args = self.config['KGAT']
        self.n_layers = int(args['n_layer'])
        self.kg_lambda = float(args['kg_lambda'])
        
        # Build CKG
        self.ckg_adj = self.kg_data.get_ckg_adj()
        self.n_users = self.kg_data.n_users
        self.n_entities = self.kg_data.num_entities
        self.n_relations = self.kg_data.num_relations
        
        # Convert CKG to edge tensor for batch processing
        # Edges: [src, dst, rel]
        edges = []
        for src, neighbors in self.ckg_adj.items():
            for dst, rel in neighbors:
                edges.append([src, dst, rel])
        
        self.edges = torch.LongTensor(edges).t() # [3, E]
        if torch.cuda.is_available():
            self.edges = self.edges.cuda()
            
        self.model = KGAT_Encoder(
            self.n_users, 
            self.n_entities, 
            self.n_relations, 
            self.emb_size, 
            self.edges, 
            self.n_layers,
            self.config['KGAT']
        )
        
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        for epoch in range(self.maxEpoch):
            model.train()
            # 1. Train KG (TransR part) - Optional or Joint?
            # KGAT usually does joint training or alternating. 
            # We will implement joint training by sampling both.
            
            # Sampling
            # CF Batch
            cf_batch_generator = next_batch_pairwise(self.data, self.batch_size)
            
            # KG Batch
            # We need a KG triplet sampler.
            kg_triplets = self.kg_data.kg_triplets
            n_kg_batches = len(kg_triplets) // self.batch_size
            
            total_loss = 0
            
            # For simplicity, we iterate over CF batches and sample KG batch randomly
            for n, batch in enumerate(cf_batch_generator):
                user_idx, pos_idx, neg_idx = batch
                
                # KG Batch
                kg_batch_idx = np.random.choice(len(kg_triplets), self.batch_size)
                kg_batch_data = kg_triplets[kg_batch_idx] # [h, r, t]
                h = torch.LongTensor(kg_batch_data[:, 0]).cuda()
                r = torch.LongTensor(kg_batch_data[:, 1]).cuda()
                t = torch.LongTensor(kg_batch_data[:, 2]).cuda()
                
                # Forward Pass
                # Propagated embeddings
                entity_emb, user_emb = model() 
                
                # CF Loss
                # Users are 0~n_users-1
                # Items are mapped to Entities 0~n_items-1 (which are 0~n_items-1 in entity space)
                # But in CKG, Items are nodes n_users + item_id.
                # Wait, my CKG construction:
                # User Node: user_id (0 ~ n_users-1)
                # Item Node: n_users + item_id
                
                u_e = user_emb[user_idx]
                pos_e = entity_emb[pos_idx] # pos_idx is item_id
                neg_e = entity_emb[neg_idx]
                
                rec_loss = bpr_loss(u_e, pos_e, neg_e)
                
                # KG Loss (TransR)
                # Uses original embeddings (0-th layer) usually, or propagated?
                # KGAT paper: L_KG uses TransR on *initial* embeddings.
                # L_CF uses *propagated* embeddings.
                
                # TransR Loss
                # h + r approx t
                # We need negative sampling for KG too.
                # Simple negative sampling: replace t with t'
                t_neg_idx = np.random.randint(0, self.n_entities, self.batch_size)
                t_neg = torch.LongTensor(t_neg_idx).cuda()
                
                kg_loss = model.calc_kg_loss(h, r, t, t_neg)
                
                # Reg Loss
                reg_loss = l2_reg_loss(self.reg, u_e, pos_e, neg_e)
                
                loss = rec_loss + self.kg_lambda * kg_loss + reg_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if n % 50 == 0:
                    print(f"Epoch {epoch+1} Batch {n}: Loss = {loss.item():.4f} (Rec={rec_loss.item():.4f}, KG={kg_loss.item():.4f})")
            
            with torch.no_grad():
                model.eval()
                entity_emb, user_emb = model()
                self.user_emb = user_emb
                self.item_emb = entity_emb[:self.data.item_num]

            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
                
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            entity_emb, user_emb = self.model.forward()
            self.best_user_emb = user_emb
            self.best_item_emb = entity_emb[:self.data.item_num]

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class KGAT_Encoder(nn.Module):
    def __init__(self, n_users, n_entities, n_relations, emb_size, edges, n_layers, conf):
        super(KGAT_Encoder, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.emb_size = emb_size
        self.edges = edges # [3, E]
        self.n_layers = n_layers
        self.mess_dropout = float(conf['mess_dropout'])
        self.aggregator_type = conf['aggregator.type']
        try:
            self.adj_type = conf['adj.type']
        except KeyError:
            self.adj_type = 'softmax'
        
        # Embeddings
        # User embeddings: 0 ~ n_users-1
        # Entity embeddings: 0 ~ n_entities-1
        self.user_embedding = nn.Embedding(n_users, emb_size)
        self.entity_embedding = nn.Embedding(n_entities, emb_size)
        
        # Relation embeddings: 
        # 0 ~ num_relations-1: Original
        # num_relations ~ 2*num_relations-1: Inverse
        # 2*num_relations: Interaction
        self.relation_embedding = nn.Embedding(2 * n_relations + 1, emb_size)
        
        # TransR projection matrix W_r
        # Maps entity embedding to relation space
        self.W_R = nn.Parameter(torch.Tensor(2 * n_relations + 1, emb_size, emb_size))
        
        # Aggregator Weights
        if self.aggregator_type == 'gcn':
            self.W_algo = nn.Parameter(torch.Tensor(emb_size, emb_size))
        elif self.aggregator_type == 'graphsage':
            self.W_algo = nn.Parameter(torch.Tensor(2 * emb_size, emb_size))
        elif self.aggregator_type == 'bi-interaction':
            self.W_1 = nn.Parameter(torch.Tensor(emb_size, emb_size))
            self.W_2 = nn.Parameter(torch.Tensor(emb_size, emb_size))
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.W_R)
        
        if self.aggregator_type == 'gcn':
            nn.init.xavier_uniform_(self.W_algo)
        elif self.aggregator_type == 'graphsage':
            nn.init.xavier_uniform_(self.W_algo)
        elif self.aggregator_type == 'bi-interaction':
            nn.init.xavier_uniform_(self.W_1)
            nn.init.xavier_uniform_(self.W_2)

    def calc_kg_loss(self, h, r, t, t_neg):
        # TransR Loss
        # h, r, t are indices
        
        h_emb = self.entity_embedding(h)
        t_emb = self.entity_embedding(t)
        t_neg_emb = self.entity_embedding(t_neg)
        r_emb = self.relation_embedding(r)
        
        W_r = self.W_R[r] # [B, d, d]
        
        # Project h, t to relation space
        # h_r = h M_r
        # [B, 1, d] x [B, d, d] -> [B, 1, d]
        h_r = torch.bmm(h_emb.unsqueeze(1), W_r).squeeze(1)
        t_r = torch.bmm(t_emb.unsqueeze(1), W_r).squeeze(1)
        t_neg_r = torch.bmm(t_neg_emb.unsqueeze(1), W_r).squeeze(1)
        
        pos_score = torch.sum((h_r + r_emb - t_r) ** 2, dim=1)
        neg_score = torch.sum((h_r + r_emb - t_neg_r) ** 2, dim=1)
        
        kg_loss = torch.mean(F.softplus(pos_score - neg_score))
        return kg_loss

    def forward(self):
        # Propagate embeddings
        
        # Initial embeddings
        # We concatenate user and entity embeddings to form node embeddings for CKG
        # User nodes: 0 ~ n_users-1
        # Entity nodes: n_users ~ n_users + n_entities - 1
        
        all_emb = torch.cat([self.user_embedding.weight, self.entity_embedding.weight], dim=0)
        ego_emb = all_emb
        
        embs = [ego_emb]
        
        # Edges
        src, dst, rel = self.edges # [E]
        n_nodes = all_emb.shape[0]
        
        for k in range(self.n_layers):
            # 1. Compute Attention Scores
            # pi(h, r, t) = LeakyReLU(W_att [h_r || t_r]) or similar
            # KGAT uses: tanh(h W t)
            
            # Simple Attention: (h W_Q) * (t W_K)
            # But we need relation-aware.
            # h_r = W_r h, t_r = W_r t
            # score = sum(h_r * t_r * r_att)
            
            # Simplified for memory:
            # score = (h * r) . t
            
            # Implementation choice:
            # Using simple GAT-like attention on projected embeddings
            
            h_emb = ego_emb[src]
            t_emb = ego_emb[dst]
            r_emb = self.relation_embedding(rel)
            
            # Relation-aware attention
            # pi = LeakyReLU(a^T [Wh || Wt || Wr])
            # Or KGAT paper: pi(h,r,t) = (Wh)^T W_r (Wt) ? No.
            # Paper Eq 4: pi(h,r,t) = (W_r e_t)^T tanh(W_r e_h + e_r)
            
            # Let's stick to a simpler GAT or closely follow if possible.
            # Given constraints, let's use a simpler bi-linear attention
            # score = sum(h * t * r, dim=1)
            
            if self.adj_type == 'softmax':
                logits = F.leaky_relu(torch.sum(h_emb * t_emb * r_emb, dim=1))
                max_per_src = torch.full((n_nodes,), float('-inf'), device=all_emb.device, dtype=all_emb.dtype)
                max_per_src.scatter_reduce_(0, src, logits, reduce='amax', include_self=True)
                exp_score = torch.exp(logits - max_per_src[src])
                denom = all_emb.new_zeros(n_nodes)
                denom.scatter_add_(0, src, exp_score)
                att = exp_score / (denom[src] + 1e-9)
            
            elif self.adj_type == 'symmetric':
                deg = all_emb.new_zeros(n_nodes)
                deg.scatter_add_(0, src, all_emb.new_ones(src.shape[0]))
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt = torch.where(torch.isinf(deg_inv_sqrt), torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
                att = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
                
            elif self.adj_type == 'random-walk':
                deg = all_emb.new_zeros(n_nodes)
                deg.scatter_add_(0, src, all_emb.new_ones(src.shape[0]))
                deg_inv = deg.pow(-1)
                deg_inv = torch.where(torch.isinf(deg_inv), torch.zeros_like(deg_inv), deg_inv)
                att = deg_inv[src]
                
            else:
                logits = F.leaky_relu(torch.sum(h_emb * t_emb * r_emb, dim=1))
                max_per_src = torch.full((n_nodes,), float('-inf'), device=all_emb.device, dtype=all_emb.dtype)
                max_per_src.scatter_reduce_(0, src, logits, reduce='amax', include_self=True)
                exp_score = torch.exp(logits - max_per_src[src])
                denom = all_emb.new_zeros(n_nodes)
                denom.scatter_add_(0, src, exp_score)
                att = exp_score / (denom[src] + 1e-9)
            
            # Aggregate: sum(att * (t + r)) or similar
            # KGAT: sum(att * t_emb) usually, relation used in attention.
            # Actually KGAT aggregates: e_Nh = sum(att * e_t)
            
            # We also need to aggregate relation info?
            # Paper: values to aggregate are e_t. Relation is in attention.
            
            msg = t_emb * att.unsqueeze(1)
            msg = F.dropout(msg, p=self.mess_dropout, training=self.training)
            
            # Aggregation
            # aggr_emb = zeros.scatter_add(0, src, msg)
            aggr_emb = torch.zeros_like(all_emb)
            aggr_emb = aggr_emb.scatter_add(0, src.unsqueeze(1).expand_as(msg), msg)
            
            # Aggregator Selection
            if self.aggregator_type == 'gcn':
                new_emb = F.leaky_relu(torch.matmul(ego_emb + aggr_emb, self.W_algo))
            elif self.aggregator_type == 'graphsage':
                new_emb = F.leaky_relu(torch.matmul(torch.cat([ego_emb, aggr_emb], dim=1), self.W_algo))
            elif self.aggregator_type == 'bi-interaction':
                new_emb = F.leaky_relu(torch.matmul(ego_emb + aggr_emb, self.W_1) + \
                                       torch.matmul(ego_emb * aggr_emb, self.W_2))
            
            # Normalize
            new_emb = F.normalize(new_emb, dim=1)
            
            embs.append(new_emb)
            ego_emb = new_emb
            
        # Stack and Mean
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        
        # Split back
        user_res = final_emb[:self.n_users]
        
        # Item embeddings are entities 0 ~ n_items-1
        # But in CKG, items are nodes n_users + item_id
        item_res = final_emb[self.n_users : self.n_users + self.n_entities]
        
        return item_res, user_res

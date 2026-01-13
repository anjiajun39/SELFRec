import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss
from data.kg_loader import KGDataLoader
import numpy as np
import random

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class KGAT(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(KGAT, self).__init__(conf, training_set, test_set)
        
        # Initialize KG Data Loader
        self.kg_loader = KGDataLoader(conf, self.data)
        
        self.model = KGAT_Encoder(self.data, self.kg_loader, self.config)
        
        # Pre-load KG data for training
        self.kg_h, self.kg_r, self.kg_t = self.kg_loader.get_kg_data()
        self.num_kg_triples = len(self.kg_h)

        # Data for Attention Update (All triples: KG + CF)
        self.all_h = torch.LongTensor(self.kg_loader.all_train_data['h'].values)
        self.all_r = torch.LongTensor(self.kg_loader.all_train_data['r'].values)
        self.all_t = torch.LongTensor(self.kg_loader.all_train_data['t'].values)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset_name = os.path.basename(os.path.dirname(self.config['training.set']))
        self.checkpoint_dir = os.path.join(self.output, f"{self.model_name}_{dataset_name}_checkpoints")
        self.start_epoch = 0
        self._resume_checkpoint = None

        if self.config.contain('KGAT') and 'resume' in self.config['KGAT']:
            resume_path = self.config['KGAT']['resume']
            if resume_path and os.path.exists(resume_path):
                try:
                    checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(resume_path, map_location=self.device)
                self._resume_checkpoint = checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.start_epoch = int(checkpoint.get('epoch', 0))
                if 'bestPerformance' in checkpoint:
                    self.bestPerformance = checkpoint['bestPerformance']
                if 'best_user_emb' in checkpoint and 'best_item_emb' in checkpoint:
                    self.best_user_emb = checkpoint['best_user_emb']
                    self.best_item_emb = checkpoint['best_item_emb']
                print(f"Resume KGAT from checkpoint: {resume_path}, start_epoch = {self.start_epoch}")

    def _save_checkpoint(self, model, optimizer, epoch, path):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        payload = {
            'model_state_dict': model.state_dict(),
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

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        if self._resume_checkpoint is not None and 'optimizer_state_dict' in self._resume_checkpoint:
            optimizer.load_state_dict(self._resume_checkpoint['optimizer_state_dict'])
            try:
                if 'torch_rng_state' in self._resume_checkpoint:
                    torch.set_rng_state(self._resume_checkpoint['torch_rng_state'])
                if torch.cuda.is_available() and 'cuda_rng_state_all' in self._resume_checkpoint:
                    torch.cuda.set_rng_state_all(self._resume_checkpoint['cuda_rng_state_all'])
            except Exception:
                pass
            try:
                if 'numpy_rng_state' in self._resume_checkpoint:
                    np.random.set_state(self._resume_checkpoint['numpy_rng_state'])
            except Exception:
                pass
            try:
                if 'python_rng_state' in self._resume_checkpoint:
                    random.setstate(self._resume_checkpoint['python_rng_state'])
            except Exception:
                pass
        
        for epoch in range(self.start_epoch, self.maxEpoch):
            # CF Training
            cf_total_loss = 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_idx = torch.LongTensor(user_idx).to(self.device)
                pos_idx = torch.LongTensor(pos_idx).to(self.device)
                neg_idx = torch.LongTensor(neg_idx).to(self.device)
                
                loss = model.calc_cf_loss(user_idx, pos_idx, neg_idx)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                cf_total_loss += loss.item()
                if n % 50 == 0:
                    print(f'Epoch {epoch + 1} CF Batch {n}: Loss = {loss.item():.4f}, Mean Loss = {cf_total_loss/(n+1):.4f}')
            
            # KG Training
            kg_total_loss = 0
            kg_batch_size = self.config['KGAT']['kg_batch_size']
            num_batches = self.num_kg_triples // kg_batch_size
            
            for i in range(num_batches):
                h_batch, r_batch, pos_t_batch, neg_t_batch = self.kg_loader.generate_kg_batch(kg_batch_size)
                
                h_batch = h_batch.to(self.device)
                r_batch = r_batch.to(self.device)
                pos_t_batch = pos_t_batch.to(self.device)
                neg_t_batch = neg_t_batch.to(self.device)
                
                kg_loss = model.calc_kg_loss(h_batch, r_batch, pos_t_batch, neg_t_batch)
                
                optimizer.zero_grad()
                kg_loss.backward()
                optimizer.step()
                
                kg_total_loss += kg_loss.item()
                if i % 50 == 0:
                    print(f'Epoch {epoch + 1} KG Batch {i}: Loss = {kg_loss.item():.4f}, Mean Loss = {kg_total_loss/(i+1):.4f}')
            
            # Move Attention data to cuda
            all_h = self.all_h.to(self.device)
            all_r = self.all_r.to(self.device)
            all_t = self.all_t.to(self.device)
            # Update Attention
            relations = list(self.kg_loader.laplacian_dict.keys())
            model.update_attention(all_h, all_t, all_r, relations)

            with torch.no_grad():
                self.user_emb, self.item_emb = model.predict_embedding()
            
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)

            checkpoint_path = os.path.join(self.checkpoint_dir, "last.pth")
            self._save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
        
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.predict_embedding()
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_epoch = self.bestPerformance[0] if self.bestPerformance else 0
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'epoch': int(best_epoch),
                'bestPerformance': self.bestPerformance,
            },
            best_model_path
        )

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        else:
            raise NotImplementedError

    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # A_in is sparse matrix
        # Equation (3)
        side_embeddings = torch.sparse.mm(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings) # (n_users + n_entities, out_dim)
        return embeddings

class KGAT_Encoder(nn.Module):
    def __init__(self, data, kg_loader, conf):
        super(KGAT_Encoder, self).__init__()
        self.data = data
        self.kg_loader = kg_loader
        self.config = conf
        
        args = self.config['KGAT']
        self.n_layer = int(args['n_layer'])
        self.embed_size = int(self.config['embedding.size'])
        self.relation_size = int(self.config['relation.size'])
        self.aggregator_type = args['aggregator.type']
        self.mess_dropout = args['mess_dropout']
        self.kg_lambda = float(args['kg_lambda'])
        self.reg_lambda = float(self.config['reg.lambda'])
        
        # Dimensions
        self.n_users = self.data.user_num
        self.n_items = self.data.item_num
        self.n_nodes = self.kg_loader.n_entities 
        self.n_relations = self.kg_loader.n_relations_total
        
        # Model Components
        self.entity_user_embed = nn.Embedding(self.n_nodes, self.embed_size)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_size)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_size, self.relation_size))
        
        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)
        
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layer):
            self.aggregator_layers.append(Aggregator(self.embed_size, self.embed_size, self.mess_dropout[k], self.aggregator_type))
            
        # Adjacency Matrix
        self.A_in = nn.Parameter(self.kg_loader.A_in, requires_grad=False)

    def update_attention_batch(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(r_mul_t * torch.tanh(r_mul_h + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            batch_v_list = self.update_attention_batch(batch_h_list, batch_t_list, r_idx)
            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse_coo_tensor(indices, values, torch.Size(shape))

        # Equation (5)
        # Use CPU for sparse softmax to avoid potential CUDA issues with sparse tensors in some versions
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)
        
    def calc_cf_embeddings(self):
        ego_embed = self.entity_user_embed.weight
        all_embed = [ego_embed]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1) # (n_users + n_entities, concat_dim)
        return all_embed

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        # user_ids: 0..U-1
        # item_ids: 0..I-1 -> Shift to U..U+I-1
        item_pos_ids = item_pos_ids + self.n_users
        item_neg_ids = item_neg_ids + self.n_users
        
        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[user_ids]
        item_pos_embed = all_embed[item_pos_ids]
        item_neg_embed = all_embed[item_neg_ids]

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)

        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.reg_lambda * l2_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)
        W_r = self.trans_M[r]

        h_embed = self.entity_user_embed(h)
        pos_t_embed = self.entity_user_embed(pos_t)
        neg_t_embed = self.entity_user_embed(neg_t)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_lambda * l2_loss
        return loss

    def predict_embedding(self):
        all_embed = self.calc_cf_embeddings()
        user_embed = all_embed[:self.n_users]
        item_embed = all_embed[self.n_users:self.n_users+self.n_items]
        return user_embed, item_embed

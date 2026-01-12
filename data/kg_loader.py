import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from collections import defaultdict
import random

class KGDataLoader:
    def __init__(self, conf, interaction_data):
        self.conf = conf
        self.data = interaction_data
        self.kg_file = conf['kg.data']
        self.laplacian_type = conf['KGAT']['adj.type']
        
        # Determine mapping
        self.n_users = self.data.user_num
        self.n_items = self.data.item_num
        
        # Load KG
        self.load_kg()
        self.process_data()
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def load_kg(self):
        # Load KG triples
        # Assuming format: head relation tail
        kg_data = []
        with open(self.kg_file, 'r') as f:
            for line in f:
                h, r, t = line.strip().split()
                kg_data.append((h, r, t))
        
        self.raw_kg_data = pd.DataFrame(kg_data, columns=['h', 'r', 't'])

    def process_data(self):
        # Map entities to IDs
        # Users: 0 ... n_users-1
        # Items: n_users ... n_users + n_items - 1
        # Other Entities: n_users + n_items ...
        
        self.entity_map = {} # raw -> id
        self.relation_map = {} # raw -> id
        
        # Initialize relation map: reserve 0 and 1 for UI interactions
        # Real KG relations start from 2
        next_relation_id = 2
        
        # Initialize entity map with items
        # Note: self.data.item is raw->internal_item_id (0..n_items-1)
        # We map them to n_users..n_users+n_items-1
        for raw_item, internal_id in self.data.item.items():
            self.entity_map[raw_item] = self.n_users + internal_id
            
        next_entity_id = self.n_users + self.n_items
        
        # Process KG triples
        processed_triples = []
        for idx, row in self.raw_kg_data.iterrows():
            h_raw, r_raw, t_raw = row['h'], row['r'], row['t']
            
            # Map H
            if h_raw in self.entity_map:
                h_id = self.entity_map[h_raw]
            else:
                h_id = next_entity_id
                self.entity_map[h_raw] = h_id
                next_entity_id += 1
                
            # Map T
            if t_raw in self.entity_map:
                t_id = self.entity_map[t_raw]
            else:
                t_id = next_entity_id
                self.entity_map[t_raw] = t_id
                next_entity_id += 1
                
            # Map R
            if r_raw in self.relation_map:
                r_id = self.relation_map[r_raw]
            else:
                r_id = next_relation_id
                self.relation_map[r_raw] = r_id
                next_relation_id += 1
            
            processed_triples.append([h_id, r_id, t_id])
            
        self.n_entities = next_entity_id # Total nodes (Users + Items + Others)
        self.n_relations = next_relation_id # Base relations (0, 1 + KG relations)
        self.n_kg_relations = next_relation_id - 2
        
        # Create DataFrames
        kg_df = pd.DataFrame(processed_triples, columns=['h', 'r', 't'])
        
        # Add inverse KG relations
        # Inverse relations: r + n_relations - 2 (since 0,1 are UI) -> No, simpler:
        # Let's say we have R relations (including 0,1).
        # We want to add inverse for KG relations.
        # KG relations are 2 ... R-1.
        # Inverse KG relations will be R ... R + (R-2) - 1.
        
        inverse_kg_df = kg_df.copy()
        inverse_kg_df = inverse_kg_df.rename(columns={'h': 't', 't': 'h'})
        # Offset inverse relations
        # Original: 2 -> Inverse: 2 + n_kg_relations
        # Original: R-1 -> Inverse: R-1 + n_kg_relations
        inverse_kg_df['r'] = inverse_kg_df['r'] + self.n_kg_relations
        
        self.n_relations_total = self.n_relations + self.n_kg_relations
        
        # Add CF interactions (User-Item)
        # Users: 0..n_users-1
        # Items: n_users..n_users+n_items-1
        # Relation 0: User -> Item
        # Relation 1: Item -> User
        
        cf_triples = []
        # self.data.training_data is list of [user_id, item_id, weight] (internal IDs)
        for u_id, i_id, _ in self.data.training_data: # internal IDs
             # u_id is already 0..n_users-1
             # i_id needs shift
             real_i_id = self.n_users + int(i_id)
             
             cf_triples.append([int(u_id), 0, real_i_id]) # User -> Item
             cf_triples.append([real_i_id, 1, int(u_id)]) # Item -> User
             
        cf_df = pd.DataFrame(cf_triples, columns=['h', 'r', 't'])
        
        # Combine all
        self.all_train_data = pd.concat([kg_df, inverse_kg_df, cf_df], ignore_index=True)
        
        # Create dictionaries for fast access
        self.train_kg_dict = defaultdict(list)
        self.train_relation_dict = defaultdict(list)
        
        for _, row in self.all_train_data.iterrows():
            h, r, t = int(row['h']), int(row['r']), int(row['t'])
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))
            
        # Store for KG training (triples only from KG + Inverse KG?)
        # Standard KGAT usually trains KG loss only on KG triples, not CF triples.
        # But attention update uses all.
        # Let's separate KG triples for KG loss.
        self.kg_triples = pd.concat([kg_df, inverse_kg_df], ignore_index=True)
        
    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_entities, self.n_entities))
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            with np.errstate(divide='ignore'):
                d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            with np.errstate(divide='ignore'):
                d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == 'symmetric':
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == 'random-walk':
            norm_lap_func = random_walk_norm_lap
        elif self.laplacian_type == 'softmax':
            norm_lap_func = symmetric_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape))
    
    def get_kg_data(self):
        # Return data needed for training
        h = torch.LongTensor(self.kg_triples['h'].values)
        r = torch.LongTensor(self.kg_triples['r'].values)
        t = torch.LongTensor(self.kg_triples['t'].values)
        return h, r, t

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, batch_size):
        kg_dict = self.train_kg_dict
        highest_neg_idx = self.n_entities
        exist_heads = list(kg_dict.keys())
        
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail


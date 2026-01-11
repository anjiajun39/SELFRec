import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import torch

class KGData:
    def __init__(self, conf, interaction_data):
        self.conf = conf
        self.interaction_data = interaction_data
        self.kg_file = conf['KG']['kg_file']
        
        # Mappings
        self.entity_id_map = {}  # original_id -> internal_id
        self.relation_id_map = {} # original_id -> internal_id
        self.id2entity = {}
        self.id2relation = {}
        
        # Initialize entity map with items from interaction data
        # Items are the first set of entities (0 to n_items-1)
        # interaction_data.item maps original_item_id -> internal_item_id
        # We want to keep this mapping consistent if possible.
        # But KG might have entities that are not items.
        # Let's align: Entity ID 0 to n_items-1 corresponds to Item ID 0 to n_items-1
        
        self.n_users = self.interaction_data.user_num
        self.n_items = self.interaction_data.item_num
        
        # Pre-populate entity map with items
        # self.interaction_data.id2item maps internal_item_id -> original_item_id
        for i in range(self.n_items):
            original_id = self.interaction_data.id2item[i]
            self.entity_id_map[original_id] = i
            self.id2entity[i] = original_id
            
        self.num_entities = self.n_items
        self.num_relations = 0
        
        self.kg_triplets = self.load_kg()
        self.kg_adj = self.build_kg_adj()
        
    def load_kg(self):
        triplets = []
        print(f"Loading KG from {self.kg_file}...")
        with open(self.kg_file, 'r') as f:
            lines = f.readlines()
            # Skip header if present
            if 'head_id' in lines[0]:
                lines = lines[1:]
                
            for line in lines:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                
                # Map entities
                if h not in self.entity_id_map:
                    self.entity_id_map[h] = self.num_entities
                    self.id2entity[self.num_entities] = h
                    self.num_entities += 1
                if t not in self.entity_id_map:
                    self.entity_id_map[t] = self.num_entities
                    self.id2entity[self.num_entities] = t
                    self.num_entities += 1
                    
                # Map relations
                if r not in self.relation_id_map:
                    self.relation_id_map[r] = self.num_relations
                    self.id2relation[self.num_relations] = r
                    self.num_relations += 1
                    
                h_id = self.entity_id_map[h]
                r_id = self.relation_id_map[r]
                t_id = self.entity_id_map[t]
                
                triplets.append((h_id, r_id, t_id))
                
        # Add inverse triplets
        extended_triplets = []
        for h, r, t in triplets:
            extended_triplets.append((h, r, t))
            extended_triplets.append((t, r + self.num_relations, h))
            
        print(f"KG loaded. Entities: {self.num_entities}, Relations: {self.num_relations}, Triplets: {len(extended_triplets)}")
        return np.array(extended_triplets)

    def build_kg_adj(self):
        # Build adjacency list for KG with inverse relations
        # return: dict {head: [(tail, relation), ...]}
        kg_adj = defaultdict(list)
        
        # kg_triplets now already contains inverse relations
        
        for h, r, t in self.kg_triplets:
            kg_adj[h].append((t, r))
            
        return kg_adj

    def get_ckg_adj(self):
        # Construct Collaborative Knowledge Graph (CKG) adjacency
        # Nodes: Users (0 ~ n_users-1) + Entities (n_users ~ n_users + num_entities - 1)
        # Relations: 
        #   - Interaction (User <-> Item): relation_id = 2 * num_relations
        #   - KG relations: 0 ~ 2*num_relations-1
        
        adj = defaultdict(list)
        
        # Add User-Item interactions
        # User ID in CKG: user_id
        # Item ID in CKG: n_users + item_id (since Item ID is same as Entity ID 0~n_items-1)
        
        interact_rel = 2 * self.num_relations
        
        for user_token, items_dict in self.interaction_data.training_set_u.items():
            user_node = self.interaction_data.user[user_token]
            for item_token in items_dict:
                item_id = self.interaction_data.item[item_token]
                item_node = self.n_users + item_id
                adj[user_node].append((item_node, interact_rel))
                adj[item_node].append((user_node, interact_rel))
        
        # Add KG triples
        for h, r, t in self.kg_triplets:
            h_node = self.n_users + h
            t_node = self.n_users + t
            
            adj[h_node].append((t_node, r))
            
        return adj
        
    def get_kg_adj_mat(self):
        # Create sparse adjacency matrix for propagation if needed
        # Or return the triplet tensor
        pass
    
    def generate_kg_batch(self, batch_size):
        # For KG loss training
        indices = np.random.choice(len(self.kg_triplets), batch_size, replace=False)
        return self.kg_triplets[indices]

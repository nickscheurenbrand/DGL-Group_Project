import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm

from dual_graph_utils import create_dual_graph, create_dual_graph_feature_matrix

    
class TargetEdgeInitializer(nn.Module):
    """TransformerConv based taregt edge initialization model"""
    def __init__(self, n_source_nodes, n_target_nodes, num_heads=4, edge_dim=1, 
                 dropout=0.2, beta=False):
        super().__init__()
        assert n_target_nodes % num_heads == 0

        self.conv1 = TransformerConv(n_source_nodes, n_target_nodes // num_heads, 
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(n_target_nodes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.pos_edge_index, data.edge_attr

        # Update node embeddings for the source graph
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        node_embeddings = F.relu(x)

        # Super-resolve source graph using matrix multiplication
        xt = node_embeddings.T @ node_embeddings    # xt will be treated as the adjacency matrix of the target graph

        # Normalize values to be between [0, 1]
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

        # Fetch and reshape upper triangular part to get dual graph's node feature matrix
        ut_mask = torch.triu(torch.ones_like(xt), diagonal=1).bool()
        x = torch.masked_select(xt, ut_mask).view(-1, 1)

        return x, node_embeddings
    
class DynamicStructureLearner(nn.Module):
    """Learns the dual graph structure dynamically using pure PyTorch (no torch-cluster)."""
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def forward(self, x, batch=None):
        # x shape: (N, D)
        
        # 1. Compute pairwise Euclidean distances
        # cdist is optimized and native to PyTorch
        dist = torch.cdist(x, x, p=2) 
        
        # 2. Find k nearest neighbors
        # We select k+1 because the closest node is always itself (dist=0)
        # largest=False gives us the smallest distances
        _, indices = dist.topk(self.k + 1, largest=False)
        
        # 3. Construct edge_index
        # indices shape: (N, k+1). 
        # Column 0 is the node itself, so we take columns 1 to k+1
        neighbor_indices = indices[:, 1:] # (N, k)
        
        num_nodes = x.size(0)
        
        # Create source indices: [0, 0, ..., 1, 1, ...]
        source_indices = torch.arange(num_nodes, device=x.device).unsqueeze(1).expand(num_nodes, self.k)
        
        # Flatten to create standard PyG edge_index format (2, Num_Edges)
        # Direction: Source Node -> Neighbor Node
        edge_index = torch.stack([source_indices.flatten(), neighbor_indices.flatten()], dim=0)
        
        return edge_index

class DualGraphLearner(nn.Module):
    """Update node features of the dual graph"""
    def __init__(self, in_dim, node_embed_dim, out_dim=1, num_heads=1, 
                 dropout=0.2, k_neighbors=10, beta=False):
        super().__init__()

        # Structure Learner
        self.structure_learner = DynamicStructureLearner(k=k_neighbors)

        # Fusion Layer 
        # Input dim increases because we concat: Edge_Feat + Node_u + Node_v
        fusion_input_dim = in_dim + (2 * node_embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, in_dim * 4), # Project to higher dim
            nn.ReLU(),
            nn.Linear(in_dim * 4, in_dim) # Project back to working dim
        )


        # Here, we override num_heads to be 1 since we output scalar primal edge weights
        # In future work, we can experiment with multiple heads
        self.conv1 = TransformerConv(in_dim, out_dim, 
                                     heads=num_heads,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(out_dim)

    def forward(self, dual_x, node_embeddings, node_indices_u, node_indices_v):
        # 1. Gather node features for the start (u) and end (v) of every edge
        node_feat_u = node_embeddings[node_indices_u] # Shape: (N_edges, D_node)
        node_feat_v = node_embeddings[node_indices_v] # Shape: (N_edges, D_node)
        
        # 2. Concatenate: [Edge_Weight || Node_u || Node_v]
        fused_x = torch.cat([dual_x, node_feat_u, node_feat_v], dim=1)
        
        # 3. Compress back to dual feature dimension
        x = self.fusion_mlp(fused_x)

        # Dynamic Structure Learning
        # Learn connectivity based on the FUSED features
        dynamic_edge_index = self.structure_learner(x)

        # Standard GNN update
        x = self.conv1(x, dynamic_edge_index)
        x = self.bn1(x)
        xt = F.relu(x)

        # Normalize
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        xt = (xt - xt_min) / (xt_max - xt_min + 1e-8)

        return xt
    

class STPGSR(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_source_nodes = config.dataset.n_source_nodes
        n_target_nodes = config.dataset.n_target_nodes
        node_hidden_dim = n_source_nodes

        self.target_edge_initializer = TargetEdgeInitializer(
                            n_source_nodes,
                            n_target_nodes,
                            num_heads=config.model.target_edge_initializer.num_heads,
                            edge_dim=config.model.target_edge_initializer.edge_dim,
                            dropout=config.model.target_edge_initializer.dropout,
                            beta=config.model.target_edge_initializer.beta
        )
        self.dual_learner = DualGraphLearner(
                            in_dim=config.model.dual_learner.in_dim,
                            node_embed_dim=node_hidden_dim,
                            out_dim=config.model.dual_learner.out_dim,
                            num_heads=config.model.dual_learner.num_heads,
                            dropout=config.model.dual_learner.dropout,
                            beta=config.model.dual_learner.beta,
                            k_neighbors=10
        )

       # We need to know which node indices (u, v) correspond to the flattened dual array
        # This replaces the static dual_edge_index creation
        mask_indices = torch.triu_indices(n_target_nodes, n_target_nodes, offset=1)
        self.register_buffer('u_indices', mask_indices[0])
        self.register_buffer('v_indices', mask_indices[1])


    def forward(self, source_pyg, target_mat):
        # 1. Initialize target edges AND get node embeddings
        target_edge_init, node_embeddings = self.target_edge_initializer(source_pyg)
        
        # 2. Update target edges with Fusion and Dynamic Structure
        # We pass the node embeddings and the indices to map them
        node_embeddings = node_embeddings.t()

        dual_pred_x = self.dual_learner(
            target_edge_init, 
            node_embeddings, 
            self.u_indices, 
            self.v_indices
        )

        # Convert target matrix into edge feature matrix
        dual_target_x = create_dual_graph_feature_matrix(target_mat)

        return dual_pred_x, dual_target_x

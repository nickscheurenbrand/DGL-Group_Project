import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv


class BiSR(nn.Module):
    """
    Bipartite Graph Super-Resolution (Bi-SR) Layer.
    Translates N low-resolution nodes to M high-resolution nodes using a learnable bipartite adjacency matrix.
    """

    def __init__(self, in_nodes=160, out_nodes=268, feature_dim=160, hidden_dim=64):
        super(BiSR, self).__init__()

        # The learnable bipartite matrix B (Shape: N x M)
        # Initializes the topological "bridges" between the 160 LR regions and 268 HR regions.
        self.B = nn.Parameter(torch.randn(in_nodes, out_nodes) * 0.01)

        # The feature transformation weight matrix W
        self.W = nn.Linear(feature_dim, hidden_dim)

    def forward(self, x_l):
        """
        Args:
            x_l: Low-resolution node features (Batch, in_nodes, feature_dim).
                 Since X_L = A_L, feature_dim is usually equal to in_nodes (160).
        Returns:
            x_h: High-resolution node embeddings (Batch, out_nodes, hidden_dim).
        """
        # 1. Transform LR node features: X_L * W
        x_transformed = self.W(x_l)  # Shape: (Batch, 160, hidden_dim)

        # 2. Bipartite Message Passing: B^T * (X_L * W)
        # We use torch.matmul to broadcast across the batch dimension.
        # self.B.t() shape is (268, 160).
        x_h = torch.matmul(self.B.t(), x_transformed)  # Shape: (Batch, 268, hidden_dim)

        return F.relu(x_h)


class DEFEND(nn.Module):
    """
    Dual Graphs for Edge Feature Learning and Detection (DEFEND).
    Takes HR node embeddings and computes the exact edge weights using dual-graph principles.
    """

    def __init__(self, num_hr_nodes=268, hidden_dim=64):
        super(DEFEND, self).__init__()
        self.num_nodes = num_hr_nodes
        self.num_edges = int(num_hr_nodes * (num_hr_nodes - 1) / 2)  # 35,778

        # Add layers to enrich the node embeddings before edge regression by having a 2-hop neighborhood aggregation.
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Edge feature predictor (acting as the message passing on the dual graph)
        # It takes the concatenated features of the two nodes forming an edge.
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Outputs a single continuous edge weight
        )

    def forward(self, x_h):
        """
        Args:
            x_h: High-resolution node embeddings from Bi-SR (Batch, 268, hidden_dim).
        Returns:
            adj_hr: The predicted symmetric HR adjacency matrix (Batch, 268, 268).
        """
        batch_size = x_h.shape[0]

        # Generate indices for the upper triangle (the 35,778 edges)
        row_indices, col_indices = torch.triu_indices(
            self.num_nodes, self.num_nodes, offset=1
        )

        
        # Enrich node features with a 2-hop neighborhood aggregation before edge regression
        x_h_enriched = self.node_mlp(x_h)  # Shape: (Batch, 268, hidden_dim)
        enriched_source_features = x_h_enriched[:, row_indices, :]
        enriched_target_features = x_h_enriched[:, col_indices, :]

        # Extract the node features for the source (row) and target (col) of each edge
        source_features = x_h[:, row_indices, :]  # Shape: (Batch, 35778, hidden_dim)
        target_features = x_h[:, col_indices, :]  # Shape: (Batch, 35778, hidden_dim)

        node_min = torch.minimum(source_features, target_features)
        node_max = torch.maximum(source_features, target_features)
        
        # Concatenate node features to form the initial "dual graph nodes" (the edges)
        edge_features = torch.cat(
            [node_min, node_max, enriched_source_features, enriched_target_features], dim=-1
        )  # Shape: (Batch, 35778, hidden_dim * 4)

        # Regress the exact connection weights
        predicted_edge_weights = self.edge_mlp(edge_features).squeeze(
            -1
        )  # Shape: (Batch, 35778)

        # Reconstruct the symmetric batch of adjacency matrices
        adj_hr = torch.zeros(
            (batch_size, self.num_nodes, self.num_nodes), device=x_h.device
        )
        adj_hr[:, row_indices, col_indices] = predicted_edge_weights
        adj_hr[:, col_indices, row_indices] = (
            predicted_edge_weights  # Make it symmetric
        )

        return adj_hr


class BrainGraphSuperResolutionModel(nn.Module):
    """
    The complete pipeline combining GraphGCN, Bi-SR, and DEFEND.
    """

    def __init__(self, in_nodes=160, out_nodes=268, hidden_dim=64, k_threshold=0.6):
        super(BrainGraphSuperResolutionModel, self).__init__()

        # Threshold for binarizing the adjacency matrix to delineate strict neighborhoods
        self.k_threshold = k_threshold
        
        
        # 2-Layer Dense GraphSAGE Architecture
        # Provides 2-hop neighborhood aggregation while concatenating central node features,
        # which is robust for structure-based embedding.
        self.sage1 = DenseSAGEConv(in_channels=in_nodes, out_channels=128)
        self.sage2 = DenseSAGEConv(in_channels=128, out_channels=in_nodes)

        # Topological feature extraction
        self.bi_sr = BiSR(
            in_nodes=in_nodes,
            out_nodes=out_nodes,
            feature_dim=in_nodes,
            hidden_dim=hidden_dim,
        )
        self.defend = DEFEND(num_hr_nodes=out_nodes, hidden_dim=hidden_dim)

    def forward(self, adj_lr):
        """
        Args:
            adj_lr: Batch of low-resolution adjacency matrices (Batch, 160, 160).
        Returns:
            adj_hr: Batch of super-resolved high-resolution adjacency matrices (Batch, 268, 268).
        """
        # 1. Prepare GNN Inputs
        # The node features are the original un-thresholded continuous connectivity profiles.
        node_features = adj_lr

        # The topological routing graph is strictly binary based on the weight threshold k.
        binary_adj = (adj_lr > self.k_threshold).float()

        # 2. GraphSAGE Forward Pass
        x_hidden = F.relu(self.sage1(node_features, binary_adj))
        x_gnn = F.relu(self.sage2(x_hidden, binary_adj))

        # We add a residual connection combining the transformed structural embedding
        # with the original continuous row profile
        x_lr_enriched = x_gnn + adj_lr

        # 3. Map topology to target HR dimensions via Bipartite formulation
        hr_node_embeddings = self.bi_sr(x_lr_enriched)

        # 2. Predict precise edge weights via Dual Graph feature learning
        hr_adjacency_matrix = self.defend(hr_node_embeddings)

        return hr_adjacency_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Edge feature predictor (acting as the message passing on the dual graph)
        # It takes the concatenated features of the two nodes forming an edge.
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
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

        # Extract the node features for the source (row) and target (col) of each edge
        source_features = x_h[:, row_indices, :]  # Shape: (Batch, 35778, hidden_dim)
        target_features = x_h[:, col_indices, :]  # Shape: (Batch, 35778, hidden_dim)

        # Concatenate node features to form the initial "dual graph nodes" (the edges)
        edge_features = torch.cat(
            [source_features, target_features], dim=-1
        )  # Shape: (Batch, 35778, hidden_dim * 2)

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
    The complete pipeline combining Bi-SR and DEFEND.
    """

    def __init__(self, in_nodes=160, out_nodes=268, hidden_dim=64):
        super(BrainGraphSuperResolutionModel, self).__init__()

        # Topological feature extraction (Identity assumption X = A handled by feature_dim=in_nodes)
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
        # 1. Map topology to target HR dimensions via Bipartite formulation
        hr_node_embeddings = self.bi_sr(adj_lr)

        # 2. Predict precise edge weights via Dual Graph feature learning
        hr_adjacency_matrix = self.defend(hr_node_embeddings)

        return hr_adjacency_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv


class BiSR(nn.Module):
    """
    Bipartite Graph Super-Resolution (Bi-SR) Layer with low-rank factorization.
    Translates N low-resolution nodes to M high-resolution nodes using a learnable
    low-rank bipartite adjacency matrix B = U @ V.
    """

    def __init__(
        self, in_nodes=160, out_nodes=268, feature_dim=160, hidden_dim=16, rank=8
    ):
        super(BiSR, self).__init__()

        # Low-rank factorization of the bipartite matrix B = U @ V
        # Instead of a free (160, 268) matrix with 42,880 params,
        # we use U (160, rank) + V (rank, 268) = rank * (160 + 268) params
        self.U = nn.Parameter(torch.randn(in_nodes, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(rank, out_nodes) * 0.01)

        # The feature transformation weight matrix W
        self.W = nn.Linear(feature_dim, hidden_dim)

    def forward(self, x_l):
        """
        Args:
            x_l: Low-resolution node features (Batch, in_nodes, feature_dim).
        Returns:
            x_h: High-resolution node embeddings (Batch, out_nodes, hidden_dim).
        """
        # 1. Transform LR node features: X_L * W
        x_transformed = self.W(x_l)  # Shape: (Batch, 160, hidden_dim)

        # 2. Reconstruct bipartite matrix from low-rank factors
        B = self.U @ self.V  # Shape: (160, 268)

        # 3. Bipartite Message Passing: B^T * (X_L * W)
        x_h = torch.matmul(B.t(), x_transformed)  # Shape: (Batch, 268, hidden_dim)

        return F.relu(x_h)


class EdgeMLP(nn.Module):
    """Configurable edge weight predictor.

    num_layers=1 (default): single linear layer (most regularized).
    num_layers>1: Linear→ReLU→Dropout stack, with the hidden size halving each
    layer, finishing with a Linear→scalar output.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.5):
        super(EdgeMLP, self).__init__()

        if num_layers == 1:
            self.net = nn.Linear(hidden_dim * 2, 1)
        else:
            layers = []
            in_dim = hidden_dim * 2
            # Build (num_layers - 1) hidden layers, halving dimension each time
            for i in range(num_layers - 1):
                out_dim = max(in_dim // 2, 1)
                layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

    def forward(self, edge_features):
        return self.net(edge_features)


class DEFEND(nn.Module):
    """
    Dual Graphs for Edge Feature Learning and Detection (DEFEND).
    Takes HR node embeddings and computes the exact edge weights.
    """

    def __init__(self, num_hr_nodes=268, hidden_dim=16, edge_mlp_layers=1, dropout=0.5):
        super(DEFEND, self).__init__()
        self.num_nodes = num_hr_nodes
        self.num_edges = int(num_hr_nodes * (num_hr_nodes - 1) / 2)  # 35,778

        self.edge_mlp = EdgeMLP(hidden_dim, num_layers=edge_mlp_layers, dropout=dropout)

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

        # Symmetric edge features: invariant to node ordering (undirected graph)
        # sum and abs-diff are both commutative: f(i,j) == f(j,i)
        edge_sum = (
            source_features + target_features
        )  # Shape: (Batch, 35778, hidden_dim)
        edge_diff = torch.abs(
            source_features - target_features
        )  # Shape: (Batch, 35778, hidden_dim)
        edge_features = torch.cat(
            [edge_sum, edge_diff], dim=-1
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
    The complete pipeline combining GraphGCN, Bi-SR, and DEFEND.
    Includes mean-prior residual learning for regularization.
    """

    def __init__(
        self,
        in_nodes=160,
        out_nodes=268,
        hidden_dim=16,
        gcn_layers=1,
        hidden_dim_gcn=32,
        k_threshold=0.8,
        bisr_rank=8,
        dropout=0.5,
        edge_mlp_layers=1,
    ):
        super(BrainGraphSuperResolutionModel, self).__init__()

        # Threshold for binarizing the adjacency matrix
        self.k_threshold = k_threshold
        self.dropout = dropout

        # Mean HR prior buffer (set from training data before training begins)
        self.register_buffer("hr_mean", torch.zeros(out_nodes, out_nodes))

        self.num_gcn_layers = gcn_layers

        # GCN layers (reduced capacity)
        if gcn_layers == 0:
            # Skip GCN entirely — pass raw LR adjacency directly to BiSR
            self.gcn_layers = nn.ModuleList()
        elif gcn_layers == 1:
            # Single hidden layer: in_nodes -> hidden_dim_gcn -> in_nodes
            self.gcn_layers = nn.ModuleList(
                [
                    DenseSAGEConv(in_channels=in_nodes, out_channels=hidden_dim_gcn),
                    DenseSAGEConv(in_channels=hidden_dim_gcn, out_channels=in_nodes),
                ]
            )
        else:
            # gcn_layers hidden layers + 1 output layer
            layers = [DenseSAGEConv(in_channels=in_nodes, out_channels=hidden_dim_gcn)]
            for _ in range(gcn_layers - 1):
                layers.append(
                    DenseSAGEConv(
                        in_channels=hidden_dim_gcn, out_channels=hidden_dim_gcn
                    )
                )
            layers.append(
                DenseSAGEConv(in_channels=hidden_dim_gcn, out_channels=in_nodes)
            )
            self.gcn_layers = nn.ModuleList(layers)

        # Topological feature extraction with low-rank BiSR
        self.bi_sr = BiSR(
            in_nodes=in_nodes,
            out_nodes=out_nodes,
            feature_dim=in_nodes,
            hidden_dim=hidden_dim,
            rank=bisr_rank,
        )
        self.defend = DEFEND(
            num_hr_nodes=out_nodes,
            hidden_dim=hidden_dim,
            edge_mlp_layers=edge_mlp_layers,
            dropout=dropout,
        )

    def forward(self, adj_lr):
        """
        Args:
            adj_lr: Batch of low-resolution adjacency matrices (Batch, 160, 160).
        Returns:
            adj_hr: Batch of super-resolved high-resolution adjacency matrices (Batch, 268, 268).
        """
        # 1. Prepare GNN Inputs
        node_features = adj_lr
        binary_adj = (adj_lr > self.k_threshold).float()

        # 2. GraphSAGE Forward Pass (skip entirely if gcn_layers=0)
        if self.num_gcn_layers > 0:
            for layer in self.gcn_layers:
                node_features = F.relu(layer(node_features, binary_adj))
                node_features = F.dropout(
                    node_features, p=self.dropout, training=self.training
                )
            # Residual connection with original input
            x_lr_enriched = node_features + adj_lr
        else:
            # No GCN: pass raw adjacency directly
            x_lr_enriched = adj_lr

        # 3. Map topology to target HR dimensions via Bipartite formulation
        hr_node_embeddings = self.bi_sr(x_lr_enriched)

        # 4. Predict residual edge weights via Dual Graph feature learning
        hr_residual = self.defend(hr_node_embeddings)

        # 5. Add the population mean prior
        hr_adjacency_matrix = self.hr_mean + hr_residual

        return hr_adjacency_matrix

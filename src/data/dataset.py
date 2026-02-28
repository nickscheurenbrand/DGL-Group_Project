import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch_geometric.data import Data
from torch.utils.data import Dataset

from src.data.MatrixVectorizer import MatrixVectorizer


def matrix_to_pyg_data(matrix: np.ndarray) -> Data:
    """Converts a dense adjacency matrix to a PyTorch Geometric Data object."""
    tensor_matrix = torch.tensor(matrix, dtype=torch.float32)
    edge_index = tensor_matrix.nonzero(as_tuple=False).t().contiguous()
    edge_weight = tensor_matrix[edge_index[0], edge_index[1]]

    # Set node features X equal to the adjacency matrix A
    x = tensor_matrix.clone()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)


class BrainGraphDataset(Dataset):
    def __init__(self, lr_file: str, hr_file: str) -> None:
        self.lr_df = pd.read_csv(lr_file)
        self.hr_df = pd.read_csv(hr_file)
        
        assert len(self.lr_df) == len(self.hr_df), "LR and HR datasets must match."
        self.n_samples: int = len(self.lr_df)
        
        self.lr_nodes: int = 160
        self.hr_nodes: int = 268
        
        # Precompute and cache all data in RAM
        self.cached_data: List[Tuple[Data, Data]] = self._preprocess()

    def _preprocess(self) -> List[Tuple[Data, Data]]:
        """Precomputes and formats the vectors into PyG data objects."""
        cached_data: List[Tuple[Data, Data]] = []
        for idx in range(self.n_samples):
            lr_vector = self.lr_df.iloc[idx].values
            hr_vector = self.hr_df.iloc[idx].values
            
            lr_matrix = MatrixVectorizer.anti_vectorize(lr_vector, self.lr_nodes, include_diagonal=False)
            hr_matrix = MatrixVectorizer.anti_vectorize(hr_vector, self.hr_nodes, include_diagonal=False)
            
            lr_data = matrix_to_pyg_data(lr_matrix)
            hr_data = matrix_to_pyg_data(hr_matrix)
            
            cached_data.append((lr_data, hr_data))
            
        return cached_data

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Data, Data]:
        # O(1) memory retrieval during training
        return self.cached_data[idx]

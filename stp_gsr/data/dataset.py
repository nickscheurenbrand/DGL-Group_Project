import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data

from data.MatrixVectorizer import MatrixVectorizer 


def matrix_to_pyg_data(matrix: np.ndarray) -> Data:
    """Converts a dense adjacency matrix to a PyTorch Geometric Data object."""
    tensor_matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Extract edges
    edge_index = tensor_matrix.nonzero(as_tuple=False).t().contiguous()
    
    # Extract continuous weights and reshape to (num_edges, 1) for PyG compatibility
    edge_weight = tensor_matrix[edge_index[0], edge_index[1]].unsqueeze(1)

    # Set node features X equal to the adjacency matrix A
    x = tensor_matrix.clone()

    # The STP-GSR model requires 'pos_edge_index' alongside the standard 'edge_index'
    return Data(
        x=x, 
        edge_index=edge_index, 
        pos_edge_index=edge_index, 
        edge_attr=edge_weight
    )


class BrainGraphDataset:
    def __init__(self, lr_train_path: str, hr_train_path: str, lr_test_path: str):
        """
        Initialises the dataset, loads the CSV files, and precomputes the anti-vectorised graphs.
        """
        self.lr_train_path = Path(lr_train_path)
        self.hr_train_path = Path(hr_train_path)
        self.lr_test_path = Path(lr_test_path) 
        
        # Dimensions for Low-Resolution and High-Resolution brain graphs
        self.lr_nodes = 160 
        self.hr_nodes = 268 
        
        # Load and process the training data
        print("Loading and anti-vectorising training data...")
        self.train_source_data, self.train_target_data = self._process_train_data()
        
        # Optionally load and process the test data (for final submission)
        if self.lr_test_path:
            print("Loading and anti-vectorising test data...")
            self.test_source_data = self._process_test_data()

    def _load_vectors(self, path: Path) -> np.ndarray:
        """Loads a vectorised connectivity CSV into a float32 array."""
        return pd.read_csv(path, header=None).values.astype(np.float32)

    def _antivectorise_batch(self, vectors: np.ndarray, matrix_size: int) -> np.ndarray:
        """Converts (N, vec_len) vectorised rows into (N, matrix_size, matrix_size) adjacency matrices."""
        n_samples = vectors.shape[0]
        matrices = np.zeros((n_samples, matrix_size, matrix_size), dtype=np.float32)
        
        for i in tqdm(range(n_samples), desc=f"Anti-vectorising {matrix_size}x{matrix_size}"):
            # Note: We must call the original module's method spelling 'anti_vectorize' to avoid an AttributeError
            matrices[i] = MatrixVectorizer.anti_vectorize(vectors[i], matrix_size)
            
        return matrices

    def _process_train_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Processes the training sets into the dictionaries required by train.py."""
        lr_train_vec = self._load_vectors(self.lr_train_path)
        hr_train_vec = self._load_vectors(self.hr_train_path)
        
        assert len(lr_train_vec) == len(hr_train_vec), "LR and HR training datasets must have the same number of samples."
        
        lr_train_mats = self._antivectorise_batch(lr_train_vec, self.lr_nodes)
        hr_train_mats = self._antivectorise_batch(hr_train_vec, self.hr_nodes)
        
        source_data = []
        target_data = []
        
        for i in range(len(lr_train_mats)):
            lr_pyg_data = matrix_to_pyg_data(lr_train_mats[i])
            
            source_data.append({
                'pyg': lr_pyg_data,
                'mat': torch.tensor(lr_train_mats[i], dtype=torch.float32)
            })
            
            target_data.append({
                'mat': torch.tensor(hr_train_mats[i], dtype=torch.float32)
            })
            
        return source_data, target_data

    def _process_test_data(self) -> List[Dict]:
        """Processes the Kaggle test set into the dictionaries required for model inference."""
        lr_test_vec = self._load_vectors(self.lr_test_path)
        lr_test_mats = self._antivectorise_batch(lr_test_vec, self.lr_nodes)
        
        test_source_data = []
        for i in range(len(lr_test_mats)):
            lr_pyg_data = matrix_to_pyg_data(lr_test_mats[i])
            
            test_source_data.append({
                'pyg': lr_pyg_data,
                'mat': torch.tensor(lr_test_mats[i], dtype=torch.float32)
            })
            
        return test_source_data

    def get_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Returns the fully processed training source and target dataset lists."""
        return self.train_source_data, self.train_target_data
        
    def get_test_data(self) -> List[Dict]:
        """Returns the fully processed test source dataset list."""
        if not self.lr_test_path:
            raise ValueError("Test path was not provided during initialisation.")
        return self.test_source_data
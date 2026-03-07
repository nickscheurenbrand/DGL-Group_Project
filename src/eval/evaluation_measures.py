from typing import Dict, List, Union

import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from src.data.MatrixVectorizer import MatrixVectorizer


def compute_evaluation_measures(
    pred_matrices: Union[np.ndarray, List[np.ndarray], List[list]],
    gt_matrices: Union[np.ndarray, List[np.ndarray], List[list]],
) -> Dict[str, float]:
    """
    Computes various evaluation measures between predicted and ground truth matrices.

    Args:
        pred_matrices: Predicted adjacency matrices, typically shape (num_samples, num_nodes, num_nodes)
                       or a list of such matrices.
        gt_matrices: Ground truth adjacency matrices, typically shape (num_samples, num_nodes, num_nodes)
                     or a list of such matrices.

    Returns:
        A dictionary containing computed evaluation metrics:
        MAE, PCC, Jensen-Shannon Distance, Cosine Similarity,
        and Average MAEs for degree, betweenness, eigenvector, and PageRank centralities.
    """
    pred_matrices = np.asarray(pred_matrices)
    gt_matrices = np.asarray(gt_matrices)

    num_test_samples = pred_matrices.shape[0]

    # Initialize lists to store MAEs for each centrality measure
    mae_dc: List[float] = []
    mae_bc: List[float] = []
    mae_ec: List[float] = []
    mae_pc: List[float] = []

    pred_1d_list: List[np.ndarray] = []
    gt_1d_list: List[np.ndarray] = []

    # Iterate over each test sample
    for i in range(num_test_samples):
        # Convert adjacency matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(pred_matrices[i], edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_matrices[i], edge_attr="weight")

        # Compute centrality measures
        pred_dc = nx.degree_centrality(pred_graph)
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight", max_iter=1000)
        pred_pc = nx.pagerank(pred_graph, weight="weight")

        gt_dc = nx.degree_centrality(gt_graph)
        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight", max_iter=1000)
        gt_pc = nx.pagerank(gt_graph, weight="weight")

        # Compute MAEs for centralities
        mae_dc.append(mean_absolute_error(list(pred_dc.values()), list(gt_dc.values())))
        mae_bc.append(mean_absolute_error(list(pred_bc.values()), list(gt_bc.values())))
        mae_ec.append(mean_absolute_error(list(pred_ec.values()), list(gt_ec.values())))
        mae_pc.append(mean_absolute_error(list(pred_pc.values()), list(gt_pc.values())))

        # Vectorize matrices
        pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrices[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(gt_matrices[i]))

    # Compute metrics for vector representation of matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    mae = float(mean_absolute_error(pred_1d, gt_1d))
    pcc = float(pearsonr(pred_1d, gt_1d)[0])
    js_dis = float(jensenshannon(pred_1d, gt_1d))
    cosine_similarity = float(
        np.dot(pred_1d, gt_1d) / (np.linalg.norm(pred_1d) * np.linalg.norm(gt_1d) + 1e-12)
    )

    avg_mae_dc = float(np.mean(mae_dc))
    avg_mae_bc = float(np.mean(mae_bc))
    avg_mae_ec = float(np.mean(mae_ec))
    avg_mae_pc = float(np.mean(mae_pc))

    metrics = {
        "MAE": mae,
        "PCC": pcc,
        "Jensen-Shannon Distance": js_dis,
        "Cosine Similarity": cosine_similarity,
        "Average MAE degree centrality": avg_mae_dc,
        "Average MAE betweenness centrality": avg_mae_bc,
        "Average MAE eigenvector centrality": avg_mae_ec,
        "Average MAE PageRank centrality": avg_mae_pc,
    }

    print(f"MAE: {mae}")
    print(f"PCC: {pcc}")
    print(f"Jensen-Shannon Distance: {js_dis}")
    print(f"Cosine Similarity: {cosine_similarity}")
    print(f"Average MAE degree centrality: {avg_mae_dc}")
    print(f"Average MAE betweenness centrality: {avg_mae_bc}")
    print(f"Average MAE eigenvector centrality: {avg_mae_ec}")
    print(f"Average MAE PageRank centrality: {avg_mae_pc}")

    return metrics

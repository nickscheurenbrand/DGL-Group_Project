import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.MatrixVectorizer import MatrixVectorizer
from src.dual.model import BrainGraphSuperResolutionModel


def get_args():
    parser = argparse.ArgumentParser(description="Predict with Graph Super-Resolution Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="generated_data",
        help="Directory containing the test data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save the predictions",
    )
    parser.add_argument(
        "--k_threshold",
        type=float,
        default=0.6,
        help="Threshold for binarizing adjacency matrix",
    )
    parser.add_argument(
        "--gcn_layers",
        type=int,
        default=2,
        help="Number of GCN layers",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension of the model",
    )
    parser.add_argument(
        "--hidden_dim_gcn",
        type=int,
        default=128,
        help="Hidden dimension of the GCN layers",
    )
    return parser.parse_args()


def predict():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate and Load Model
    print(f"Loading model from {args.model_path}...")
    model = BrainGraphSuperResolutionModel(
        in_nodes=160,
        out_nodes=268,
        hidden_dim=args.hidden_dim,
        k_threshold=args.k_threshold,
        gcn_layers=args.gcn_layers,
        hidden_dim_gcn=args.hidden_dim_gcn,
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Load Test Data
    test_file_path = os.path.join(args.data_dir, "lr_test.csv")
    print(f"Loading test data from {test_file_path}...")
    df_test = pd.read_csv(test_file_path)

    n_samples = len(df_test)
    all_predictions = []

    print("Running predictions...")
    with torch.no_grad():
        for idx in tqdm(range(n_samples), desc="Predicting"):
            vector = df_test.iloc[idx].values
            matrix = MatrixVectorizer.anti_vectorize(
                vector, 160, include_diagonal=False
            )

            # Predict
            tensor_matrix = (
                torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
            )
            pred_matrix = model(tensor_matrix)
            pred_matrix = pred_matrix.squeeze(0).cpu().numpy()

            # Vectorize the predicted HR graph
            pred_vector = MatrixVectorizer.vectorize(
                pred_matrix, include_diagonal=False
            )
            all_predictions.append(pred_vector)

    # Flatten the predictions into a single 1D array
    flat_predictions = np.concatenate(all_predictions)

    # Save to submission
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create DataFrame with ID and Predicted to match sample submission (4,007,136 rows)
    ids = np.arange(1, len(flat_predictions) + 1)
    submission_df = pd.DataFrame({"ID": ids, "Predicted": flat_predictions})

    output_path = os.path.join(args.output_dir, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")


if __name__ == "__main__":
    predict()

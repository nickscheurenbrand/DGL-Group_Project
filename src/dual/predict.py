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
        help="Path to the trained model checkpoint (used when --ensemble_seeds is not set)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory containing model checkpoints (used with --ensemble_seeds)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="best_model",
        help="Base model name for ensemble: looks for <model_name>_seed<N>.pth",
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
        default=0.8,
        help="Threshold for binarizing adjacency matrix",
    )
    parser.add_argument(
        "--gcn_layers",
        type=int,
        default=1,
        help="Number of GCN layers (0 = skip GCN)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=16,
        help="Hidden dimension of the model",
    )
    parser.add_argument(
        "--hidden_dim_gcn",
        type=int,
        default=32,
        help="Hidden dimension of the GCN layers",
    )
    parser.add_argument(
        "--bisr_rank",
        type=int,
        default=8,
        help="Rank for low-rank BiSR factorization",
    )
    parser.add_argument(
        "--edge_mlp_layers",
        type=int,
        default=1,
        help="Number of layers in EdgeMLP (1 = single linear; >1 adds hidden layers halving in size)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Clip predictions to [0, clip]. Set to 0 to disable. Default: 1.0",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Mean-blending weight: final = alpha * model_pred + (1-alpha) * hr_mean. "
        "1.0 = no blending (default), 0.9 = 10%% towards mean, etc.",
    )
    parser.add_argument(
        "--ensemble_seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of seeds for multi-seed ensemble. Looks for <model_dir>/<model_name>_seed<N>.pth. "
        "Example: --ensemble_seeds 42 123 456",
    )
    return parser.parse_args()


def build_model(args, device):
    """Instantiate an empty model with the given architecture args."""
    return BrainGraphSuperResolutionModel(
        in_nodes=160,
        out_nodes=268,
        hidden_dim=args.hidden_dim,
        k_threshold=args.k_threshold,
        gcn_layers=args.gcn_layers,
        hidden_dim_gcn=args.hidden_dim_gcn,
        bisr_rank=args.bisr_rank,
        dropout=args.dropout,
        edge_mlp_layers=args.edge_mlp_layers,
    ).to(device)


def load_model(args, device, checkpoint_path):
    """Load a model from a checkpoint path."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    model = build_model(args, device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def predict():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model(s) ---
    if args.ensemble_seeds:
        print(f"Ensemble mode: loading {len(args.ensemble_seeds)} models...")
        models = []
        for seed in args.ensemble_seeds:
            ckpt = os.path.join(args.model_dir, f"{args.model_name}_seed{seed}.pth")
            print(f"  Loading {ckpt}")
            models.append(load_model(args, device, ckpt))
        print(f"Ensemble of {len(models)} models loaded.")
    else:
        print(f"Loading single model from {args.model_path}...")
        models = [load_model(args, device, args.model_path)]

    # Extract hr_mean from first model for blending
    hr_mean = models[0].hr_mean  # Shape: (268, 268), on device

    # --- Load Test Data ---
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
            tensor_matrix = (
                torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
            )

            # Average predictions across ensemble
            pred_matrix = sum(m(tensor_matrix) for m in models) / len(models)
            pred_matrix = pred_matrix.squeeze(0)  # (268, 268)

            # Alpha mean-blending
            if args.alpha < 1.0:
                pred_matrix = args.alpha * pred_matrix + (1.0 - args.alpha) * hr_mean

            pred_matrix = pred_matrix.cpu().numpy()

            # Clip to [0, clip]
            if args.clip > 0:
                pred_matrix = np.clip(pred_matrix, 0.0, args.clip)

            # Vectorize the predicted HR graph
            pred_vector = MatrixVectorizer.vectorize(
                pred_matrix, include_diagonal=False
            )
            all_predictions.append(pred_vector)

    # Flatten predictions
    flat_predictions = np.concatenate(all_predictions)

    # Save submission
    os.makedirs(args.output_dir, exist_ok=True)
    ids = np.arange(1, len(flat_predictions) + 1)
    submission_df = pd.DataFrame({"ID": ids, "Predicted": flat_predictions})

    output_path = os.path.join(args.output_dir, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")


if __name__ == "__main__":
    predict()

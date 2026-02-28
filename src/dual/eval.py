import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm

from src.data.data import get_dataloaders
from src.dual.model import BrainGraphSuperResolutionModel


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Graph Super-Resolution Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="generated_data",
        help="Directory containing the training/validation data",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.pth",
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation DataLoader"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation plots",
    )
    parser.add_argument(
        "--k_threshold",
        type=float,
        default=0.6,
        help="Threshold for binarizing adjacency matrix",
    )
    return parser.parse_args()


def evaluate():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Prepare Data
    print("Loading data...")
    # We only need the validation loader for evaluation
    _, val_loader = get_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, shuffle_train=False
    )

    # 2. Instantiate and Load Model
    print(f"Loading model from {args.model_path}...")
    model = BrainGraphSuperResolutionModel(
        in_nodes=160, out_nodes=268, hidden_dim=64, k_threshold=args.k_threshold
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    # Map location handles loading a model saved on GPU to CPU if necessary
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # Metrics Tracking
    all_preds = []
    all_targets = []
    sample_pred, sample_target = None, None

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    total_mse = 0.0
    total_mae = 0.0

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            lr_data, hr_data = batch

            num_graphs = lr_data.num_graphs
            lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)
            hr_adj_target = hr_data.x.view(num_graphs, 268, 268).to(device)

            hr_adj_pred = model(lr_adj)

            # Accumulate Metrics
            total_mse += criterion_mse(hr_adj_pred, hr_adj_target).item() * num_graphs
            total_mae += criterion_mae(hr_adj_pred, hr_adj_target).item() * num_graphs

            # Store predictions for global statistics (flattened edges)
            # Use upper triangle indices if we want to ignore symmetric duplicates & diagonals
            # For simplicity, we can just flatten the entire matrix since it's symmetric
            pred_flat = hr_adj_pred.cpu().view(-1).numpy()
            target_flat = hr_adj_target.cpu().view(-1).numpy()

            all_preds.append(pred_flat)
            all_targets.append(target_flat)

            # Keep one sample for heatmap visualization
            if sample_pred is None:
                sample_pred = hr_adj_pred[0].cpu().numpy()
                sample_target = hr_adj_target[0].cpu().numpy()

    # Finalize Metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    global_mse = total_mse / len(val_loader.dataset)
    global_mae = total_mae / len(val_loader.dataset)
    pearson_corr, _ = pearsonr(all_preds, all_targets)

    print("\n" + "=" * 40)
    print("🎯 EVALUATION METRICS")
    print("=" * 40)
    print(f"Mean Squared Error (MSE):   {global_mse:.6f}")
    print(f"Mean Absolute Error (MAE):  {global_mae:.6f}")
    print(f"Pearson Correlation (r):    {pearson_corr:.6f}")
    print("=" * 40)

    # 3. Generate Visualizations
    print("\nGenerating evaluation plots...")

    # Plot 1: Scatter Plot (Density / Hexbin)
    plt.figure(figsize=(8, 8))
    plt.hexbin(
        all_targets, all_preds, gridsize=50, cmap="viridis", mincnt=1, bins="log"
    )
    plt.plot(
        [all_targets.min(), all_targets.max()],
        [all_targets.min(), all_targets.max()],
        "r--",
        lw=2,
        label="Ideal (y=x)",
    )
    plt.colorbar(label="log10(count)")
    plt.xlabel("True Edge Weights")
    plt.ylabel("Predicted Edge Weights")
    plt.title("Predicted vs True HR Edge Weights")
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(args.output_dir, "scatter_plot.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # Plot 2: Residuals Histogram
    residuals = all_preds - all_targets
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=100, kde=True, color="blue")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Error (Predicted - True)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Errors (Residuals)")
    plt.tight_layout()
    error_hist_path = os.path.join(args.output_dir, "error_histogram.png")
    plt.savefig(error_hist_path, dpi=300)
    plt.close()

    # Plot 3: Example Heatmaps
    if sample_pred is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Common value range for Ground Truth and Prediction to make colors comparable
        vmin = min(sample_target.min(), sample_pred.min())
        vmax = max(sample_target.max(), sample_pred.max())

        sns.heatmap(
            sample_target,
            ax=axes[0],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        axes[0].set_title("Ground Truth HR Adjacency Matrix")

        sns.heatmap(
            sample_pred,
            ax=axes[1],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        axes[1].set_title("Predicted HR Adjacency Matrix")

        absolute_error = np.abs(sample_pred - sample_target)
        sns.heatmap(
            absolute_error,
            ax=axes[2],
            cmap="Reds",
            cbar=True,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        axes[2].set_title("Absolute Error Heatmap")

        plt.tight_layout()
        heatmap_path = os.path.join(args.output_dir, "sample_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

    print(f"✅ Visualizations saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    evaluate()

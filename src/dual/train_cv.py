import argparse
import copy
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



from src.data.dataset import BrainGraphDataset
from src.data.MatrixVectorizer import MatrixVectorizer
from src.dual.model import BrainGraphSuperResolutionModel
from src.eval.evaluation_measures import compute_evaluation_measures
from src.utils import set_random_seed


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Graph Super-Resolution Model with 3-fold CV"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="generated_data",
        help="Directory containing the training data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save the checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="best_model.pth",
        help="Base model filename for each fold",
    )
    parser.add_argument(
        "--pred_output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save fold validation predictions",
    )
    parser.add_argument(
        "--k_threshold",
        type=float,
        default=0.8,
        help="Threshold for binarizing adjacency matrix",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for the learning rate scheduler",
    )
    parser.add_argument(
        "--gcn_layers", type=int, default=1, help="Number of GCN layers"
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
        help="Number of layers in EdgeMLP (1 = single linear, most regularized; "
        ">1 adds hidden layers that halve in size)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.01,
        help="Standard deviation of Gaussian noise added to LR inputs during training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility and KFold split",
    )
    return parser.parse_args()


def compute_mean_hr_for_indices(
    data_dir: str, sample_indices: np.ndarray, num_nodes: int = 268
) -> torch.Tensor:
    """Compute the mean HR adjacency matrix over the provided sample indices."""
    hr_df = pd.read_csv(os.path.join(data_dir, "hr_train.csv"))
    mean_vector = hr_df.iloc[sample_indices].values.mean(axis=0)
    mean_matrix = MatrixVectorizer.anti_vectorize(
        mean_vector, num_nodes, include_diagonal=False
    )
    return torch.tensor(mean_matrix, dtype=torch.float32)


def save_fold_val_predictions(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    fold_num: int,
) -> None:
    """Save flattened validation predictions in the same format as predict.py."""
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            lr_data, _ = batch
            num_graphs = lr_data.num_graphs
            lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)

            pred_batch = model(lr_adj).cpu().numpy()  # (B, 268, 268)
            for pred_matrix in pred_batch:
                pred_vector = MatrixVectorizer.vectorize(
                    pred_matrix, include_diagonal=False
                )
                all_predictions.append(pred_vector)

    if len(all_predictions) == 0:
        flat_predictions = np.array([])
    else:
        flat_predictions = np.concatenate(all_predictions)

    ids = np.arange(1, len(flat_predictions) + 1)
    submission_df = pd.DataFrame({"ID": ids, "Predicted": flat_predictions})

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_fold_{fold_num}.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Saved validation predictions to {output_path}")


def train_cv():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.pred_output_dir, exist_ok=True)

    set_random_seed(args.seed)
    print(f"Random seed: {args.seed}")

    print("Loading full training dataset...")
    full_dataset = BrainGraphDataset(
        lr_file=os.path.join(args.data_dir, "lr_train.csv"),
        hr_file=os.path.join(args.data_dir, "hr_train.csv"),
    )
    num_samples = len(full_dataset)
    all_indices = np.arange(num_samples)

    kf = KFold(n_splits=3, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(all_indices), start=1):
        print(f"\n===== Fold {fold_num}/3 =====")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        print("Computing fold-specific mean HR prior...")
        hr_mean = compute_mean_hr_for_indices(args.data_dir, train_idx)

        model = BrainGraphSuperResolutionModel(
            in_nodes=160,
            out_nodes=268,
            hidden_dim=args.hidden_dim,
            gcn_layers=args.gcn_layers,
            hidden_dim_gcn=args.hidden_dim_gcn,
            k_threshold=args.k_threshold,
            bisr_rank=args.bisr_rank,
            dropout=args.dropout,
            edge_mlp_layers=args.edge_mlp_layers,
        ).to(device)
        model.hr_mean.copy_(hr_mean.to(device))

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Fold {fold_num} trainable parameters: {total_params:,}")

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.patience
        )

        best_val_loss = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0

            pbar = tqdm(
                train_loader, desc=f"Fold {fold_num} Epoch {epoch}/{args.epochs} [Train]"
            )
            for batch in pbar:
                lr_data, hr_data = batch

                num_graphs = lr_data.num_graphs
                lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)
                hr_adj_target = hr_data.x.view(num_graphs, 268, 268).to(device)

                if args.noise_std > 0:
                    lr_adj = lr_adj + torch.randn_like(lr_adj) * args.noise_std

                optimizer.zero_grad()
                hr_adj_pred = model(lr_adj)
                loss = criterion(hr_adj_pred, hr_adj_target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * num_graphs
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    lr_data, hr_data = batch

                    num_graphs = lr_data.num_graphs
                    lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)
                    hr_adj_target = hr_data.x.view(num_graphs, 268, 268).to(device)

                    hr_adj_pred = model(lr_adj)
                    loss = criterion(hr_adj_pred, hr_adj_target)
                    val_loss += loss.item() * num_graphs

            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)

            print(
                f"Fold {fold_num} Epoch {epoch} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())

                base, ext = os.path.splitext(args.model_name)
                fold_model_name = f"{base}_fold{fold_num}{ext or '.pth'}"
                checkpoint_path = os.path.join(args.save_dir, fold_model_name)
                torch.save(best_state_dict, checkpoint_path)
                print(f"--> Fold {fold_num}: saved new best model to {checkpoint_path}")

        model.load_state_dict(best_state_dict)

        pred_matrices = []
        gt_matrices = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                lr_data, hr_data = batch
                num_graphs = lr_data.num_graphs
                lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)
                hr_adj_target = hr_data.x.view(num_graphs, 268, 268)

                hr_adj_pred = model(lr_adj).cpu()

                pred_matrices.extend(hr_adj_pred.numpy())
                gt_matrices.extend(hr_adj_target.numpy())

        metrics_raw = compute_evaluation_measures(pred_matrices, gt_matrices)
        metrics_fold = {
            "Fold": fold_num,
            "MAE": metrics_raw["MAE"],
            "PCC": metrics_raw["PCC"],
            "JSD": metrics_raw["Jensen-Shannon Distance"],
            "Cosine Sim": metrics_raw["Cosine Similarity"],
            "MAE (DC)": metrics_raw["Average MAE degree centrality"],
            "MAE (PC)": metrics_raw["Average MAE PageRank centrality"],
            "MAE (EC)": metrics_raw["Average MAE eigenvector centrality"],
            "MAE (BC)": metrics_raw["Average MAE betweenness centrality"],
        }
        fold_metrics.append(metrics_fold)

        print(
            f"Fold {fold_num} metrics | "
            f"MAE: {metrics_fold['MAE']:.6f} | "
            f"PCC: {metrics_fold['PCC']:.6f} | "
            f"JSD: {metrics_fold['JSD']:.6f} | "
            f"CosineSim: {metrics_fold['Cosine Sim']:.6f} | "
            f"MAE(DC): {metrics_fold['MAE (DC)']:.6f} | "
            f"MAE(PC): {metrics_fold['MAE (PC)']:.6f} | "
            f"MAE(EC): {metrics_fold['MAE (EC)']:.6f} | "
            f"MAE(BC): {metrics_fold['MAE (BC)']:.6f}"
        )

        save_fold_val_predictions(
            model=model,
            val_loader=val_loader,
            device=device,
            output_dir=args.pred_output_dir,
            fold_num=fold_num,
        )

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_csv_path = os.path.join(args.pred_output_dir, "cv_fold_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    metric_names = [
        "MAE",
        "PCC",
        "JSD",
        "Cosine Sim",
        "MAE (DC)",
        "MAE (PC)",
        "MAE (EC)",
        "MAE (BC)",
    ]
    metric_means = metrics_df[metric_names].mean()
    metric_stds = metrics_df[metric_names].std(ddof=1)

    summary_df = pd.DataFrame(
        {
            "Metric": metric_names,
            "Mean": [metric_means[m] for m in metric_names],
            "Std": [metric_stds[m] for m in metric_names],
        }
    )
    summary_csv_path = os.path.join(args.pred_output_dir, "cv_metrics_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    bar_colors = [
        "#ff6666",
        "#66b266",
        "#6666e6",
        "#4f9ec4",
        "#b57edc",
        "#f0c060",
        "#61d4d4",
        "#62ea62",
    ]

    def add_value_labels(ax, bars):
        y_min, y_max = ax.get_ylim()
        offset = 0.01 * (y_max - y_min if y_max > y_min else 1.0)
        for bar in bars:
            h = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2.0
            if h >= 0:
                ax.text(x, h + offset, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(x, h - offset, f"{h:.3f}", ha="center", va="top", fontsize=8)

    for i, row in metrics_df.iterrows():
        ax = axes[i]
        values = [row[m] for m in metric_names]
        bars = ax.bar(metric_names, values, color=bar_colors)
        ax.set_title(f"Fold {int(row['Fold'])}")
        ax.tick_params(axis="x", rotation=45)
        add_value_labels(ax, bars)

    avg_ax = axes[3]
    avg_values = [metric_means[m] for m in metric_names]
    avg_err = [metric_stds[m] for m in metric_names]
    avg_bars = avg_ax.bar(metric_names, avg_values, yerr=avg_err, capsize=4, color=bar_colors)
    avg_ax.set_title("Avg. Across Folds")
    avg_ax.tick_params(axis="x", rotation=45)
    add_value_labels(avg_ax, avg_bars)

    plt.tight_layout()
    plot_path = os.path.join(args.pred_output_dir, "cv_metrics_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Saved fold metrics to {metrics_csv_path}")
    print(f"Saved metrics summary to {summary_csv_path}")
    print(f"Saved CV metrics plot to {plot_path}")

    print("3-fold cross-validation training complete!")


if __name__ == "__main__":
    train_cv()

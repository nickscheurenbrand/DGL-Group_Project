import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from src.data.data import get_dataloaders
from src.data.MatrixVectorizer import MatrixVectorizer
from src.dual.model import BrainGraphSuperResolutionModel
from src.utils import set_random_seed


def get_args():
    parser = argparse.ArgumentParser(description="Train Graph Super-Resolution Model")
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
        help="Name of the model file",
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
        "--train_all",
        action="store_true",
        default=False,
        help="Train on ALL 167 samples from lr_train.csv/hr_train.csv (no val split). "
        "Use this for the final Kaggle submission model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. When provided, the checkpoint is saved as <model_name>_seed<N>.pth "
        "to support multi-seed ensemble training.",
    )
    return parser.parse_args()


def compute_mean_hr(data_dir: str, num_nodes: int = 268) -> torch.Tensor:
    """Compute the mean HR adjacency matrix across all training samples."""
    hr_df = pd.read_csv(os.path.join(data_dir, "hr_train.csv"))
    mean_vector = hr_df.values.mean(axis=0)
    mean_matrix = MatrixVectorizer.anti_vectorize(
        mean_vector, num_nodes, include_diagonal=False
    )
    return torch.tensor(mean_matrix, dtype=torch.float32)


def train():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else 42
    set_random_seed(seed)
    print(f"Random seed: {seed}")

    # Determine checkpoint name: append _seed<N> when a seed is explicitly provided
    if args.seed is not None:
        base, ext = os.path.splitext(args.model_name)
        model_filename = f"{base}_seed{args.seed}{ext or '.pth'}"
    else:
        model_filename = args.model_name

    # 1. Prepare Data
    print("Loading data...")
    if args.train_all:
        print("[train_all] Loading all 167 samples from lr_train.csv / hr_train.csv")
        from src.data.dataset import BrainGraphDataset
        from torch_geometric.loader import DataLoader as PygDataLoader
        from pathlib import Path

        data_path = Path(args.data_dir)
        full_dataset = BrainGraphDataset(
            lr_file=str(data_path / "lr_train.csv"),
            hr_file=str(data_path / "hr_train.csv"),
        )
        train_loader = PygDataLoader(
            full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = None  # no validation when training on all samples
    else:
        train_loader, val_loader, _ = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
        )

    # 2. Compute Mean HR Prior
    print("Computing mean HR prior from training data...")
    hr_mean = compute_mean_hr(args.data_dir)

    # 3. Instantiate Model
    print("Initializing model...")
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

    # Set the mean HR prior
    model.hr_mean.copy_(hr_mean.to(device))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # 4. Define Loss Function and Optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )

    # 5. Training Loop
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            lr_data, hr_data = batch

            num_graphs = lr_data.num_graphs
            lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)
            hr_adj_target = hr_data.x.view(num_graphs, 268, 268).to(device)

            # Input noise augmentation
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

        # Log the mean-only baseline MAE vs model MAE to see residual contribution
        model.eval()
        with torch.no_grad():
            mean_only_mae = 0.0
            residual_norm = 0.0
            for batch in train_loader:
                lr_data, hr_data = batch
                num_graphs = lr_data.num_graphs
                hr_adj_target = hr_data.x.view(num_graphs, 268, 268).to(device)
                lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)
                hr_adj_pred = model(lr_adj)
                # MAE if we just predicted the mean (no residual)
                mean_only_mae += (
                    torch.mean(
                        torch.abs(model.hr_mean.unsqueeze(0) - hr_adj_target)
                    ).item()
                    * num_graphs
                )
                # How large is the residual the model learned?
                residual_norm += (
                    torch.mean(torch.abs(hr_adj_pred - model.hr_mean)).item()
                    * num_graphs
                )
            mean_only_mae /= len(train_loader.dataset)
            residual_norm /= len(train_loader.dataset)

        # Validation Phase
        model.eval()

        if val_loader is not None:
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

            # Step the scheduler based on validation loss
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Mean-only: {mean_only_mae:.6f} | Residual norm: {residual_norm:.6f}"
            )

            # Save best model (val-guided)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.save_dir, model_filename)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"--> Saved new best model to {checkpoint_path}")
        else:
            # train_all mode: step scheduler on train loss, save every epoch
            scheduler.step(train_loss)
            print(
                f"Epoch {epoch} | Train: {train_loss:.6f} | "
                f"Mean-only: {mean_only_mae:.6f} | Residual norm: {residual_norm:.6f}"
            )
            checkpoint_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), checkpoint_path)

    print("Training Complete!")


if __name__ == "__main__":
    train()

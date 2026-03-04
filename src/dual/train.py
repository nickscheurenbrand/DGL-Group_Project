import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.data.data import get_dataloaders
from src.dual.model import BrainGraphSuperResolutionModel
from src.utils import set_random_seed


def get_args():
    parser = argparse.ArgumentParser(description="Train Graph Super-Resolution Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../cw2/dgl-2026-brain-graph-super-resolution-challenge/",
        help="Directory containing the training data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/dual",
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
        default=0.9,
        help="Threshold for binarising adjacency matrix",
    )
    return parser.parse_args()


def train():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    set_random_seed(42)

    # 1. Prepare Data
    print("Loading data...")
    n_splits = 5
    all_folds = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_splits=n_splits
    )

    # Save hyperparameters for reproducibility
    with open(os.path.join(args.save_dir, "hyperparameters.txt"), "w") as f:
        f.write(f"Data Directory: {args.data_dir}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"K Threshold: {args.k_threshold}\n")

    # Track evaluation metrics across all folds
    fold_metrics = []

    for fold_idx, (train_loader, val_loader) in enumerate(all_folds):
        print(f"\n======================================")
        print(f"--- Fold {fold_idx + 1}/{len(all_folds)} ---")
        print(f"======================================")

        # Re-initialise model for each fold to prevent data leakage
        model = BrainGraphSuperResolutionModel(
            in_nodes=160, out_nodes=268, hidden_dim=64, k_threshold=args.k_threshold
        ).to(device)

        # 3. Define Loss Function and Optimiser
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        
        # 4. Training Loop
        best_val_loss = float("inf")
        
        # Determine specific checkpoint path for this fold
        base_name, ext = os.path.splitext(args.model_name)
        checkpoint_path = os.path.join(args.save_dir, f"{base_name}_fold_{fold_idx + 1}{ext}")

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

                optimizer.zero_grad()
                hr_adj_pred = model(lr_adj)
                loss = criterion(hr_adj_pred, hr_adj_target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * num_graphs
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(train_loader.dataset)

            # Validation Phase
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
            print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Save Best Model Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"--> Saved new best model for Fold {fold_idx + 1}")

        # ---------------------------------------------------------
        # 5. FOLD EVALUATION
        # ---------------------------------------------------------
        print(f"\nEvaluating best model for Fold {fold_idx + 1}...")
        
        # Load the best weights we just saved for this fold
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        fold_mse = 0.0
        fold_mae = 0.0

        with torch.no_grad():
            for batch in val_loader:
                lr_data, hr_data = batch
                num_graphs = lr_data.num_graphs
                lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)
                hr_adj_target = hr_data.x.view(num_graphs, 268, 268).to(device)

                hr_adj_pred = model(lr_adj)
                
                # Calculate metrics
                fold_mse += F.mse_loss(hr_adj_pred, hr_adj_target).item() * num_graphs
                fold_mae += F.l1_loss(hr_adj_pred, hr_adj_target).item() * num_graphs

        # Average metrics over the validation set
        fold_mse /= len(val_loader.dataset)
        fold_mae /= len(val_loader.dataset)
        
        fold_metrics.append({"mse": fold_mse, "mae": fold_mae})
        print(f"Fold {fold_idx + 1} Results - MSE: {fold_mse:.6f} | MAE: {fold_mae:.6f}\n")

    # ---------------------------------------------------------
    # 6. FINAL CROSS-VALIDATION SUMMARY
    # ---------------------------------------------------------
    print("\n======================================")
    print("      CROSS-VALIDATION COMPLETE")
    print("======================================")
    
    avg_mse = sum(m["mse"] for m in fold_metrics) / n_splits
    avg_mae = sum(m["mae"] for m in fold_metrics) / n_splits
    
    for i, m in enumerate(fold_metrics):
        print(f"Fold {i + 1}: MSE = {m['mse']:.6f}, MAE = {m['mae']:.6f}")
        
    print("--------------------------------------")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average MAE: {avg_mae:.6f}")
    
    # Save the summary to a text file
    with open(os.path.join(args.save_dir, "cv_evaluation_summary.txt"), "w") as f:
        f.write("Cross-Validation Evaluation Summary\n")
        f.write("======================================\n")
        for i, m in enumerate(fold_metrics):
            f.write(f"Fold {i + 1}: MSE = {m['mse']:.6f}, MAE = {m['mae']:.6f}\n")
        f.write("--------------------------------------\n")
        f.write(f"Average MSE: {avg_mse:.6f}\n")
        f.write(f"Average MAE: {avg_mae:.6f}\n")


if __name__ == "__main__":
    train()
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.data import get_dataloaders
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
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
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
        default=0.6,
        help="Threshold for binarizing adjacency matrix",
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
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # 2. Instantiate Model
    print("Initializing model...")
    # BiSR layer maps from 160 input nodes to 268 output nodes
    model = BrainGraphSuperResolutionModel(
        in_nodes=160, out_nodes=268, hidden_dim=64, k_threshold=args.k_threshold
    ).to(device)

    # 3. Define Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # The dataloader yields a Tuple of [LR Data, HR Data]
            lr_data, hr_data = batch

            # The nodes features 'x' store the dense matrix, flattened across the batch.
            # Shape of lr_data.x is (Batch * 160, 160).
            # We reshape it back to (Batch, 160, 160).
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

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.save_dir, args.model_name)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--> Saved new best model to {checkpoint_path}")

    print("Training Complete!")


if __name__ == "__main__":
    train()

import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data.dataset import BrainGraphDataset
from src.data.MatrixVectorizer import MatrixVectorizer
from src.eval.evaluation_measures import compute_evaluation_measures
from src.utils import set_random_seed
from src.sgc.model import SGC

# ==========================================
# 1. Baseline SGC Model Definition
# ==========================================
class BaselineSGC(nn.Module):
    """
    Naive Simple Graph Convolution (SGC) for Brain Graph Super-Resolution.
    Takes a (B, 160, 160) dense matrix and outputs a symmetric (B, 268, 268) matrix.
    """
    def __init__(self, in_nodes=160, out_nodes=268, K=2):
        super(BaselineSGC, self).__init__()
        self.K = K
        self.out_nodes = out_nodes
        
        # Linear projection from the flattened LR graph to the vectorised HR graph
        hr_vector_size = (out_nodes * (out_nodes - 1)) // 2  # 35778
        self.fc = nn.Linear(in_nodes * in_nodes, hr_vector_size)

    def forward(self, A):
        # A: (B, 160, 160)
        
        # 1. Symmetric Normalisation
        I = torch.eye(A.size(-1), device=A.device)
        A_tilde = A + I
        D = A_tilde.sum(dim=-1) + 1e-5
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt_diag = torch.diag_embed(D_inv_sqrt)
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt_diag, A_tilde), D_inv_sqrt_diag)
        
        # 2. Linear K-hop aggregation (using A as both topology and features)
        H = A
        for _ in range(self.K):
            H = torch.bmm(A_norm, H)
            
        # 3. Flatten and project to HR vector size
        H_flat = H.view(H.size(0), -1)
        out_vec = self.fc(H_flat)
        out_vec = F.relu(out_vec) # Ensure no negative edge weights
        
        # 4. Differentiable reconstruction of the symmetric (B, 268, 268) matrix
        b = out_vec.shape[0]
        mat = torch.zeros(b, self.out_nodes, self.out_nodes, device=out_vec.device)
        idx = torch.triu_indices(self.out_nodes, self.out_nodes, offset=1)
        
        # Fill the upper triangle and mirror it
        mat[:, idx[0], idx[1]] = out_vec
        mat = mat + mat.transpose(1, 2)
        
        return mat


def get_args():
    parser = argparse.ArgumentParser(description="Train Baseline SGC Model with 3-fold CV")
    parser.add_argument("--data_dir", type=str, default="../cw2/dgl-2026-brain-graph-super-resolution-challenge", help="Directory containing the training data")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="/vol/gpudata/trm25-dgl/results_sgc", help="Directory to save the checkpoints")
    parser.add_argument("--model_name", type=str, default="best_model.pth", help="Base model filename for each fold")
    parser.add_argument("--pred_output_dir", type=str, default="/vol/gpudata/trm25-dgl/results_sgc", help="Directory to save fold validation predictions")
    parser.add_argument("--patience", type=int, default=20, help="Patience for the learning rate scheduler")
    parser.add_argument("--K", type=int, default=2, help="Number of propagation steps for SGC")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Standard deviation of Gaussian noise added to LR inputs during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def save_fold_val_predictions(model, val_loader, device, output_dir, fold_num):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            lr_data, _ = batch
            num_graphs = lr_data.num_graphs
            lr_adj = lr_data.x.view(num_graphs, 160, 160).to(device)

            pred_batch = model(lr_adj).cpu().numpy()  # (B, 268, 268)
            for pred_matrix in pred_batch:
                pred_vector = MatrixVectorizer.vectorize(pred_matrix, include_diagonal=False)
                all_predictions.append(pred_vector)

    flat_predictions = np.concatenate(all_predictions) if len(all_predictions) > 0 else np.array([])
    ids = np.arange(1, len(flat_predictions) + 1)
    submission_df = pd.DataFrame({"ID": ids, "Predicted": flat_predictions})

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_fold_{fold_num}.csv")
    submission_df.to_csv(output_path, index=False)


def train_cv():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.pred_output_dir, exist_ok=True)

    set_random_seed(args.seed)

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
        if fold_num == 1:
            continue
        print(f"\n===== Fold {fold_num}/3 =====")

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Swapped to the Baseline SGC model
        model = BaselineSGC(in_nodes=160, out_nodes=268, K=args.K).to(device)
        # model = SGC(input_dim=160*160, output_dim=(268*267)//2).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Fold {fold_num} trainable parameters: {total_params:,}")

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)

        best_val_loss = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch}/{args.epochs} [Train]")
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                base, ext = os.path.splitext(args.model_name)
                fold_model_name = f"{base}_fold{fold_num}{ext or '.pth'}"
                checkpoint_path = os.path.join(args.save_dir, fold_model_name)
                torch.save(best_state_dict, checkpoint_path)

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
        print(f"Fold {fold_num} MAE: {metrics_fold['MAE']:.6f} | PCC: {metrics_fold['PCC']:.6f}")

        save_fold_val_predictions(model, val_loader, device, args.pred_output_dir, fold_num)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_csv_path = os.path.join(args.pred_output_dir, "cv_fold_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    print("3-fold cross-validation training complete!")

if __name__ == "__main__":
    train_cv()
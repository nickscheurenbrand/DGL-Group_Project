import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.MatrixVectorizer import MatrixVectorizer
from src.dual.model import BrainGraphSuperResolutionModel


def expected_vector_length(num_nodes: int) -> int:
    return (num_nodes * (num_nodes - 1)) // 2

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate test-set predictions and save Kaggle-style submission CSV"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="../cw2/dgl-2026-brain-graph-super-resolution-challenge/lr_test.csv",
        help="Path to LR test CSV",
    )
    parser.add_argument(
        "--sample_submission",
        type=str,
        default="../cw2/dgl-2026-brain-graph-super-resolution-challenge/sample_submission.csv",
        help="Path to sample submission CSV template",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/dual/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/dual/submission.csv",
        help="Path to write predictions CSV",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument(
        "--k_threshold",
        type=float,
        default=0.6,
        help="Threshold for model adjacency binarization",
    )
    parser.add_argument("--in_nodes", type=int, default=160, help="LR graph size")
    parser.add_argument("--out_nodes", type=int, default=268, help="HR graph size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Model hidden dimension")
    return parser.parse_args()


def load_lr_test_vectors(test_file: Path, in_nodes: int) -> np.ndarray:
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    lr_df = pd.read_csv(test_file)
    expected_len = expected_vector_length(in_nodes)

    if lr_df.shape[1] == expected_len + 1:
        lr_df = lr_df.iloc[:, 1:]

    if lr_df.shape[1] != expected_len:
        raise ValueError(
            f"Unexpected lr_test column count: got {lr_df.shape[1]}, expected {expected_len} "
            f"(or {expected_len + 1} with an ID column)"
        )

    return lr_df.values.astype(np.float32)


def predict_hr_vectors(
    model: torch.nn.Module,
    lr_vectors: np.ndarray,
    in_nodes: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = []

    for start in tqdm(range(0, len(lr_vectors), batch_size), desc="Predicting"):
        end = min(start + batch_size, len(lr_vectors))
        batch_vectors = lr_vectors[start:end]

        batch_mats = np.stack(
            [
                MatrixVectorizer.anti_vectorize(v, in_nodes, include_diagonal=False)
                for v in batch_vectors
            ],
            axis=0,
        ).astype(np.float32)

        lr_adj = torch.from_numpy(batch_mats).to(device)

        with torch.no_grad():
            hr_adj_pred = model(lr_adj).cpu().numpy()

        pred_vectors = np.stack(
            [
                MatrixVectorizer.vectorize(m, include_diagonal=False)
                for m in hr_adj_pred
            ],
            axis=0,
        )
        predictions.append(pred_vectors)

    return np.concatenate(predictions, axis=0)


def format_submission(
    pred_vectors: np.ndarray, sample_submission_path: Path
) -> pd.DataFrame:
    flat_preds = pd.DataFrame(pred_vectors).to_numpy().flatten()

    if not sample_submission_path.exists():
        return pd.DataFrame({"target": flat_preds})

    template = pd.read_csv(sample_submission_path)

    if len(template) != len(flat_preds):
        raise ValueError(
            f"Row mismatch with sample_submission: template has {len(template)}, "
            f"flattened predictions have {len(flat_preds)}"
        )

    if template.shape[1] == 1:
        col = template.columns[0]
        return pd.DataFrame({col: flat_preds})

    out = template.copy()
    out.iloc[:, -1] = flat_preds
    return out


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_file = Path(args.test_file)
    sample_submission = Path(args.sample_submission)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print("Loading checkpoint...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)


    model = BrainGraphSuperResolutionModel(
        in_nodes=args.in_nodes,
        out_nodes=args.out_nodes,
        hidden_dim=args.hidden_dim,
        k_threshold=args.k_threshold,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loading test vectors...")
    lr_vectors = load_lr_test_vectors(test_file, args.in_nodes)

    print("Running inference...")
    pred_vectors = predict_hr_vectors(
        model=model,
        lr_vectors=lr_vectors,
        in_nodes=args.in_nodes,
        batch_size=args.batch_size,
        device=device,
    )

    print("Formatting submission...")
    submission_df = format_submission(pred_vectors, sample_submission)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

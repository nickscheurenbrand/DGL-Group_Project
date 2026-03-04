import torch
from pathlib import Path
from typing import Tuple
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from src.data.dataset import BrainGraphDataset

def get_dataloaders(
    data_dir: str,
    lr_file: str = "lr_train.csv",
    hr_file: str = "hr_train.csv",
    batch_size: int = 32,
    n_splits: int = 5,
    seed: int = 42,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads data, instantiates the dataset, splits into train/val, and returns DataLoaders.

    Args:
        data_dir (str): Directory where the data files are located.
        lr_file (str, optional): Filename for the low-resolution data. Defaults to "lr_train.csv".
        hr_file (str, optional): Filename for the high-resolution data. Defaults to "hr_train.csv".
        batch_size (int, optional): Batch size. Defaults to 32.
        val_split (float, optional): Fraction of the data to use for validation. Defaults to 0.2.
        seed (int, optional): Random seed for the dataset split. Defaults to 42.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        shuffle_train (bool, optional): Whether to shuffle the training data. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader]: The training and validation loaders.
    """
    data_path = Path(data_dir)
    lr_path = data_path / lr_file
    hr_path = data_path / hr_file

    if not lr_path.exists():
        raise FileNotFoundError(f"Low-resolution file not found at: {lr_path}")
    if not hr_path.exists():
        raise FileNotFoundError(f"High-resolution file not found at: {hr_path}")

    # The dataset handles reading the CSVs and converting to PyTorch Geometric Data
    # Preprocessing (un-vectorizing, tensorizing) is done in BrainGraphDataset
    dataset = BrainGraphDataset(
        lr_file=str(lr_path),
        hr_file=str(hr_path)
    )

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_loaders = []

    # Iterate through the splits to create specific subsets and loaders for each fold
    for train_idx, val_idx in kfold.split(dataset):
        
        # Create PyTorch Subsets using the indices from scikit-learn
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        # Create DataLoaders for this specific fold
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        fold_loaders.append((train_loader, val_loader))

    return fold_loaders

if __name__ == "__main__":
    all_folds = get_dataloaders(
        data_dir="generated_data",
        lr_file="lr_train.csv",
        hr_file="hr_train.csv",
        batch_size=32,
        val_split=0.2,
        seed=42,
        num_workers=0,
        shuffle_train=True
    )

    fold_0_train, fold_0_val = all_folds[0]
    print("Fold 1 Train Batch LR:", next(iter(fold_0_train))[0])
    print("Fold 1 Train Batch HR:", next(iter(fold_0_train))[1])
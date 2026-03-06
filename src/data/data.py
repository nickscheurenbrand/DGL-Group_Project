import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from src.data.dataset import BrainGraphDataset


def prepare_data_splits(data_dir: str, seed: int = 42) -> None:
    """
    Checks if train/val/test split files exist. If not, reads the original
    train sets, splits them into 70% train, 15% validation, and 15% test,
    and saves them to disk to ensure the test set is separated preemptively.
    """
    data_path = Path(data_dir)
    files = [
        "lr_train_split.csv",
        "hr_train_split.csv",
        "lr_val_split.csv",
        "hr_val_split.csv",
        "lr_test_split.csv",
        "hr_test_split.csv",
    ]

    if all((data_path / f).exists() for f in files):
        return  # Data is already split

    print("Preemptively splitting data into train (70%), val (15%), test (15%)...")
    lr_df = pd.read_csv(data_path / "lr_train.csv")
    hr_df = pd.read_csv(data_path / "hr_train.csv")

    assert len(lr_df) == len(hr_df), "Mismatch in rows between LR and HR training data."

    indices = np.arange(len(lr_df))
    # Split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=seed)
    # Split temp: 50% val (15% overall), 50% test (15% overall)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)

    # Save train
    lr_df.iloc[train_idx].to_csv(data_path / "lr_train_split.csv", index=False)
    hr_df.iloc[train_idx].to_csv(data_path / "hr_train_split.csv", index=False)

    # Save val
    lr_df.iloc[val_idx].to_csv(data_path / "lr_val_split.csv", index=False)
    hr_df.iloc[val_idx].to_csv(data_path / "hr_val_split.csv", index=False)

    # Save test
    lr_df.iloc[test_idx].to_csv(data_path / "lr_test_split.csv", index=False)
    hr_df.iloc[test_idx].to_csv(data_path / "hr_test_split.csv", index=False)

    print("Data splitting complete.")


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Ensures data is split preemptively, instantiates the datasets for train/val/test,
    and returns DataLoaders.

    Args:
        data_dir (str): Directory where the data files are located.
        batch_size (int, optional): Batch size. Defaults to 32.
        seed (int, optional): Random seed for the dataset split. Defaults to 42.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        shuffle_train (bool, optional): Whether to shuffle the training data. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The train, val, and test loaders.
    """
    prepare_data_splits(data_dir, seed=seed)

    data_path = Path(data_dir)

    train_dataset = BrainGraphDataset(
        lr_file=str(data_path / "lr_train_split.csv"),
        hr_file=str(data_path / "hr_train_split.csv"),
    )

    val_dataset = BrainGraphDataset(
        lr_file=str(data_path / "lr_val_split.csv"),
        hr_file=str(data_path / "hr_val_split.csv"),
    )

    test_dataset = BrainGraphDataset(
        lr_file=str(data_path / "lr_test_split.csv"),
        hr_file=str(data_path / "hr_test_split.csv"),
    )

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

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="generated_data",
        batch_size=32,
        seed=42,
        num_workers=0,
        shuffle_train=True,
    )
    print("Train batch:")
    print(next(iter(train_loader))[0])
    print("Val batch:")
    print(next(iter(val_loader))[0])
    print("Test batch:")
    print(next(iter(test_loader))[0])

"""Train cell-grouped Zhang and Mohtat SOH model ensembles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .cigar_eis_soh import SOHRegressor
else:
    from cigar_eis_soh import SOHRegressor

N_SPLITS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 5
ZHANG_EPOCHS = 500
MOHTAT_EPOCHS = 800
INPUT_SEQ_LENGTH = 60
TARGET_SEQ_LENGTH = 41


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared data file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "cell_ids"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")
        features = np.asarray(data["X"], dtype=np.float32)
        targets = np.asarray(data["y"], dtype=np.float32).reshape(-1, 1)
        groups = data["cell_ids"].astype(str)
    if features.ndim != 3 or features.shape[2] != 2:
        raise ValueError(f"Expected X with shape (samples, sequence, 2), got {features.shape}")
    if features.shape[1] != INPUT_SEQ_LENGTH:
        raise ValueError(
            f"Expected {INPUT_SEQ_LENGTH} input EIS steps for the existing "
            f"convolution configuration, got {features.shape[1]}"
        )
    if len(features) != len(targets) or len(targets) != len(groups):
        raise ValueError(f"Sample counts do not agree in {path}")
    if not len(targets) or not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError(f"Prepared dataset must be non-empty and finite: {path}")
    return features, targets, groups


def train_ensemble(
    data_path: Path,
    output_dir: Path,
    *,
    holdout_cell: str,
    epochs: int,
    seed: int,
    device: torch.device,
) -> None:
    features, targets, groups = _load_dataset(data_path)
    unique_groups = np.unique(groups)
    if holdout_cell not in unique_groups:
        available = ", ".join(unique_groups.tolist())
        raise ValueError(f"Holdout cell {holdout_cell!r} is not present in {data_path}; found: {available}")
    included = groups != holdout_cell
    train_x, train_y, train_groups = features[included], targets[included], groups[included]
    group_count = len(np.unique(train_groups))
    if N_SPLITS > group_count:
        raise ValueError(
            f"{data_path} has {group_count} training cells; at least {N_SPLITS} are required"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_indices, _) in enumerate(
        splitter.split(train_x, train_y, train_groups)
    ):
        fold_seed = seed + fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)

        fold_x = train_x[train_indices]
        fold_train_y = train_y[train_indices]
        target_min = float(fold_train_y.min())
        target_scale = float(fold_train_y.max() - target_min)
        if target_scale < 1e-8:
            target_scale = 1.0
        fold_y = (fold_train_y - target_min) / target_scale

        dataset = TensorDataset(
            torch.as_tensor(fold_x, dtype=torch.float32),
            torch.as_tensor(fold_y, dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(fold_seed)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )
        model = SOHRegressor().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
        )
        loss_function = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
                mean_loss = total_loss / len(dataset)
                print(
                    f"{data_path.stem} fold {fold + 1}/{N_SPLITS} "
                    f"epoch {epoch + 1}/{epochs}: MSE={mean_loss:.6f}"
                )

        checkpoint = {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "input_size": 2,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "sequence_length": int(features.shape[1]),
            "target_seq_length": TARGET_SEQ_LENGTH,
            "target_min": target_min,
            "target_scale": target_scale,
            "holdout_cell": holdout_cell,
            "seed": fold_seed,
        }
        checkpoint_path = output_dir / f"fold_{fold:02d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zhang-data", type=Path, default=Path("prepared_data/zhang.npz"))
    parser.add_argument("--mohtat-data", type=Path, default=Path("prepared_data/mohtat.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--zhang-holdout", default="25C05")
    parser.add_argument("--mohtat-holdout", default="MC21")
    parser.add_argument("--zhang-epochs", type=int, default=ZHANG_EPOCHS)
    parser.add_argument("--mohtat-epochs", type=int, default=MOHTAT_EPOCHS)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.zhang_epochs < 1 or args.mohtat_epochs < 1:
        parser.error("--zhang-epochs and --mohtat-epochs must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    device = torch.device(args.device)
    train_ensemble(
        args.zhang_data,
        args.output_dir / "zhang",
        holdout_cell=args.zhang_holdout,
        epochs=args.zhang_epochs,
        seed=args.seed,
        device=device,
    )
    train_ensemble(
        args.mohtat_data,
        args.output_dir / "mohtat",
        holdout_cell=args.mohtat_holdout,
        epochs=args.mohtat_epochs,
        seed=args.seed + N_SPLITS,
        device=device,
    )


if __name__ == "__main__":
    main()

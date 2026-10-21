"""Train-independent CIGAR fusion for Zhang and Mohtat EIS-to-SOH models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn


class SOHModel(Protocol):
    def predict(self, features: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Return mean SOH and ensemble uncertainty in physical SOH units."""
        ...

    def dy(self, features: ArrayLike, soh: ArrayLike) -> ArrayLike:
        """Return the per-sample score with respect to candidate SOH."""
        ...


class SOHRegressor(nn.Module):
    """The convolutional BiLSTM configuration used by the existing CIGAR models."""

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        num_layers: int = 5,
        target_seq_length: int = 41,
    ) -> None:
        super().__init__()
        self.target_seq_length = target_seq_length
        self.conv = nn.Conv1d(input_size, input_size, kernel_size=1)
        self.conv_adjust = nn.Conv1d(
            input_size, input_size, kernel_size=20, stride=1, padding=0
        )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sequence_length: int | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.permute(0, 2, 1)
        features = self.conv(features)
        if features.size(2) != self.target_seq_length:
            features = self.conv_adjust(features)
        if features.size(2) != self.target_seq_length:
            raise ValueError(
                f"Model expects a sequence that becomes {self.target_seq_length} steps "
                "after convolution"
            )
        features = features.permute(0, 2, 1)
        state_shape = (
            self.lstm.num_layers * 2,
            features.size(0),
            self.lstm.hidden_size,
        )
        if self.training:
            hidden = torch.randn(state_shape, device=features.device, dtype=features.dtype)
            cell = torch.randn(state_shape, device=features.device, dtype=features.dtype)
        else:
            hidden = torch.zeros(state_shape, device=features.device, dtype=features.dtype)
            cell = torch.zeros(state_shape, device=features.device, dtype=features.dtype)
        output, _ = self.lstm(features, (hidden, cell))
        output = self.dropout(output)
        return self.fc(output[:, -1, :])


def _as_sample_vector(values: ArrayLike, *, name: str, n_samples: int) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape == (n_samples, 1):
        vector = vector[:, 0]
    if vector.shape != (n_samples,):
        raise ValueError(f"{name} must have shape ({n_samples},) or ({n_samples}, 1)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def fuse_soh_predictions(
    eis_features: ArrayLike,
    models: Sequence[SOHModel],
    *,
    n_iter: int = 30,
    learning_rate: float = 0.005,
    initial_model: int = 0,
    noise_std: float = 1e-4,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Iteratively combine the score signals of compatible SOH models."""
    features = np.asarray(eis_features, dtype=np.float64)
    if features.ndim != 3 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("eis_features must have shape (samples, sequence, channels)")
    if not np.isfinite(features).all():
        raise ValueError("eis_features must contain only finite values")
    if not models:
        raise ValueError("at least one trained SOH model is required")
    if not isinstance(n_iter, (int, np.integer)) or isinstance(n_iter, (bool, np.bool_)):
        raise TypeError("n_iter must be an integer")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")
    if not isinstance(initial_model, (int, np.integer)) or isinstance(initial_model, (bool, np.bool_)):
        raise TypeError("initial_model must be an integer index")
    if not 0 <= initial_model < len(models):
        raise ValueError("initial_model must refer to one of the supplied models")

    n_samples = features.shape[0]
    initial_prediction, _ = models[initial_model].predict(features)
    soh = _as_sample_vector(
        initial_prediction, name="initial model predictions", n_samples=n_samples
    ).copy()
    if noise_std:
        soh += np.random.default_rng(random_state).normal(0.0, noise_std, n_samples)

    for iteration in range(1, int(n_iter) + 1):
        combined_score = np.zeros(n_samples, dtype=np.float64)
        for model_index, model in enumerate(models):
            score = _as_sample_vector(
                model.dy(features, soh),
                name=f"model {model_index} score",
                n_samples=n_samples,
            )
            combined_score += score

        score_norm = np.linalg.norm(combined_score)
        if not np.isfinite(score_norm):
            raise ValueError("combined model score has a non-finite norm")
        if score_norm == 0:
            break
        soh += learning_rate * combined_score / (score_norm * np.sqrt(iteration))

    return soh


class SOHEnsemble:
    """Load a fold ensemble and expose predictions and CIGAR score signals."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        variance_floor: float = 1e-4,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        if not np.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")
        self.variance_floor = float(variance_floor)
        checkpoint_paths = sorted(self.model_dir.glob("fold_*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No fold_*.pt model checkpoints found in {self.model_dir}")

        self.models: list[tuple[SOHRegressor, float, float]] = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            model = SOHRegressor(
                input_size=int(checkpoint["input_size"]),
                hidden_size=int(checkpoint["hidden_size"]),
                num_layers=int(checkpoint["num_layers"]),
                target_seq_length=int(checkpoint["target_seq_length"]),
            ).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            model.sequence_length = int(checkpoint["sequence_length"])
            target_min = float(checkpoint["target_min"])
            target_scale = float(checkpoint["target_scale"])
            if not np.isfinite(target_min) or not np.isfinite(target_scale) or target_scale <= 0:
                raise ValueError(f"Invalid target scaling in checkpoint: {checkpoint_path}")
            self.models.append((model, target_min, target_scale))

    def _predict_members(self, features: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError("features must have shape (samples, sequence, channels)")
        if not np.isfinite(values).all():
            raise ValueError("features must contain only finite values")

        predictions = []
        with torch.inference_mode():
            for model, target_min, target_scale in self.models:
                if values.shape[1] != model.sequence_length:
                    raise ValueError(
                        f"features have {values.shape[1]} time steps; "
                        f"model expects {model.sequence_length}"
                    )
                if values.shape[2] != model.lstm.input_size:
                    raise ValueError(
                        f"features have {values.shape[2]} channels; "
                        f"model expects {model.lstm.input_size}"
                    )
                tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
                scaled_prediction = model(tensor).squeeze(1)
                predictions.append(
                    (scaled_prediction * target_scale + target_min).cpu().numpy()
                )
        return np.stack(predictions, axis=0)

    def predict(self, features: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        member_predictions = self._predict_members(features)
        return member_predictions.mean(axis=0), member_predictions.std(axis=0)

    def dy(self, features: ArrayLike, soh: ArrayLike) -> NDArray[np.float64]:
        mean, std = self.predict(features)
        candidate = _as_sample_vector(soh, name="soh", n_samples=len(mean))
        variance = np.maximum(np.square(std), self.variance_floor)
        return (mean - candidate) / variance


def _load_prepared_cell(path: Path, cell_id: str) -> tuple[NDArray[np.float32], NDArray, NDArray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared data file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "cell_ids", "cycles"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")
        cell_ids = data["cell_ids"].astype(str)
        mask = cell_ids == cell_id
        if not mask.any():
            available = ", ".join(sorted(set(cell_ids.tolist())))
            raise ValueError(f"Cell {cell_id!r} not found in {path}. Available cells: {available}")
        return data["X"][mask], data["y"][mask].reshape(-1), data["cycles"][mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared .npz data for one cell domain")
    parser.add_argument("--cell", required=True, help="Cell identifier, e.g. 25C05 or MC21")
    parser.add_argument("--zhang-models", type=Path, required=True, help="Zhang fold checkpoint directory")
    parser.add_argument("--mohtat-models", type=Path, required=True, help="Mohtat fold checkpoint directory")
    parser.add_argument("--output", type=Path, default=Path("cigar_predictions.csv"))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    features, truth, cycles = _load_prepared_cell(args.data, args.cell)
    models = [
        SOHEnsemble(args.zhang_models, device=args.device),
        SOHEnsemble(args.mohtat_models, device=args.device),
    ]
    individual = [model.predict(features)[0] for model in models]
    fused = fuse_soh_predictions(
        features,
        models,
        n_iter=args.iterations,
        learning_rate=args.learning_rate,
        random_state=args.seed,
    )
    result = pd.DataFrame(
        {
            "cell_id": args.cell,
            "cycle": cycles,
            "soh_true": truth,
            "soh_zhang": individual[0],
            "soh_mohtat": individual[1],
            "soh_cigar": fused,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    rmse = float(np.sqrt(np.mean(np.square(fused - truth))))
    print(f"Saved {len(result)} predictions to {args.output}")
    print(f"CIGAR RMSE for {args.cell}: {rmse:.6f}")


if __name__ == "__main__":
    main()

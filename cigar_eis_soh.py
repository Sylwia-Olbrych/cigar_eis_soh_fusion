"""Train-independent CIGAR fusion for Zhang and Mohtat EIS-to-SOH models."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import r2_score

from src.model import SOHRegressor

DEFAULT_FUSION_ITERATIONS = 30
DEFAULT_FUSION_LEARNING_RATE = 0.005
DEFAULT_FUSION_RANDOM_SEED = 2010
FUSION_SCORE_SAMPLES = 10
FUSION_SCORE_STEP = 0.001
PROJECT_ROOT = Path(__file__).resolve().parent


class SOHModel(Protocol):
    def predict(self, features: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Return mean SOH and ensemble uncertainty in physical SOH units."""
        ...

    def dy(
        self, features: ArrayLike, soh: ArrayLike, *, rng: random.Random
    ) -> ArrayLike:
        """Return the per-sample score with respect to candidate SOH."""
        ...


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
    n_iter: int = DEFAULT_FUSION_ITERATIONS,
    learning_rate: float = DEFAULT_FUSION_LEARNING_RATE,
    initial_model: int = 0,
    noise_std: float = 1e-4,
    random_state: int | None = DEFAULT_FUSION_RANDOM_SEED,
    on_iteration: Callable[[int, NDArray[np.float64]], None] | None = None,
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
    rng = random.Random(random_state)
    if noise_std:
        soh = np.asarray(
            [rng.gauss(float(value), noise_std) for value in soh], dtype=np.float64
        )

    for iteration in range(1, int(n_iter) + 1):
        combined_score = np.zeros(n_samples, dtype=np.float64)
        for model_index, model in enumerate(models):
            score = _as_sample_vector(
                model.dy(features, soh, rng=rng),
                name=f"model {model_index} score",
                n_samples=n_samples,
            )
            combined_score += score

        score_norm = np.linalg.norm(combined_score)
        if not np.isfinite(score_norm):
            raise ValueError("combined model score has a non-finite norm")
        if score_norm == 0:
            if on_iteration is not None:
                on_iteration(iteration, soh.copy())
            break
        soh += learning_rate * combined_score / (score_norm * np.sqrt(iteration))
        if on_iteration is not None:
            on_iteration(iteration, soh.copy())

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

    def dy(
        self,
        features: ArrayLike,
        soh: ArrayLike,
        *,
        rng: random.Random,
    ) -> NDArray[np.float64]:
        member_predictions = self._predict_members(features)
        mean = member_predictions.mean(axis=0)
        variance = np.maximum(
            member_predictions.var(axis=0), self.variance_floor
        )
        candidate = _as_sample_vector(soh, name="soh", n_samples=len(mean))
        baseline = -0.5 * np.log(2.0 * np.pi * variance) - (
            0.5 * np.square(candidate - mean) / variance
        )
        score = np.zeros_like(candidate)
        for _ in range(FUSION_SCORE_SAMPLES):
            perturbation = rng.gauss(0.0, 1.0)
            perturbed = candidate + FUSION_SCORE_STEP * perturbation
            likelihood = -0.5 * np.log(2.0 * np.pi * variance) - (
                0.5 * np.square(perturbed - mean) / variance
            )
            score += (
                perturbation
                / (FUSION_SCORE_SAMPLES * FUSION_SCORE_STEP)
                * (likelihood - baseline)
            )
        return score


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


def _regression_metrics(
    truth: ArrayLike, predictions: ArrayLike
) -> tuple[float, float]:
    actual = np.asarray(truth, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("truth and predictions must have the same non-empty shape")
    rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
    return rmse, float(r2_score(actual, predicted))


def _save_metrics_plot(
    output_path: Path, iterations: Sequence[int], rmses: Sequence[float], r2s: Sequence[float]
) -> None:
    if not iterations or len(iterations) != len(rmses) or len(iterations) != len(r2s):
        raise ValueError("iteration, RMSE, and R2 histories must be non-empty and aligned")

    figure, (rmse_axis, r2_axis) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    try:
        rmse_axis.plot(iterations, rmses, marker="o")
        rmse_axis.set_ylabel("RMSE")
        rmse_axis.set_title("Fusion performance by iteration")
        rmse_axis.grid(True, alpha=0.3)

        r2_axis.plot(iterations, r2s, marker="o", color="tab:orange")
        r2_axis.set_xlabel("Fusion iteration")
        r2_axis.set_ylabel("R²")
        r2_axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
    finally:
        plt.close(figure)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Prepared .npz data for a single custom-cell run (requires --cell)",
    )
    parser.add_argument("--cell", default=None, help="Cell identifier for a single custom-cell run")
    parser.add_argument(
        "--zhang-models",
        type=Path,
        default=PROJECT_ROOT / "models" / "zhang",
        help="Zhang fold checkpoint directory (default: models/zhang)",
    )
    parser.add_argument(
        "--mohtat-models",
        type=Path,
        default=PROJECT_ROOT / "models" / "mohtat",
        help="Mohtat fold checkpoint directory (default: models/mohtat)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prediction CSV path (default: domain-specific file under results/)",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_FUSION_ITERATIONS)
    parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_FUSION_LEARNING_RATE
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_FUSION_RANDOM_SEED)
    return parser


def _resolve_targets(args: argparse.Namespace) -> list[tuple[str, Path, str, Path]]:
    if args.data is None and args.cell is None:
        return [
            (
                "Zhang + Mohtat",
                PROJECT_ROOT / "prepared_data" / "zhang.npz",
                "25C05",
                PROJECT_ROOT / "results" / "zhang_25C05.csv",
            ),
            (
                "Mohtat + Zhang",
                PROJECT_ROOT / "prepared_data" / "mohtat.npz",
                "MC21",
                PROJECT_ROOT / "results" / "mohtat_MC21.csv",
            ),
        ]
    if args.data is None or args.cell is None:
        raise ValueError("--data and --cell must be provided together")

    domain = "Mohtat + Zhang" if args.cell.startswith("MC") else "Zhang + Mohtat"
    output = args.output or PROJECT_ROOT / "results" / f"{args.cell}_cigar_predictions.csv"
    return [(domain, args.data, args.cell, output)]


def _run_fusion_target(
    domain: str,
    data_path: Path,
    cell: str,
    output_path: Path,
    zhang_model: SOHEnsemble,
    mohtat_model: SOHEnsemble,
    *,
    iterations: int,
    learning_rate: float,
    seed: int,
) -> None:
    features, truth, cycles = _load_prepared_cell(data_path, cell)
    if domain.startswith("Mohtat"):
        models = [mohtat_model, zhang_model]
        individual = {
            "soh_mohtat": mohtat_model.predict(features)[0],
            "soh_zhang": zhang_model.predict(features)[0],
        }
    else:
        models = [zhang_model, mohtat_model]
        individual = {
            "soh_zhang": zhang_model.predict(features)[0],
            "soh_mohtat": mohtat_model.predict(features)[0],
        }
    metric_history: list[tuple[int, float, float]] = []

    def report_fusion_metrics(
        iteration: int, prediction: NDArray[np.float64]
    ) -> None:
        rmse, r2 = _regression_metrics(truth, prediction)
        metric_history.append((iteration, rmse, r2))
        print(
            f"{domain} ({cell}) - Iteration {iteration}/{iterations}: "
            f"RMSE={rmse:.6f}, R2={r2:.6f}"
        )

    fused = fuse_soh_predictions(
        features,
        models,
        n_iter=iterations,
        learning_rate=learning_rate,
        initial_model=0,
        random_state=seed,
        on_iteration=report_fusion_metrics,
    )
    result = pd.DataFrame(
        {
            "cell_id": cell,
            "cycle": cycles,
            "soh_true": truth,
            "soh_zhang": individual["soh_zhang"],
            "soh_mohtat": individual["soh_mohtat"],
            "soh_cigar": fused,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    if metric_history:
        plot_path = output_path.with_name(f"{output_path.stem}_metrics.png")
        _save_metrics_plot(
            plot_path,
            [item[0] for item in metric_history],
            [item[1] for item in metric_history],
            [item[2] for item in metric_history],
        )
        print(f"Metrics plot saved to: {plot_path.resolve()}")


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    try:
        targets = _resolve_targets(args)
    except ValueError as error:
        parser.error(str(error))

    zhang_model = SOHEnsemble(args.zhang_models, device=args.device)
    mohtat_model = SOHEnsemble(args.mohtat_models, device=args.device)
    for domain, data_path, cell, output_path in targets:
        _run_fusion_target(
            domain,
            data_path,
            cell,
            output_path,
            zhang_model,
            mohtat_model,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

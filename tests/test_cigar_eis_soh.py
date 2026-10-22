import inspect
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from cigar_eis_soh import (
    DEFAULT_FUSION_ITERATIONS,
    DEFAULT_FUSION_LEARNING_RATE,
    DEFAULT_FUSION_RANDOM_SEED,
    FUSION_SCORE_SAMPLES,
    FUSION_SCORE_STEP,
    SOHEnsemble,
    PROJECT_ROOT,
    _build_argument_parser,
    _resolve_targets,
    _regression_metrics,
    _save_metrics_plot,
    fuse_soh_predictions,
)
from src.data_preprocessing_Zhang import ZHANG_CELLS, _zhang_cell
from src.data_preprocessing_mohtat import prepare_mohtat
from prepare_data import main as prepare_data_main
from train_models import N_SPLITS, train_ensemble


class ConstantScoreModel:
    def predict(self, features):
        return np.zeros((len(features), 1)), np.zeros((len(features), 1))

    def dy(self, features, soh, *, rng):
        return np.ones(len(features))


class CIGARTests(unittest.TestCase):
    def test_fusion_cli_runs_with_project_local_defaults(self):
        args = _build_argument_parser().parse_args([])

        self.assertIsNone(args.data)
        self.assertIsNone(args.cell)
        self.assertEqual(args.zhang_models, PROJECT_ROOT / "models" / "zhang")
        self.assertEqual(args.mohtat_models, PROJECT_ROOT / "models" / "mohtat")
        self.assertEqual(
            _resolve_targets(args),
            [
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
            ],
        )

    def test_custom_cell_cli_keeps_single_domain_override(self):
        args = _build_argument_parser().parse_args(
            ["--data", "prepared_data/mohtat.npz", "--cell", "MC21"]
        )

        self.assertEqual(
            _resolve_targets(args),
            [
                (
                    "Mohtat + Zhang",
                    Path("prepared_data/mohtat.npz"),
                    "MC21",
                    PROJECT_ROOT / "results" / "MC21_cigar_predictions.csv",
                )
            ],
        )

    def test_fusion_defaults_match_old_pipeline_settings(self):
        parameters = inspect.signature(fuse_soh_predictions).parameters

        self.assertEqual(DEFAULT_FUSION_ITERATIONS, 30)
        self.assertEqual(DEFAULT_FUSION_LEARNING_RATE, 0.005)
        self.assertEqual(DEFAULT_FUSION_RANDOM_SEED, 2010)
        self.assertEqual(parameters["n_iter"].default, DEFAULT_FUSION_ITERATIONS)
        self.assertEqual(
            parameters["learning_rate"].default, DEFAULT_FUSION_LEARNING_RATE
        )

    def test_ensemble_score_matches_old_stochastic_finite_difference(self):
        member_predictions = np.array([[0.7, 0.8], [0.9, 1.0], [0.8, 0.9]])
        candidates = np.array([0.75, 0.95])
        rng = random.Random(123)
        model = SOHEnsemble.__new__(SOHEnsemble)
        model.variance_floor = 1e-4
        model._predict_members = lambda features: member_predictions

        actual = model.dy(np.zeros((2, 3, 2)), candidates, rng=rng)

        mean = member_predictions.mean(axis=0)
        variance = np.maximum(member_predictions.var(axis=0), 1e-4)
        baseline = -0.5 * np.log(2 * np.pi * variance) - (
            0.5 * np.square(candidates - mean) / variance
        )
        expected = np.zeros_like(candidates)
        reference_rng = random.Random(123)
        for _ in range(FUSION_SCORE_SAMPLES):
            z = reference_rng.gauss(0, 1)
            perturbed = candidates + FUSION_SCORE_STEP * z
            likelihood = -0.5 * np.log(2 * np.pi * variance) - (
                0.5 * np.square(perturbed - mean) / variance
            )
            expected += (z / (FUSION_SCORE_SAMPLES * FUSION_SCORE_STEP)) * (
                likelihood - baseline
            )
        np.testing.assert_allclose(actual, expected)

    def test_fusion_initialization_noise_matches_old_python_gaussian(self):
        seed = 2010
        fused = fuse_soh_predictions(
            np.zeros((3, 4, 2)),
            [ConstantScoreModel()],
            n_iter=0,
            random_state=seed,
        )
        rng = random.Random(seed)
        expected = np.array([rng.gauss(0.0, 1e-4) for _ in range(3)])

        np.testing.assert_array_equal(fused, expected)

    def test_regression_metrics_report_rmse_and_r2(self):
        rmse, r2 = _regression_metrics([1, 2, 3], [1, 2, 4])

        self.assertAlmostEqual(rmse, np.sqrt(1 / 3))
        self.assertAlmostEqual(r2, 0.5)

    def test_metrics_plot_is_saved(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            plot_path = Path(temporary_directory) / "fusion_metrics.png"

            _save_metrics_plot(
                plot_path,
                [1, 2, 3],
                [0.3, 0.2, 0.1],
                [0.1, 0.5, 0.8],
            )

            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_fusion_applies_norm_scaled_score_updates(self):
        features = np.zeros((2, 4, 2))
        iteration_predictions = []
        fused = fuse_soh_predictions(
            features,
            [ConstantScoreModel(), ConstantScoreModel()],
            n_iter=2,
            learning_rate=0.5,
            noise_std=0,
            on_iteration=lambda iteration, prediction: iteration_predictions.append(
                (iteration, prediction)
            ),
        )
        expected = np.full(2, 0.5 / np.sqrt(2) + 0.25)
        np.testing.assert_allclose(fused, expected)
        self.assertEqual([iteration for iteration, _ in iteration_predictions], [1, 2])
        np.testing.assert_allclose(iteration_predictions[-1][1], fused)

    def test_zero_score_leaves_initial_prediction_unchanged(self):
        class ZeroScoreModel(ConstantScoreModel):
            def dy(self, features, soh, *, rng):
                return np.zeros(len(features))

        fused = fuse_soh_predictions(
            np.zeros((2, 4, 2)), [ZeroScoreModel()], n_iter=5, noise_std=0
        )
        np.testing.assert_array_equal(fused, np.zeros(2))

    def test_mohtat_preparation_pairs_capacity_and_eis_by_cycle(self):
        rows = []
        for cycle, capacity in ((1, 1.0), (2, 0.9), (3, 0.65)):
            for frequency, magnitude in ((1.0, 2.0), (10.0, 3.0)):
                rows.append(
                    {
                        "n": cycle,
                        "Capacity": capacity,
                        "SOC": 0.9,
                        "EIS_Frequency": frequency,
                        "EIS_Z_abs": magnitude,
                        "EIS_Z_phase": 0.5,
                    }
                )

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            pd.DataFrame(rows).to_csv(
                Path(temporary_directory) / "Mohtat_01.csv", index=False
            )
            features, targets, cells, cycles = prepare_mohtat(
                temporary_directory, sequence_length=4
            )

        self.assertEqual(features.shape, (2, 4, 2))
        np.testing.assert_allclose(targets, [1.0, 0.9])
        np.testing.assert_array_equal(cells, ["MC01", "MC01"])
        np.testing.assert_array_equal(cycles, [1, 2])
        self.assertTrue(np.isfinite(features).all())
        self.assertGreaterEqual(float(features.min()), 0.0)
        self.assertLessEqual(float(features.max()), 1.0)

    def test_prepare_data_runs_and_saves_both_preprocessors(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            zhang_dir = root / "zhang"
            capacity_dir = zhang_dir / "Capacity_data"
            eis_dir = zhang_dir / "EIS_data"
            capacity_dir.mkdir(parents=True)
            eis_dir.mkdir()

            for cell_id, temperature, cell_number in ZHANG_CELLS:
                capacity_column = (
                    3 if temperature != 25 or cell_number in (1, 5) else 5
                )
                capacity_rows = []
                for cycle, value in ((0, 100.0), (1, 90.0)):
                    row = [0, cycle, 0, 0, 0, 0]
                    row[capacity_column] = value
                    capacity_rows.append(row)
                pd.DataFrame(
                    capacity_rows,
                    columns=["a", "cycle", "step", "cap", "d", "other"],
                ).to_csv(
                    capacity_dir
                    / f"Data_Capacity_{temperature}C{cell_number:02d}.txt",
                    sep=" ",
                    index=False,
                )
                pd.DataFrame(
                    [
                        [0, cycle, 0, real, imag]
                        for cycle in (1, 2)
                        for real, imag in ((1.0, -1.0), (2.0, -2.0))
                    ],
                    columns=["a", "cycle", "b", "real", "imag"],
                ).to_csv(
                    eis_dir / f"EIS_state_V_{temperature}C{cell_number:02d}.txt",
                    sep=" ",
                    index=False,
                )

            mohtat_dir = root / "mohtat"
            mohtat_dir.mkdir()
            mohtat_rows = [
                {
                    "n": cycle,
                    "Capacity": capacity,
                    "SOC": 0.9,
                    "EIS_Frequency": frequency,
                    "EIS_Z_abs": magnitude,
                    "EIS_Z_phase": 0.5,
                }
                for cycle, capacity in ((1, 1.0), (2, 0.9))
                for frequency, magnitude in ((1.0, 2.0), (10.0, 3.0))
            ]
            pd.DataFrame(mohtat_rows).to_csv(
                mohtat_dir / "Mohtat_01.csv", index=False
            )
            output_dir = root / "prepared"

            with patch(
                "sys.argv",
                [
                    "prepare_data.py",
                    "--zhang-dir",
                    str(zhang_dir),
                    "--mohtat-dir",
                    str(mohtat_dir),
                    "--output-dir",
                    str(output_dir),
                ],
            ):
                prepare_data_main()

            with np.load(output_dir / "zhang.npz", allow_pickle=False) as data:
                self.assertEqual(data["X"].shape, (20, 60, 2))
                self.assertEqual(data["source"].item(), "zhang")
            with np.load(output_dir / "mohtat.npz", allow_pickle=False) as data:
                self.assertEqual(data["X"].shape, (2, 60, 2))
                np.testing.assert_array_equal(data["cell_ids"], ["MC01", "MC01"])
                self.assertEqual(data["source"].item(), "mohtat")

    def test_zhang_preparation_uses_expected_capacity_column(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            (root / "Capacity_data").mkdir()
            (root / "EIS_data").mkdir()
            capacity_path = root / "Capacity_data" / "Data_Capacity_25C01.txt"
            eis_path = root / "EIS_data" / "EIS_state_V_25C01.txt"
            pd.DataFrame(
                [
                    [0, 0, 0, 100.0, 0, 999.0],
                    [0, 1, 0, 80.0, 0, 999.0],
                ],
                columns=["a", "cycle", "step", "capacity", "e", "unused"],
            ).to_csv(capacity_path, sep=" ", index=False)
            pd.DataFrame(
                [
                    [0, 1, 0, 1.0, -1.0],
                    [0, 1, 0, 2.0, -2.0],
                    [0, 2, 0, 1.5, -1.5],
                    [0, 2, 0, 2.5, -2.5],
                ],
                columns=["a", "cycle", "b", "real", "imag"],
            ).to_csv(eis_path, sep=" ", index=False)

            rows = _zhang_cell(root, "25C01", 25, 1, 4)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0:2], ("25C01", 1))
        self.assertAlmostEqual(rows[0][3], 1.0)
        self.assertEqual(rows[0][2].shape, (4, 2))
        self.assertEqual(rows[1][0:2], ("25C01", 2))
        self.assertAlmostEqual(rows[1][3], 0.8)

    def test_training_scales_targets_from_each_fold_training_groups(self):
        class TinyRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.0))

            def forward(self, features):
                return self.weight.expand(len(features), 1)

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "data.npz"
            output_dir = root / "models"
            groups = np.array(["A", "B", "C", "D", "E", "HOLDOUT"])
            targets = np.array([0, 1, 2, 3, 4, 100], dtype=np.float32)
            features = np.zeros((len(groups), 60, 2), dtype=np.float32)
            np.savez(data_path, X=features, y=targets, cell_ids=groups)

            with patch("train_models.SOHRegressor", TinyRegressor):
                train_ensemble(
                    data_path,
                    output_dir,
                    holdout_cell="HOLDOUT",
                    epochs=1,
                    seed=1,
                    device=torch.device("cpu"),
                )

            train_mask = groups != "HOLDOUT"
            train_groups = groups[train_mask]
            train_targets = targets[train_mask]
            splits = GroupKFold(n_splits=N_SPLITS).split(
                features[train_mask], train_targets, train_groups
            )
            checkpoints = sorted(output_dir.glob("fold_*.pt"))
            self.assertEqual(len(checkpoints), N_SPLITS)
            for checkpoint_path, (fold_indices, _) in zip(checkpoints, splits):
                fold_targets = train_targets[fold_indices]
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                expected_min = float(fold_targets.min())
                expected_scale = float(fold_targets.max() - expected_min)
                self.assertEqual(checkpoint["target_min"], expected_min)
                self.assertEqual(checkpoint["target_scale"], expected_scale)


if __name__ == "__main__":
    unittest.main()

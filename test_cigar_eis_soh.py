import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

from cigar_eis_soh import fuse_soh_predictions
from prepare_data import _pack_rows, _zhang_cell, prepare_mohtat
from train_models import N_SPLITS, train_ensemble


class ConstantScoreModel:
    def predict(self, features):
        return np.zeros((len(features), 1)), np.zeros((len(features), 1))

    def dy(self, features, soh):
        return np.ones(len(features))


class CIGARTests(unittest.TestCase):
    def test_fusion_applies_norm_scaled_score_updates(self):
        features = np.zeros((2, 4, 2))
        fused = fuse_soh_predictions(
            features,
            [ConstantScoreModel(), ConstantScoreModel()],
            n_iter=2,
            learning_rate=0.5,
            noise_std=0,
        )
        expected = np.full(2, 0.5 / np.sqrt(2) + 0.25)
        np.testing.assert_allclose(fused, expected)

    def test_zero_score_leaves_initial_prediction_unchanged(self):
        class ZeroScoreModel(ConstantScoreModel):
            def dy(self, features, soh):
                return np.zeros(len(features))

        fused = fuse_soh_predictions(
            np.zeros((2, 4, 2)), [ZeroScoreModel()], n_iter=5, noise_std=0
        )
        np.testing.assert_array_equal(fused, np.zeros(2))

    def test_mohtat_preparation_pairs_capacity_and_eis_by_cycle(self):
        rows = []
        for cycle, capacity in ((1, 1.0), (2, 0.8)):
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

        with tempfile.TemporaryDirectory() as temporary_directory:
            pd.DataFrame(rows).to_csv(
                Path(temporary_directory) / "Mohtat_01.csv", index=False
            )
            features, targets, cells, cycles = prepare_mohtat(
                temporary_directory, sequence_length=4
            )

        self.assertEqual(features.shape, (2, 4, 2))
        np.testing.assert_allclose(targets, [1.0, 0.8])
        np.testing.assert_array_equal(cells, ["MC01", "MC01"])
        np.testing.assert_array_equal(cycles, [1, 2])

    def test_packing_normalizes_each_cell_channel_across_all_its_samples(self):
        rows = [
            ("cell-a", 1, np.array([[0, 10], [1, 20]], dtype=np.float32), 1.0),
            ("cell-a", 2, np.array([[2, 30], [3, 40]], dtype=np.float32), 0.8),
        ]

        features, _, _, _ = _pack_rows(rows, sequence_length=2)

        np.testing.assert_allclose(
            features,
            [
                [[0, 0], [1 / 3, 1 / 3]],
                [[2 / 3, 2 / 3], [1, 1]],
            ],
        )

    def test_zhang_preparation_uses_expected_capacity_column(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
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

        with tempfile.TemporaryDirectory() as temporary_directory:
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

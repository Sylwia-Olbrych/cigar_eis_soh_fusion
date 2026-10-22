"""Prepare the Zhang EIS and capacity files for CIGAR training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ZHANG_CELLS = (
    ("25C01", 25, 1),
    ("25C02", 25, 2),
    ("25C03", 25, 3),
    ("25C05", 25, 5),
    ("25C06", 25, 6),
    ("25C07", 25, 7),
    ("35C01", 35, 1),
    ("35C02", 35, 2),
    ("45C01", 45, 1),
    ("45C02", 45, 2),
)


def _resample_spectrum(
    spectrum: NDArray[np.float64], sequence_length: int
) -> NDArray[np.float32]:
    if spectrum.ndim != 2 or spectrum.shape[1] != 2 or len(spectrum) < 2:
        raise ValueError("An EIS spectrum must contain at least two real/imaginary points")
    if not np.isfinite(spectrum).all():
        raise ValueError("EIS spectrum contains non-finite values")
    old_axis = np.linspace(0.0, 1.0, len(spectrum))
    new_axis = np.linspace(0.0, 1.0, sequence_length)
    return np.column_stack(
        [np.interp(new_axis, old_axis, spectrum[:, channel]) for channel in range(2)]
    ).astype(np.float32)


def _numeric_column(frame: pd.DataFrame, column: int, path: Path) -> NDArray[np.float64]:
    if column >= frame.shape[1]:
        raise ValueError(f"Missing column {column} in {path}")
    return pd.to_numeric(frame.iloc[:, column], errors="coerce").to_numpy(dtype=np.float64)


def _zhang_cell(
    root: Path,
    cell_id: str,
    temperature: int,
    cell_number: int,
    sequence_length: int,
) -> list[tuple[str, int, NDArray[np.float32], float]]:
    capacity_path = (
        root
        / "Capacity_data"
        / f"Data_Capacity_{temperature}C{cell_number:02d}.txt"
    )
    eis_path = root / "EIS_data" / f"EIS_state_V_{temperature}C{cell_number:02d}.txt"
    if not capacity_path.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {capacity_path}")
    if not eis_path.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {eis_path}")
    capacity = pd.read_csv(capacity_path, sep=r"\s+", header=0)
    eis = pd.read_csv(eis_path, sep=r"\s+", header=0)

    capacity_cycle = _numeric_column(capacity, 1, capacity_path)
    capacity_step = _numeric_column(capacity, 2, capacity_path)
    capacity_column = 3 if temperature != 25 or cell_number in (1, 5) else 5
    capacity_value = _numeric_column(capacity, capacity_column, capacity_path)
    eis_cycle = _numeric_column(eis, 1, eis_path)
    eis_real = _numeric_column(eis, 3, eis_path)
    eis_imag = _numeric_column(eis, 4, eis_path)

    initial_mask = (
        (capacity_cycle == 0)
        & (capacity_step == 0)
        & np.isfinite(capacity_value)
    )
    if not initial_mask.any():
        raise ValueError(f"No initial capacity row found in {capacity_path}")
    initial_capacity = float(np.max(capacity_value[initial_mask]))
    if initial_capacity <= 0:
        raise ValueError(f"Initial capacity must be positive in {capacity_path}")

    rows = []
    cycle_values = np.unique(eis_cycle[np.isfinite(eis_cycle)]).astype(int)
    for cycle in sorted(cycle_values):
        capacity_mask = (
            (capacity_cycle == cycle - 1)
            & (capacity_step == 0)
            & np.isfinite(capacity_value)
        )
        point_mask = (
            (eis_cycle == cycle)
            & np.isfinite(eis_real)
            & np.isfinite(eis_imag)
        )
        if not capacity_mask.any() or point_mask.sum() < 2:
            continue
        spectrum = np.column_stack((eis_real[point_mask], eis_imag[point_mask]))
        soh = float(np.max(capacity_value[capacity_mask]) / initial_capacity)
        rows.append((cell_id, cycle, _resample_spectrum(spectrum, sequence_length), soh))
    return rows


def prepare_zhang(
    source_dir: str | Path, *, sequence_length: int = 60
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[str], NDArray[np.int64]]:
    """Load Zhang cells and return normalized sequences with raw SOH ratios."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    root = Path(source_dir)
    rows = [
        row
        for cell_id, temperature, cell_number in ZHANG_CELLS
        for row in _zhang_cell(
            root, cell_id, temperature, cell_number, sequence_length
        )
    ]
    if not rows:
        raise ValueError(f"No aligned Zhang EIS/SOH samples found in {root}")

    features = np.stack([row[2] for row in rows]).astype(np.float32)
    targets = np.asarray([row[3] for row in rows], dtype=np.float32)
    cells = np.asarray([row[0] for row in rows], dtype=str)
    cycles = np.asarray([row[1] for row in rows], dtype=np.int64)

    # Match the Zhang preprocessing convention: fit each channel's range to
    # the first spectrum for that cell, then apply it to all cycles.
    for cell_id in np.unique(cells):
        cell_mask = cells == cell_id
        for channel in range(features.shape[2]):
            reference = features[cell_mask, :, channel][0]
            minimum = float(reference.min())
            scale = float(reference.max() - minimum)
            if scale < 1e-8:
                features[cell_mask, :, channel] = 0.0
            else:
                features[cell_mask, :, channel] = (
                    features[cell_mask, :, channel] - minimum
                ) / scale

    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Prepared Zhang values must be finite")
    return features, targets, cells, cycles

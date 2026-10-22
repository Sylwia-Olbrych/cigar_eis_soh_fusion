"""Prepare Mohtat EIS and capacity CSV files for CIGAR training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


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


def _drop_soh_spikes(
    rows: list[tuple[str, int, NDArray[np.float32], float]], jump: float
) -> list[tuple[str, int, NDArray[np.float32], float]]:
    if len(rows) <= 2:
        return rows
    soh = np.asarray([row[3] for row in rows], dtype=np.float64)
    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = np.abs(np.diff(soh)) <= jump
    if abs(soh[1] - soh[0]) > jump:
        keep[0 if abs(1.0 - soh[0]) > abs(1.0 - soh[1]) else 1] = False
    return [row for row, include in zip(rows, keep) if include]


def _mohtat_cell(
    path: Path,
    cell_id: str,
    sequence_length: int,
    soc_target: float,
    soc_tolerance: float,
    spike_threshold: float,
) -> list[tuple[str, int, NDArray[np.float32], float]]:
    frame = pd.read_csv(path)
    required = {
        "n",
        "Capacity",
        "SOC",
        "EIS_Frequency",
        "EIS_Z_abs",
        "EIS_Z_phase",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    capacity = frame[["n", "Capacity"]].copy()
    capacity["n"] = pd.to_numeric(capacity["n"], errors="coerce")
    capacity["Capacity"] = pd.to_numeric(capacity["Capacity"], errors="coerce")
    capacity = capacity.dropna().groupby("n")["Capacity"].max()
    if capacity.empty:
        raise ValueError(f"No capacity values found in {path}")
    initial_capacity = float(capacity.max())
    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        raise ValueError(f"Initial capacity must be positive in {path}")

    for column in ("n", "SOC", "EIS_Frequency", "EIS_Z_abs", "EIS_Z_phase"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        np.isfinite(frame["SOC"])
        & (np.abs(frame["SOC"] - soc_target) <= soc_tolerance)
    ].copy()
    if frame.empty:
        raise ValueError(f"No EIS rows at SOC {soc_target:g} (±{soc_tolerance:g}) in {path}")

    phase = frame["EIS_Z_phase"].to_numpy(dtype=np.float64)
    if np.isfinite(phase).any() and np.nanmax(np.abs(phase)) > 3.5:
        phase = np.deg2rad(phase)
    magnitude = frame["EIS_Z_abs"].to_numpy(dtype=np.float64)
    frame["Zreal"] = magnitude * np.cos(phase)
    frame["Zimag"] = -magnitude * np.sin(phase)

    rows = []
    for cycle_value in sorted(frame["n"].dropna().unique()):
        if cycle_value not in capacity.index:
            continue
        points = frame.loc[
            frame["n"].eq(cycle_value),
            ["EIS_Frequency", "Zreal", "Zimag"],
        ].dropna()
        points = (
            points.groupby("EIS_Frequency", as_index=False)[["Zreal", "Zimag"]]
            .mean()
            .sort_values("EIS_Frequency")
        )
        if len(points) < 2:
            continue
        spectrum = points[["Zreal", "Zimag"]].to_numpy(dtype=np.float64)
        soh = float(capacity.loc[cycle_value] / initial_capacity)
        rows.append(
            (
                cell_id,
                int(cycle_value),
                _resample_spectrum(spectrum, sequence_length),
                soh,
            )
        )
    return _drop_soh_spikes(rows, spike_threshold)


def prepare_mohtat(
    source_dir: str | Path,
    *,
    sequence_length: int = 60,
    soc_target: float = 0.9,
    soc_tolerance: float = 1e-6,
    spike_threshold: float = 0.15,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[str], NDArray[np.int64]]:
    """Load Mohtat CSVs and return normalized sequences with raw SOH ratios."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if soc_tolerance < 0:
        raise ValueError("soc_tolerance must be non-negative")
    if spike_threshold < 0:
        raise ValueError("spike_threshold must be non-negative")
    root = Path(source_dir)
    paths = sorted(root.glob("Mohtat_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No Mohtat_*.csv source files found in {root}")

    rows = []
    for path in paths:
        suffix = path.stem.removeprefix("Mohtat_")
        if not suffix.isdigit():
            raise ValueError(f"Unexpected Mohtat filename: {path.name}")
        rows.extend(
            _mohtat_cell(
                path,
                f"MC{int(suffix):02d}",
                sequence_length,
                soc_target,
                soc_tolerance,
                spike_threshold,
            )
        )
    if not rows:
        raise ValueError(f"No aligned Mohtat EIS/SOH samples found in {root}")

    features = np.stack([row[2] for row in rows]).astype(np.float32)
    targets = np.asarray([row[3] for row in rows], dtype=np.float32)
    cells = np.asarray([row[0] for row in rows], dtype=str)
    cycles = np.asarray([row[1] for row in rows], dtype=np.int64)

    for cell_id in np.unique(cells):
        cell_mask = cells == cell_id
        for channel in range(features.shape[2]):
            values = features[cell_mask, :, channel]
            minimum = float(values.min())
            scale = float(values.max() - minimum)
            if scale < 1e-8:
                features[cell_mask, :, channel] = 0.0
            else:
                features[cell_mask, :, channel] = (values - minimum) / scale

    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Prepared Mohtat values must be finite")
    return features, targets, cells, cycles

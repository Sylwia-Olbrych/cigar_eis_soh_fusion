"""Convert the Zhang text files and Mohtat CSV files to common EIS/SOH arrays."""

from __future__ import annotations

import argparse
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


def _read_table(path: Path, *, whitespace: bool = False) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {path}")
    return pd.read_csv(path, sep=r"\s+" if whitespace else ",", header=0)


def _resample_spectrum(spectrum: NDArray[np.float64], sequence_length: int) -> NDArray[np.float32]:
    if spectrum.ndim != 2 or spectrum.shape[1] != 2 or len(spectrum) < 2:
        raise ValueError("An EIS spectrum must contain at least two real/imaginary points")
    if not np.isfinite(spectrum).all():
        raise ValueError("EIS spectrum contains non-finite values")
    old_axis = np.linspace(0.0, 1.0, len(spectrum))
    new_axis = np.linspace(0.0, 1.0, sequence_length)
    return np.column_stack(
        [np.interp(new_axis, old_axis, spectrum[:, channel]) for channel in range(2)]
    ).astype(np.float32)


def _numeric_column(frame: pd.DataFrame, column: str | int, *, path: Path) -> NDArray[np.float64]:
    try:
        values = frame.iloc[:, column] if isinstance(column, int) else frame[column]
    except (IndexError, KeyError) as exc:
        raise ValueError(f"Missing column {column!r} in {path}") from exc
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _zhang_cell(
    root: Path, cell_id: str, temperature: int, cell_number: int, sequence_length: int
) -> list[tuple[str, int, NDArray[np.float32], float]]:
    capacity_path = root / "Capacity_data" / f"Data_Capacity_{temperature}C{cell_number:02d}.txt"
    eis_path = root / "EIS_data" / f"EIS_state_V_{temperature}C{cell_number:02d}.txt"
    capacity = _read_table(capacity_path, whitespace=True)
    eis = _read_table(eis_path, whitespace=True)

    # The source format stores cycle and step in columns 1 and 2.
    capacity_cycle = _numeric_column(capacity, 1, path=capacity_path)
    capacity_step = _numeric_column(capacity, 2, path=capacity_path)
    capacity_column = 3 if temperature != 25 or cell_number in (1, 5) else 5
    capacity_value = _numeric_column(capacity, capacity_column, path=capacity_path)
    eis_cycle = _numeric_column(eis, 1, path=eis_path)
    eis_real = _numeric_column(eis, 3, path=eis_path)
    eis_imag = _numeric_column(eis, 4, path=eis_path)

    initial_mask = (capacity_cycle == 0) & (capacity_step == 0) & np.isfinite(capacity_value)
    if not initial_mask.any():
        raise ValueError(f"No initial capacity row found in {capacity_path}")
    initial_capacity = capacity_value[initial_mask][-1]
    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        raise ValueError(f"Initial capacity must be positive in {capacity_path}")

    samples = []
    for cycle in sorted(np.unique(eis_cycle[np.isfinite(eis_cycle)]).astype(int)):
        cap_mask = (
            (capacity_cycle == cycle - 1)
            & (capacity_step == 0)
            & np.isfinite(capacity_value)
        )
        point_mask = (
            (eis_cycle == cycle)
            & np.isfinite(eis_real)
            & np.isfinite(eis_imag)
        )
        if not cap_mask.any() or point_mask.sum() < 2:
            continue
        spectrum = np.column_stack((eis_real[point_mask], eis_imag[point_mask]))
        soh = float(np.max(capacity_value[cap_mask]) / initial_capacity)
        samples.append((cell_id, cycle, _resample_spectrum(spectrum, sequence_length), soh))
    return samples


def prepare_zhang(
    source_dir: str | Path, *, sequence_length: int = 60
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[str], NDArray[np.int64]]:
    """Read the Zhang capacity/EIS text files in the repository's source layout."""
    root = Path(source_dir)
    rows = [
        row
        for cell_id, temperature, cell_number in ZHANG_CELLS
        for row in _zhang_cell(root, cell_id, temperature, cell_number, sequence_length)
    ]
    return _pack_rows(rows, sequence_length)


def _mohtat_cell(
    path: Path, cell_id: str, sequence_length: int, soc_target: float, soc_tolerance: float
) -> list[tuple[str, int, NDArray[np.float32], float]]:
    frame = _read_table(path)
    required_columns = {
        "n",
        "Capacity",
        "SOC",
        "EIS_Frequency",
        "EIS_Z_abs",
        "EIS_Z_phase",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    capacity_rows = frame[["n", "Capacity"]].copy()
    capacity_rows["n"] = pd.to_numeric(capacity_rows["n"], errors="coerce")
    capacity_rows["Capacity"] = pd.to_numeric(capacity_rows["Capacity"], errors="coerce")
    capacity = capacity_rows.dropna().groupby("n")["Capacity"].max()
    if capacity.empty or float(capacity.max()) <= 0:
        raise ValueError(f"No positive capacity values found in {path}")
    initial_capacity = float(capacity.max())

    soc = pd.to_numeric(frame["SOC"], errors="coerce").to_numpy(dtype=np.float64)
    frame = frame.loc[np.isfinite(soc) & (np.abs(soc - soc_target) <= soc_tolerance)].copy()
    if frame.empty:
        raise ValueError(f"No EIS rows at SOC {soc_target:g} (±{soc_tolerance:g}) in {path}")

    frame["n"] = pd.to_numeric(frame["n"], errors="coerce")
    frame["Capacity"] = pd.to_numeric(frame["Capacity"], errors="coerce")
    frame["EIS_Frequency"] = pd.to_numeric(frame["EIS_Frequency"], errors="coerce")
    frame["EIS_Z_abs"] = pd.to_numeric(frame["EIS_Z_abs"], errors="coerce")
    phase = pd.to_numeric(frame["EIS_Z_phase"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(phase).any():
        raise ValueError(f"No valid EIS phase values in {path}")
    if np.nanmax(np.abs(phase)) > 3.5:
        phase = np.deg2rad(phase)
    magnitude = frame["EIS_Z_abs"].to_numpy(dtype=np.float64)
    frame["Zreal"] = magnitude * np.cos(phase)
    frame["Zimag"] = -magnitude * np.sin(phase)

    samples = []
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
        samples.append(
            (
                cell_id,
                int(cycle_value),
                _resample_spectrum(spectrum, sequence_length),
                soh,
            )
        )
    return samples


def prepare_mohtat(
    source_dir: str | Path,
    *,
    sequence_length: int = 60,
    soc_target: float = 0.9,
    soc_tolerance: float = 1e-6,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[str], NDArray[np.int64]]:
    """Read Mohtat_M.csv files containing capacity and EIS measurements."""
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
            _mohtat_cell(path, f"MC{int(suffix):02d}", sequence_length, soc_target, soc_tolerance)
        )
    return _pack_rows(rows, sequence_length)


def _pack_rows(
    rows: list[tuple[str, int, NDArray[np.float32], float]],
    sequence_length: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[str], NDArray[np.int64]]:
    if not rows:
        raise ValueError("No aligned EIS/SOH samples were found")
    features = np.stack([row[2] for row in rows]).astype(np.float32)
    targets = np.asarray([row[3] for row in rows], dtype=np.float32)
    cells = np.asarray([row[0] for row in rows], dtype=str)
    cycles = np.asarray([row[1] for row in rows], dtype=np.int64)
    for cell_id in np.unique(cells):
        cell_mask = cells == cell_id
        for channel in range(features.shape[2]):
            channel_values = features[cell_mask, :, channel]
            minimum = channel_values.min()
            scale = channel_values.max() - minimum
            if scale < 1e-8:
                features[cell_mask, :, channel] = 0.0
            else:
                features[cell_mask, :, channel] = (channel_values - minimum) / scale
    expected = (sequence_length, 2)
    if features.shape[1:] != expected:
        raise ValueError(f"Expected EIS feature shape {expected}, got {features.shape[1:]}")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("Prepared EIS/SOH values must be finite")
    return features, targets, cells, cycles


def _save_prepared(
    output_path: Path,
    source: str,
    arrays: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[str], NDArray[np.int64]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features, targets, cells, cycles = arrays
    np.savez_compressed(
        output_path,
        X=features,
        y=targets,
        cell_ids=cells,
        cycles=cycles,
        source=np.asarray(source),
    )
    print(f"Saved {len(targets)} samples from {len(np.unique(cells))} cells to {output_path}")


def main() -> None:
    default_root = Path(__file__).resolve().parent / "data" / "raw"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zhang-dir",
        type=Path,
        default=default_root / "zhang",
        help="Folder containing Capacity_data/ and EIS_data/ (default: data/raw/zhang)",
    )
    parser.add_argument(
        "--mohtat-dir",
        type=Path,
        default=default_root / "mohtat",
        help="Folder containing Mohtat_*.csv files (default: data/raw/mohtat)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--soc", type=float, default=0.9, help="Mohtat EIS SOC target")
    parser.add_argument("--soc-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if args.soc_tolerance < 0:
        parser.error("--soc-tolerance must be non-negative")

    zhang = prepare_zhang(args.zhang_dir, sequence_length=args.sequence_length)
    mohtat = prepare_mohtat(
        args.mohtat_dir,
        sequence_length=args.sequence_length,
        soc_target=args.soc,
        soc_tolerance=args.soc_tolerance,
    )
    _save_prepared(args.output_dir / "zhang.npz", "zhang", zhang)
    _save_prepared(args.output_dir / "mohtat.npz", "mohtat", mohtat)


if __name__ == "__main__":
    main()

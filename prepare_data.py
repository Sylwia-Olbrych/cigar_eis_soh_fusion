"""Run Zhang and Mohtat preprocessing and save CIGAR-ready NPZ datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.data_preprocessing_Zhang import prepare_zhang
from src.data_preprocessing_mohtat import prepare_mohtat


PreparedArrays = tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.str_], NDArray[np.int64]
]


def _save_prepared(
    output_path: Path, source: str, arrays: PreparedArrays
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
    print(
        f"Saved {len(targets)} samples from {len(np.unique(cells))} cells "
        f"to {output_path}"
    )


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
    parser.add_argument("--spike-threshold", type=float, default=0.15)
    args = parser.parse_args()
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if args.soc_tolerance < 0:
        parser.error("--soc-tolerance must be non-negative")
    if args.spike_threshold < 0:
        parser.error("--spike-threshold must be non-negative")

    zhang = prepare_zhang(args.zhang_dir, sequence_length=args.sequence_length)
    mohtat = prepare_mohtat(
        args.mohtat_dir,
        sequence_length=args.sequence_length,
        soc_target=args.soc,
        soc_tolerance=args.soc_tolerance,
        spike_threshold=args.spike_threshold,
    )
    _save_prepared(args.output_dir / "zhang.npz", "zhang", zhang)
    _save_prepared(args.output_dir / "mohtat.npz", "mohtat", mohtat)


if __name__ == "__main__":
    main()

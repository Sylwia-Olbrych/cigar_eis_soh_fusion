# CIGAR EIS-to-SOH

A reproducible example of Zhang/Mohtat electrochemical impedance spectroscopy
(EIS) preprocessing, CNN-BiLSTM SOH model training, and CIGAR-style prediction
fusion. Run commands from the project root.

## Publication

This repository is a demo of the work reported in:

> S. Olbrych, Z. X. Tung, S. Celik, H. A. Zhou, A. Abdelrazeq, D. U. Sauer,
> and R. H. Schmitt, “Application of CIGAR for collective learning between
> CNN–BiLSTM models in lithium-ion battery state of health prediction,”
> *Neural Computing and Applications*, vol. 38, no. 11 (2026).
> [https://doi.org/10.1007/s00521-026-12144-6](https://doi.org/10.1007/s00521-026-12144-6)

## Data

Raw datasets are not included. Obtain them from their authoritative sources
and follow their terms. Cite the exact releases and add any required
acknowledgements when publishing results.

- **Zhang:** The data originates from the publicly available dataset by Zhang
  et al. ([Zenodo, 2020](https://doi.org/10.5281/zenodo.3633835)), accompanying
  their study on lithium-ion battery degradation analysis using impedance
  spectroscopy ([Nature Communications, 2020](https://doi.org/10.1038/s41467-020-15235-7)).
  The pipeline uses capacity and state-V EIS from
  `data/raw/zhang/Capacity_data/` and `data/raw/zhang/EIS_data/` for cells
  `25C01, 25C02, 25C03, 25C05, 25C06, 25C07, 35C01, 35C02, 45C01, 45C02`.
- **Mohtat:** The source is the publicly available University of Michigan
  pouch-cell voltage and expansion cyclic ageing dataset by Mohtat et al.
  ([University of Michigan, 2021](https://doi.org/10.7302/7TW1-KC35)). For the
  dissertation work, this project uses the curated EIS representation provided
  by Blömeke et al. ([Journal of Power Sources, 2024](https://doi.org/10.1016/j.jpowsour.2024.235049)).
  Files are expected at `data/raw/mohtat/Mohtat_<number>.csv`, with columns
  `n`, `Capacity`, `SOC`, `EIS_Frequency`, `EIS_Z_abs`, and `EIS_Z_phase`.
  Default preprocessing selects SOC 0.9.

Raw data, prepared datasets, model checkpoints, and results are excluded from
Git. Preprocessing resamples EIS spectra to 60 real/imaginary points and saves
`.npz` files under `prepared_data/`.

## Install and run

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python prepare_data.py
python train_models.py
python cigar_eis_soh.py
```

The default commands prepare both datasets, train five cell-grouped folds per
domain (holding out Zhang `25C05` and Mohtat `MC21`), and run both fusion
directions. Checkpoints are saved under `models/`; prediction CSVs and metric
plots are saved under `results/`.

Override data paths, holdout cells, output directories, epochs, device, and
fusion options with each script's `--help`. For example:

```powershell
python train_models.py --zhang-epochs 500 --mohtat-epochs 800 --device cpu
python cigar_eis_soh.py --help
```

## Project layout and method

- `src/`: preprocessing modules and the shared CNN-BiLSTM (`model.py`)
- `prepare_data.py`: creates prepared datasets
- `train_models.py`: trains domain-specific ensembles
- `cigar_eis_soh.py`: inference, fusion, metrics, and plots
- `tests/`: automated tests (`python -m unittest discover -s tests -v`)

The model uses two input channels, a convolutional sequence adjustment, a
five-layer bidirectional LSTM, and fold-specific target scaling based only on
training cells. This is a compact research example, not evidence of validated
cross-domain generalization. Check dataset terms, provenance, and licenses
before redistribution or publication.

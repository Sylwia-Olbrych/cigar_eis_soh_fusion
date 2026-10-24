# Collective Model Fusion with CIGAR for EIS-Based Lithium-Ion Battery SoH Estimation

The implementation serves as a compact demonstration of collective model fusion with CIGAR [] for lithium-ion battery state-of-health (SoH) estimation using electrochemical impedance spectroscopy (EIS).
It reproduces the main preprocessing, model-training, and fusion steps in a reduced workflow, but is not intended to reproduce the dissertation results numerically.
Instead, it provides an accessible example of the corresponding experimental procedure and supports reproducibility of the underlying methodology.

## Publication

This repository provides a simplified implementation of the pipeline developed and evaluated in the following publication:

Olbrych, S., Tung, Z. X., Celik, S., Zhou, H. A., Abdelrazeq, A., Sauer, D. U., & Schmitt, R. H. (2026). Application of CIGAR for collective learning between CNN–BiLSTM models in lithium-ion battery state of health prediction. *Neural Computing and Applications, 38*(11). https://doi.org/10.1007/s00521-026-12144-6

### Citation

If you use this repository, its code, or the associated methodology in your research, please cite both the **original publication** and the **archived version of this repository on Zenodo**:

- **Publication:** [Olbrych et al. (2026)](https://doi.org/10.1007/s00521-026-12144-6)
- **Archived code:** [Olbrych-Nasuta et al. (2026)](https://zenodo.org/10.5281/zenodo.23623866)

## Data

Raw datasets are not included. Obtain them from their authoritative sources
and follow their terms. 

If you use the datasets in your research, please acknowledge the original data sources by citing the corresponding references listed below: **[1, 2] for the Zhang dataset** and **[3, 4] for the Mohtat dataset**..

- **Zhang:** The data originates from the publicly available dataset by Zhang
  et al. [2], which accompanies their study on lithium-ion battery degradation
  analysis using impedance spectroscopy [3].
  The pipeline uses capacity and state-V EIS from
  `data/raw/zhang/Capacity_data/` and `data/raw/zhang/EIS_data/` for cells
  `25C01, 25C02, 25C03, 25C05, 25C06, 25C07, 35C01, 35C02, 45C01, 45C02`.
- **Mohtat:** The source is the publicly available University of Michigan
  pouch-cell voltage and expansion cyclic ageing dataset by Mohtat et al. [4].
  For the dissertation work, this project uses the curated EIS representation
  provided by Blömeke et al. [5].
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
domain for 100 epochs each (holding out Zhang `25C05` and Mohtat `MC21`), and
run both fusion directions. Checkpoints are saved under `models/`; prediction
CSVs and metric plots are saved under `results/`.

Override data paths, holdout cells, output directories, epochs, device, and
fusion options with each script's `--help`. For example:

```powershell
python train_models.py --device cpu
python cigar_eis_soh.py --help
```

Both training runs default to the same 100-epoch setting for simplicity. Use
`--zhang-epochs` and `--mohtat-epochs` to override these values.

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

## References

[1] Hoang, M., Hoang, N., Low, B. K. H., & Kingsford, C.
(2019). Collective model fusion for multiple black-box experts. In Proceedings of the 36th International Conference on Machine Learning (Vol. 97, pp. 2742–2750). Proceedings of Machine Learning Research.
https://proceedings.mlr.press/v97/hoang19a.html

[2] Zhang, Y., Tang, Q., Zhang, Y., Wang, J., Stimming, U., & Lee, A. A.
(2020). *Identifying degradation patterns of lithium ion batteries from
impedance spectroscopy using machine learning* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.3633835

[3] Zhang, Y., Tang, Q., Zhang, Y., Wang, J., Stimming, U., & Lee, A. A.
(2020). Identifying degradation patterns of lithium ion batteries from
impedance spectroscopy using machine learning. *Nature Communications, 11*(1).
https://doi.org/10.1038/s41467-020-15235-7

[4] Mohtat, P., Siegel, J. B., & Stefanopoulou, A. G. (2021). *UofM pouch cell
voltage and expansion cyclic aging dataset* [Data set]. University of Michigan.
https://doi.org/10.7302/7TW1-KC35

[5] Blömeke, A., Kappelhoff, O., Wasylowski, D., Ringbeck, F., & Sauer, D. U.
(2024). Open source online electrochemical impedance spectroscopy data
analytics tool. *Journal of Power Sources, 615*, 235049.
https://doi.org/10.1016/j.jpowsour.2024.235049


# CIGAR EIS-to-SOH example

A small, self-contained example of CIGAR-style model fusion using two
electrochemical impedance spectroscopy (EIS) datasets: Zhang and Mohtat. It
contains the data converter, model training script, and fusion/inference
script. It does not require WandB or files at a developer-specific path.

## Data and attribution

The source datasets are intentionally not included. Obtain them from their
authoritative sources and follow their terms of use. Before publishing, add
the dataset citations, provenance, and any required acknowledgements for the
exact Zhang and Mohtat releases you use. The repository currently does not
provide a license for redistributing those measurements.

Expected source layout:

```text
data/raw/zhang/
  Capacity_data/Data_Capacity_25C01.txt
  EIS_data/EIS_state_V_25C01.txt
  ... (all 10 Zhang cells listed below)
data/raw/mohtat/
  Mohtat_01.csv
  ... (at least 6 distinct cells for the default five-fold training)
```

### Exactly which source data files?

The Zhang preparation script requires a capacity file and a state-V EIS file
for each of these 10 cells:

```text
25C01  25C02  25C03  25C05  25C06  25C07
35C01  35C02  45C01  45C02
```

For each cell, the filenames are
`Capacity_data/Data_Capacity_<cell>.txt` and
`EIS_data/EIS_state_V_<cell>.txt`. Other EIS states and cells are not used.

Mohtat preparation reads `Mohtat_<number>.csv`, with the six required columns
`n`, `Capacity`, `SOC`, `EIS_Frequency`, `EIS_Z_abs`, and `EIS_Z_phase`.
For the default five-fold training, provide at least six cells, including the
default held-out `MC21`; the other five or more cells are used for training.
More files are accepted. Mohtat EIS is selected at SOC 0.9 by default.

These directories are configured to be ignored by Git so you can keep
licensed or large source datasets locally without committing them. When you
move this folder to a new repository, keep the raw files in those same paths.
The `.gitkeep` files only preserve empty folders; they contain no dataset.

Zhang files are whitespace-separated and follow the source project's
zero-based column positions: cycle and step in positions 1 and 2, capacity in
position 3 for 25C01/25C05 (position 5 for the other 25C cells), and
real/imaginary impedance in positions 3 and 4 of the EIS file. Mohtat CSVs
need `n`, `Capacity`, `SOC`, `EIS_Frequency`,
`EIS_Z_abs`, and `EIS_Z_phase`. EIS is selected at SOC 0.9 by default.
Preprocessing converts both formats to 60-point real/imaginary sequences,
uses the source project's per-cell real/imaginary normalization, and computes
SOH ratios (`capacity / initial capacity`).
No separate pretrained weights are needed: `train_models.py` creates the
checkpoints under the ignored `models/` directory.

## Install and run

Use Python 3.10 or newer. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the source files in the ignored `data/raw/zhang` and `data/raw/mohtat`
folders above. Or override either location with:

```powershell
python prepare_data.py `
  --zhang-dir data\raw\zhang `
  --mohtat-dir data\raw\mohtat `
  --output-dir prepared_data
```

Train five cell-grouped folds per domain. The default held-out cells are
excluded from every training fold, so they can be used for evaluation. The
model architecture and main training settings match the existing project
configuration: 2 input channels, 128 hidden units, 5 bidirectional LSTM
layers, a 20-point convolution adjustment from 60 to 41 steps, and 0.3
dropout. Training uses batch size 128, Adam at `1e-4` with betas `(0.5,
0.999)`, 500 Zhang epochs, and 800 Mohtat epochs. As in the source training
model, recurrent states are randomly initialized during training; inference
uses zero states, matching the CIGAR model implementation.

```powershell
python train_models.py `
  --zhang-data prepared_data\zhang.npz `
  --mohtat-data prepared_data\mohtat.npz `
  --zhang-holdout 25C05 `
  --mohtat-holdout MC21 `
  --output-dir models
```

Run fusion on one prepared cell. Either domain can be selected as the input;
both trained model ensembles score the same EIS batch:

```powershell
python cigar_eis_soh.py `
  --data prepared_data\zhang.npz `
  --cell 25C05 `
  --zhang-models models\zhang `
  --mohtat-models models\mohtat `
  --output results\zhang_25C05.csv
```

The output CSV includes the measured SOH, both individual ensemble
predictions, and the fused prediction. The command also reports CIGAR RMSE
against the included target for this evaluation cell.

Training and inference default to CPU. Add `--device cuda` to either command
to use a CUDA GPU. `--zhang-epochs` and `--mohtat-epochs` can override the
project's training epoch counts; `--seed` controls repeatability.
`--iterations` and `--learning-rate` on `cigar_eis_soh.py` control fusion.

## Method and limitations

Each domain has a bidirectional LSTM ensemble. Folds are split by cell, with
the selected holdout cell excluded from every fold. SOH targets are min-max
scaled independently using each fold's training cells only, then restored
before prediction/fusion, so held-out targets do not affect training
normalization and model outputs and CIGAR candidate SOH use physical SOH
ratios. The ensemble score is the
derivative of its Gaussian log-likelihood with respect to candidate SOH,
using fold prediction variance with a small variance floor. CIGAR sums the
Zhang and Mohtat scores and applies the norm-scaled iterative update.

This is a compact reproducible baseline, not a claim of validated
cross-domain generalization. The parser reflects the source file layouts
described above; verify those layouts against the exact dataset releases you
use. Choose and add a code license before publishing the repository.

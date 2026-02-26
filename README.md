# Investigating the Predictive Power of Seismic Statistical Features Using Ensemble Learning

This repository contains the code for the paper:

> Quan W, Gorse D (2026) "Investigating the predictive power of seismic statistical features using ensemble learning." *PLoS ONE* 21(2): e0342765.  
> [https://doi.org/10.1371/journal.pone.0342765](https://doi.org/10.1371/journal.pone.0342765)

## Overview

We investigate whether the predictive value of seismic statistical features (SSFs) for earthquake prediction stems from their ability to capture domain-specific knowledge. We compare 60 SSFs against 428 generic time series features from [tsfresh](https://tsfresh.readthedocs.io/), training XGBoost models to predict whether an earthquake of magnitude M ≥ 5 will occur within the next 15 days.

Key finding: models using SSFs achieve AUCs up to **0.87**, while models using tsfresh features alone cannot substantially exceed random performance.

## Data

The earthquake catalogue data used in this study is available at:  
[https://doi.org/10.6084/m9.figshare.31048849](https://doi.org/10.6084/m9.figshare.31048849)

Download the data and place the CSV files in the `data/` directory.

## Installation

```bash
git clone https://github.com/<your-username>/tsf_origin.git
cd tsf_origin
pip install -r requirements.txt
```

**Python 3.9+** is recommended.

## Project Structure

```
├── main.py                    # Entry point for reproducing experiments
├── data_processor.py          # Data preprocessing pipeline
├── select_tune.py             # Feature selection & hyperparameter tuning
├── tsf_utils.py               # tsfresh feature extraction & imputation
├── seismic_indicator_com.py   # Seismic indicator utilities (Beta/Z recomputation)
├── utils.py                   # Shared utility functions
├── requirements.txt           # Python dependencies
├── data/                      # Input data directory (not tracked)
└── results/                   # Output directory for results
```

## Usage

### Full pipeline (from raw data)

Edit the configuration section in `main.py` to set your region and file names, then:

```bash
python main.py
```

By default, `main.py` is configured for the **Chile** region with a magnitude threshold of **M ≥ 5**. To run preprocessing from raw catalogue data, uncomment the `preprocess()` call in `main.py`.

### Using pre-processed data

If you already have the processed feature files (e.g., from the figshare dataset), place them in `data/` and run `main.py` directly. It will load the pre-processed CSVs and run feature selection, tuning, and evaluation.

### Pipeline stages

The experiment runs three comparisons for each region:

1. **Indicator features only** — 60 seismic statistical features
2. **tsfresh features only** — generic time series features extracted from the magnitude series
3. **Mixed features** — both combined

For each, the pipeline:
1. Performs initial hyperparameter tuning (GridSearchCV with PredefinedSplit)
2. Runs BorutaShap feature selection
3. Fine-tunes hyperparameters on selected features
4. Evaluates on the held-out test set (AUC, MCC, SHAP importance)

## Key Design Decisions

- **No data leakage in imputation**: tsfresh's built-in `impute` is intentionally avoided. Instead, missing/infinite values are replaced using only past data (see `tsf_utils.py`).
- **50-event gap**: A gap of 50 events (matching the sliding window size) is maintained between train/validation and train/test splits to prevent information leakage.
- **Beta/Z recomputation**: For the whole dataset, Beta and Z values are recomputed using only the training period as the background, since the original MATLAB-computed values use the full catalogue.
- **Normalisation**: Only the 60 seismic indicator features are z-score standardised (using training statistics). tsfresh features are left unnormalised as some are binary.

## Regions

The paper evaluates three seismically active regions:
- **Chile** (M ≥ 4.5 and M ≥ 5)
- **Japan** (M ≥ 5)
- **Switzerland** (M ≥ 2.5)

To run a different region, update the configuration constants at the top of `main.py`.

## Citation

```bibtex
@article{quan2026investigating,
  title={Investigating the predictive power of seismic statistical features using ensemble learning},
  author={Quan, Wei and Gorse, Denise},
  journal={PLoS ONE},
  volume={21},
  number={2},
  pages={e0342765},
  year={2026},
  publisher={Public Library of Science},
  doi={10.1371/journal.pone.0342765}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

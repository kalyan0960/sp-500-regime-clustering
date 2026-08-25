# Reproducibility

## Environment

The project requires Python 3.12 (`>=3.12,<3.13`). The commands below target Windows PowerShell and use the repository's Python 3.12 environment:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

No machine-specific absolute paths are required. Run commands from the repository root.

## Inputs and order

Required local inputs are:

- `data/raw/spy_daily.csv`
- `data/raw/vix_daily.csv`
- `configs/research_config.yaml`
- the saved checkpoints in `data/processed/` required by later stages

The revised pipeline order is:

1. Notebook 05 validates and aligns SPY/VIX data.
2. Notebook 06 or the corresponding feature utility creates `data/processed/market_features_pre_garch.csv`.
3. Notebook 07 creates GARCH volatility and `data/processed/market_features_with_garch.csv`.
4. Notebook 08 creates the revised K-means baseline and `outputs/tables/kmeans_model_comparison.csv`.
5. Run `\.venv312\Scripts\python.exe scripts\run_hmm_candidates.py`.
6. Run `\.venv312\Scripts\python.exe scripts\run_hmm_finalist_analysis.py`.
7. Run `\.venv312\Scripts\python.exe scripts\run_selected_hmm.py`.
8. Run `\.venv312\Scripts\python.exe scripts\run_transition_targets.py`.
9. Run `\.venv312\Scripts\python.exe scripts\run_volume_analysis.py`.
10. Run `\.venv312\Scripts\python.exe scripts\run_logistic_regression.py`.
11. Run `\.venv312\Scripts\python.exe scripts\run_final_synthesis.py`.

Notebooks 01-04 are the preserved original exploratory baseline. Notebook 13 loads canonical saved outputs for presentation and validation; it does not refit accepted models.

The expensive HMM candidate search and finalist analysis ordinarily do not need to be rerun when their accepted saved outputs, hashes, and metadata are unchanged. The selected HMM, transition, volume, logistic, and final-synthesis stages can likewise be skipped when their accepted artifacts are being audited rather than regenerated. Do not download new data or refit candidate models for a documentation-only closeout.

## Reproducibility controls

The configured random seed is 42. The selected HMM reconstruction uses seed 44 and saved Stage 10B parameters. Logistic selection uses five expanding chronological folds with a five-observation purge, training-only scaling and threshold selection, and a test period untouched during fitting. The paired logistic metric bootstrap uses block length 20, 2,000 replications, and seed 13013. The Stage 12 volume bootstrap uses block length 20, 2,000 replications, and seed 12012. The training cutoff is 2017-12-31; the test period begins 2018-01-01. The primary transition horizon is five trading observations, with 1, 10, and 20 as robustness horizons.

## Validation

Focused synthesis tests:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q tests/test_synthesis.py
```

Complete test suite:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
```

Dependency consistency:

```powershell
.\.venv312\Scripts\python.exe -m pip check
```

Notebook 13 should have five executed code cells and zero error outputs. The synthesis validation report should state that required inputs are valid, accepted models were not refit, ten tables and seven figures were created, JSON values are finite, and substantive outputs are deterministic.

## Generated outputs

Generated tables and model metadata are written to `outputs/tables/` and `outputs/models/`. The final synthesis writes ten tables, seven figures, JSON summaries, the input hash manifest, and the validation report under `outputs/final_synthesis/`. Intermediate processed checkpoints are written under `data/processed/`.

Generated data, models, figures, temporary directories, Python environments, caches, and notebook checkpoints are ignored by `.gitignore`. Source code, tests, configuration, documentation, and the preserved historical figures remain trackable. `.test-temp/` and `stage*-test-tmp/` are temporary test-artifact locations, not research outputs.

## Warnings and interpretation

HMM candidate fitting can emit convergence warnings for unstable candidate specifications; failed or non-converged candidates are recorded and are not silently accepted. Statistical routines may emit warnings related to covariance estimates, small secondary event samples, or multiple-comparison methods. These are nonfatal only when the saved validation metadata and tests confirm the accepted artifacts. Warnings do not establish model validity.

All reported predictive results use filtered, not smoothed, state information. The project is research code, not trading advice. It makes no causal or trading-profit claim.

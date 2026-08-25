# Market Regime Identification and Abnormal-Volume Transition Prediction

## Overview

This repository preserves an original exploratory K-means study and implements a leakage-safe SPY market-regime pipeline using GARCH volatility, a time-aware HMM, transition analysis, abnormal-volume association tests, and chronological logistic prediction.

The final synthesis finds four economically interpretable HMM states. The Stage 12 ANOVA found no meaningful overall abnormal-volume difference, but did find a volatility-dependent training association. Stage 13 found no stable general incremental abnormal-volume predictability out of sample: PR AUC improved slightly, but ROC AUC, log loss, calibration, and the interaction model did not show consistent improvement. Conditional and transition-specific findings remain exploratory.

## Project layout

```text
configs/                 Research configuration
data/                    Local raw, processed, and external data checkpoints
docs/                    Research design and preserved original progress narrative
notebooks/               Original Week 1-5 exploratory K-means notebooks
outputs/                 Generated figures, tables, and model artifacts
src/market_regime/       Reusable research-code package
tests/                   Validation and regression tests
```

The original progress narrative is preserved at [docs/ORIGINAL_PROGRESS.md](docs/ORIGINAL_PROGRESS.md). The two historical K-means figures are retained in `outputs/figures/`.
The final results are summarized in [docs/FINAL_RESULTS.md](docs/FINAL_RESULTS.md), reproducibility steps are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), and notebook roles are listed in [docs/NOTEBOOK_GUIDE.md](docs/NOTEBOOK_GUIDE.md).

## Setup

Python 3.12 is the project target. The existing Python 3.14 environments are retained for historical work, but the complete planned dependency set could not be verified as reliably available for Python 3.14.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project is configured as a `src`-layout package. During development, it can be installed in editable mode with `pip install -e .`.

## Final synthesis

Stages 6–14 are implemented; Notebooks 01–04 remain the preserved exploratory baseline. Reproduce the paper-ready synthesis after generating prior-stage outputs with:

```powershell
.\.venv312\Scripts\python.exe scripts\run_final_synthesis.py
```

Machine-readable results, ten final tables, validation metadata, and seven figures are written to `outputs/final_synthesis/`. Run `python -m pytest -q` for repository-wide validation.

The project distinguishes descriptive analysis, association, prediction, and causation. The revised methods test predictive value on a later unseen period; they do not establish causation, trading profitability, or investment advice. This is research code, not trading advice.

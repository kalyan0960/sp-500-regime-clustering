# Market Regime Identification and Abnormal-Volume Transition Prediction

## Overview

This repository preserves an original exploratory K-means study of S&P 500 market regimes and provides the structure for a revised research pipeline. The original work is retained as a historical baseline. The revised pipeline is planned and will examine SPY market regimes with GARCH conditional volatility, a hidden Markov model (HMM), and chronological regression tests of abnormal volume as a predictor of future regime transitions.

No revised GARCH, HMM, ANOVA, or regression results have been implemented in this repository yet. The methodological specification is available in [docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md).

## Project layout

```text
configs/                 Research configuration
data/                    Local raw, processed, and external data checkpoints
docs/                    Research design and preserved original progress narrative
notebooks/               Original Week 1-5 exploratory K-means notebooks
outputs/                 Generated figures, tables, and model artifacts
src/market_regime/       Planned reusable research-code package
tests/                   Planned validation and regression tests
```

The original progress narrative is preserved at [docs/ORIGINAL_PROGRESS.md](docs/ORIGINAL_PROGRESS.md). The two historical K-means figures are retained in `outputs/figures/`.

## Setup

Python 3.12 is the project target. The existing Python 3.14 environments are retained for historical work, but the complete planned dependency set could not be verified as reliably available for Python 3.14.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project is configured as a `src`-layout package. During development, it can be installed in editable mode with `pip install -e .`.

## Research status

- Original K-means notebooks: preserved exploratory baseline.
- Stage 6 data validation and Stage 7 pre-GARCH feature construction: implemented.
- Stage 8 GARCH conditional-volatility estimation: implemented; HMM regimes remain planned.
- Transition outcomes, supplementary ANOVA, and logistic regression: planned.

The project distinguishes descriptive analysis, association, prediction, and causation. The revised methods are intended to test predictive value on a later unseen period; they do not, by themselves, establish causation.

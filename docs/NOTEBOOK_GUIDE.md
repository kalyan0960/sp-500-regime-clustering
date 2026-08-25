# Notebook Guide

The repository contains three kinds of notebooks: the original exploratory work, the revised modeling pipeline, and the results/presentation notebook. Notebook numbers reflect the research progression; they are not interchangeable versions of the same analysis.

## Original exploratory work

| Notebook | Purpose | Estimates a model or loads saved results? | Primary output |
| --- | --- | --- | --- |
| `01_data_acquisition.ipynb` | Download historical ^GSPC data and build the original simple-return, intraday-volatility, and abnormal-volume fields. | No fitted model; creates features. | `data/sp500_engineered_features.csv` |
| `02_exploratory_analysis.ipynb` | Inspect distributions, correlations, time patterns, and exploratory relationships in the original features. | No fitted model. | Exploratory figures and summaries. |
| `03_kmeans_clustering.ipynb` | Standardize original features and fit the initial K-means cluster comparisons. | Estimates K-means models. | Original cluster assignments, centroids, and evaluation figures. |
| `04_regime_validation.ipynb` | Compare and describe the original cluster labels and their market behavior. | Loads and validates saved exploratory results. | Original regime-validation figures and tables. |

## Revised modeling pipeline

| Notebook | Purpose | Estimates a model or loads saved results? | Primary output |
| --- | --- | --- | --- |
| `05_spy_vix_data_validation.ipynb` | Validate SPY and VIX inputs, dates, required fields, and alignment. | No fitted model. | `data/processed/spy_vix_aligned_raw.csv` |
| `06_feature_engineering.ipynb` | Build leakage-safe returns, prior-only abnormal volume, and 252/60-day drawdowns. | No fitted model; creates features. | `data/processed/market_features_pre_garch.csv` |
| `07_garch_volatility.ipynb` | Fit training-period GARCH specifications and recursively generate conditional volatility. | Estimates GARCH models. | GARCH parameters, diagnostics, and `data/processed/market_features_with_garch.csv` |
| `08_revised_kmeans.ipynb` | Fit the matched-feature K-means baseline with training-only scaling. | Estimates K-means models. | `outputs/tables/kmeans_model_comparison.csv` and baseline figures. |
| `09_hmm_regime_analysis.ipynb` | Explore HMM regime identification, candidate structures, filtering, profiles, and diagnostics. | Estimates or inspects HMM models depending on the cell. | HMM candidate and selected-regime diagnostics. |
| `10_transition_targets.ipynb` | Construct leakage-safe future transition outcomes at multiple horizons. | No fitted model; creates targets. | `outputs/tables/market_with_transition_targets.csv` and target summaries. |
| `11_volume_transition_analysis.ipynb` | Test training-period abnormal-volume association, volatility interaction, and transition-specific differences. | Estimates statistical association models and tests. | Volume association tables, robust tests, and figures. |
| `12_logistic_transition_prediction.ipynb` | Fit chronological logistic models and evaluate volume’s incremental test-period predictive value. | Estimates logistic prediction models. | Logistic metrics, coefficients, predictions, and bootstrap comparisons. |

## Results/presentation notebook

| Notebook | Purpose | Estimates a model or loads saved results? | Primary output |
| --- | --- | --- | --- |
| `13_final_empirical_synthesis.ipynb` | Present the final research questions, selected HMM, Stage 12 association findings, Stage 13 prediction findings, limitations, and reproducibility checks. | Loads saved canonical results; does not refit accepted models. | Final synthesis tables and validation summary under `outputs/final_synthesis/`. |

## Recommended advisor walkthrough

For the research design and progression, show `01_data_acquisition.ipynb` through `04_regime_validation.ipynb` as the preserved exploratory baseline, then `05_spy_vix_data_validation.ipynb`, `06_feature_engineering.ipynb`, `08_revised_kmeans.ipynb`, `09_hmm_regime_analysis.ipynb`, `11_volume_transition_analysis.ipynb`, and `12_logistic_transition_prediction.ipynb` as the revised evidence chain. Use `13_final_empirical_synthesis.ipynb` as the concise final presentation. Notebook 07 and Notebook 10 are important implementation checkpoints when the advisor wants to inspect volatility construction or target definitions.

The final conclusion is deliberately narrower than a trading claim: abnormal volume did not provide stable general incremental out-of-sample predictive value. Conditional and transition-specific findings remain exploratory.

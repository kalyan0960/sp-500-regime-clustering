# Hypothesis and Week 1-5 Progress

## Research Hypothesis

I expect daily U.S. equity market regimes to be detectable with unsupervised clustering, and I expect changes in trading volume (captured by `Abnormal_Volume`) to provide evidence that regime boundaries are behaviorally distinct rather than random noise.

## Mathematical Definitions I Used (Plain Language)

I use **simple daily return** from `pct_change()` to keep interpretation intuitive:

- Daily return: `Returns_t = (Close_t - Close_{t-1}) / Close_{t-1}`
- Intraday volatility proxy: `Volatility_t = (High_t - Low_t) / Open_t`
- Abnormal volume: `Abnormal_Volume_t = Volume_t / MA20(Volume)_t`

For clustering, I standardize each feature:

- `z_{t,j} = (x_{t,j} - mean_j) / std_j`

Then I use K-Means, which finds clusters by minimizing within-cluster squared distance:

- `Objective = sum over clusters and points of ||x_i - centroid_k||^2`

This keeps the math explicit while still understandable for non-technical readers.

## Operational Version of the Hypothesis

1. A K-Means model on `Returns`, `Volatility`, and `Abnormal_Volume` will identify stable clusters.
2. The detected clusters will differ in market behavior (especially volatility and return direction).
3. Regimes with stressed or extreme price behavior will also show systematic shifts in abnormal volume.

## Scope Control (Current Stage)

I am currently finalizing this repository through **Week 5** only:

- Week 1: I finalized my research question and direction.
- Week 2: I selected the historical market data source and variables.
- Week 3: I completed data cleaning and feature engineering.
- Week 4: I completed exploratory analysis and built my first clustering model.
- Week 5: I am refining and evaluating the clusters.

I am intentionally deferring Week 6-8 items (full interpretation write-up, full paper drafting, and revisions).

## Evidence Targets for Week 5

- Document why `K=3` is selected (Elbow + Silhouette).
- Quantify regime size and regime feature centroids.
- Show direct evidence that `Abnormal_Volume` contributes to meaningful regime structure.

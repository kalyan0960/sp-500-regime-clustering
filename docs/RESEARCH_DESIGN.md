# Market Regime Identification and Abnormal-Volume Transition Prediction

## 1. Study purpose

This project examines whether daily market regimes can be identified in a time-aware manner and whether abnormal trading volume improves forecasts of future regime transitions. The central research question is:

> Can time-aware market regimes be identified using daily SPY market information, and does abnormal SPY trading volume add useful information for predicting future regime transitions?

The project distinguishes among four forms of evidence. Descriptive analysis summarizes historical patterns. Statistical association describes relationships conditional on a specified model. Prediction evaluates performance on future, unseen observations. Causation requires a stronger research design and is not established by this observational study.

## 2. Relationship to the original exploratory analysis

Notebooks 01-04 document the original Week 1-5 exploratory K-means analysis. They remain an historical baseline and are not replaced by this design. That work used daily S&P 500 index data, an intraday range proxy for volatility, and an abnormal-volume measure to describe clusters and their timing.

The revised framework is designed to address the main limitations of that exploratory approach. It uses SPY, a traded S&P 500 proxy with an observable share-volume series; uses a GARCH conditional-volatility measure rather than treating an intraday range as conditional volatility; keeps volume outside the regime-identification variables; and evaluates volume through chronologically out-of-sample transition prediction. K-means remains a useful similarity-based benchmark, while the HMM is the planned primary time-aware regime model.

The original K-means results found a higher silhouette score for two clusters than for three. Accordingly, the historical three-cluster result is not described as silhouette-optimal. Three regimes are an expected, interpretable starting point for the revised comparison, not a preselected empirical result.

## 3. Data, sample, and quality controls

SPY will serve as the traded proxy for the S&P 500. SPY volume will be used because the S&P 500 index itself has no directly traded share-volume measure. The CBOE Volatility Index (^VIX) will provide an external implied-volatility measure for validation and use as a regression control.

The main sample spans 2000-01-01 through 2026-05-31. Observations through 2017-12-31 form the training period, and observations from 2018-01-01 through 2026-05-31 form the test period. The merged dataset will retain common valid trading dates. Missing observations will be documented; long gaps will not be automatically forward-filled.

Data quality checks will establish that dates are sorted and unique, required columns are present, and volume is strictly positive before abnormal-volume calculations are made. The zero-volume observation on 2023-05-24 in the earlier ^GSPC dataset is treated as a data-quality issue rather than a genuine zero-volume market event. The revised SPY data will nevertheless be checked for `Volume > 0` before analysis.

Pandas DataFrames are the principal analytical data structure. Because the sample contains only several thousand daily observations, reproducible CSV checkpoints are appropriate in place of a database.

## 4. Variables and feature construction

Let \(P_t\) denote SPY's adjusted closing price on trading day \(t\), and let \(V_t\) denote SPY's share volume on day \(t\). The subscript \(t-1\) refers to the preceding trading day.

### 4.1 Returns

Simple return supports direct descriptive interpretation:

\[
R_t = \frac{P_t-P_{t-1}}{P_{t-1}}.
\]

Here, \(R_t\) is the percentage change in adjusted closing price from day \(t-1\) to day \(t\). Log return is used in the planned GARCH and HMM models:

\[
r_t = \log(P_t)-\log(P_{t-1}).
\]

In this expression, \(r_t\) is the log return and \(\log\) is the natural logarithm. Prices must be positive for this calculation.

### 4.2 Conditional volatility and drawdown

The planned conditional-volatility measure is the conditional standard deviation, \(\sigma_t\), from a GARCH(1,1) model with Student-t innovations. This model is appropriate because financial returns commonly show volatility clustering and heavy tails. The GARCH component is planned methodology; it has not yet been implemented.

For prediction at date \(t\), \(\sigma_t\) will be estimated without access to observations after date \(t\). This time ordering prevents look-ahead leakage.

The principal drawdown is a 252-trading-day drawdown:

\[
DD^{252}_t = \frac{P_t}{\max(P_{t-251},\ldots,P_t)}-1.
\]

Here, \(DD^{252}_t\) is the percentage distance from the highest adjusted closing price observed in the current 252-trading-day window, and \(\max(\cdot)\) denotes the maximum in that window. A 60-trading-day drawdown, \(DD^{60}_t\), will be examined as a robustness measure using the same formula with a 60-day window. Drawdown should not exceed zero except for small numerical tolerance.

VIX is an external implied-volatility measure. In the regression equations, \(VIX_t\) denotes its value on day \(t\). It will be used for validation and as a regression control, but it will not be an observed variable in the main HMM.

### 4.3 Abnormal volume

Abnormal volume is the principal predictor under examination:

\[
AV_t=\frac{V_t}{\operatorname{mean}(V_{t-20},\ldots,V_{t-1})}.
\]

In this equation, \(AV_t\) is abnormal volume on day \(t\), and the denominator is the mean of the preceding 20 trading days' volume. The current day's volume is deliberately excluded from its own benchmark so that the measure represents information available at day \(t\). In pandas, this corresponds to applying `shift(1)` before `rolling(20)`.

The analysis will also use log abnormal volume:

\[
LogAV_t=\log(AV_t).
\]

The continuous values of \(AV_t\) and \(LogAV_t\) are retained. An observation with \(AV_t>1\) is above its recent average, but is not automatically treated as an economically important shock. Training-period-defined high- and low-volume tails may later be examined as robustness checks.

## 5. Planned regime-identification methodology

### 5.1 Main hidden Markov model

The planned primary regime model is a hidden Markov model (HMM). Let \(S_t\) denote the unobserved market regime on day \(t\). The HMM will use the observed feature vector

\[
X_t = (r_t,\sigma_t,DD^{252}_t),
\]

where \(X_t\) is the set of observed market features, \(r_t\) is log return, \(\sigma_t\) is leakage-safe GARCH conditional volatility, and \(DD^{252}_t\) is 252-day drawdown. The HMM component is planned methodology and has not yet been implemented.

Abnormal volume and VIX are excluded from \(X_t\). This separation is intentional: it prevents abnormal volume from helping define the regimes that it will later be asked to predict, and it preserves VIX as an external validation measure and regression control.

Models with two through five hidden states will be compared. Estimation will begin with diagonal covariance matrices because they are more parsimonious and may be more stable for a modest sample. Full covariance models will be considered only when they are estimable and stable. Multiple random initializations will be used because HMM estimates can depend on starting values.

The comparison will consider Bayesian information criterion (BIC), convergence, stability across initializations, state size, transition probabilities, episode duration, one-day episodes, and the economic interpretation of return, volatility, and drawdown profiles. BIC balances in-sample fit against model complexity. The number of states will therefore be selected from the evidence rather than fixed in advance.

States will initially remain anonymous. Descriptive names will be assigned only after their return, volatility, and drawdown profiles have been examined. The complete state-probability matrix will be retained, and dates with a maximum state probability below 0.60 will be flagged as uncertain.

Smoothed states use information from the full sample and are suitable only for retrospective historical description. Predictive analysis will rely on filtered or otherwise real-time-safe state information available at date \(t\). A full-sequence decoded state will not be described as filtered when future observations contributed to it.

### 5.2 Revised K-means baseline

A revised K-means baseline is planned for comparison with the HMM. It will use the same three regime features, \(r_t\), \(\sigma_t\), and \(DD^{252}_t\), rather than abnormal volume or VIX. This matched feature set makes the comparison more informative and avoids circularly using volume to construct and test regimes.

The scaler will be fitted on training-period observations only and then applied unchanged to test-period observations. Values of \(K\) from 2 through 5 will be reported transparently. K-means cluster numbers have no intrinsic meaning and will be relabelled only after their feature profiles have been reviewed. This revised K-means analysis is planned methodology; it has not yet been implemented.

## 6. Planned transition outcomes

The transition analysis will define forward-looking outcomes at horizons \(h\in\{1,5,10,20\}\) trading days, with \(h=5\) as the primary horizon. The forecast origin is day \(t\). Dates without sufficient future observations through \(t+h\) will remain missing rather than being coded as zero.

Let \(s^*\) denote the state later identified as stressed after profile-based labelling. A state change counts even if the state returns to its original value before the end of the horizon.

| Outcome | Definition |
| --- | --- |
| Any transition, \(Y_{any}(t,h)\) | Equals 1 when any state change occurs from \(t+1\) through \(t+h\); otherwise equals 0 when the full horizon is observed. |
| Entry into stress, \(Y_{stress}(t,h)\) | Equals 1 when the market enters \(s^*\) at least once from \(t+1\) through \(t+h\). |
| Exit from stress, \(Y_{exit}(t,h)\) | Equals 1 when the market leaves \(s^*\) at least once from \(t+1\) through \(t+h\). |

These outcomes will be constructed with real-time-safe state information for the predictive setting. They are planned methodology and have not yet been implemented.

## 7. Planned predictive regression analysis

Binary logistic regression is the principal predictive test. For any transition outcome \(Y_{t,h}\), \(P(Y_{t,h}=1)\) is the probability of an event during the next \(h\) trading days, and \(\operatorname{logit}(p)=\log[p/(1-p)]\) is the log-odds transformation of a probability \(p\).

The baseline model is

\[
\operatorname{logit}[P(Y_{t,h}=1)] = \beta_0+\beta_1R_t+\beta_2\sigma_t+\beta_3DD^{252}_t+\beta_4VIX_t+\text{current-regime indicators}.
\]

The volume model adds log abnormal volume:

\[
\operatorname{logit}[P(Y_{t,h}=1)] = \beta_0+\beta_1R_t+\beta_2\sigma_t+\beta_3DD^{252}_t+\beta_4VIX_t+\beta_5LogAV_t+\text{current-regime indicators}.
\]

The secondary model adds the interaction

\[
\beta_6(LogAV_t\times\sigma_t).
\]

In these equations, \(\beta_0\) is the intercept; \(\beta_1\) through \(\beta_6\) are coefficients for the terms shown; and current-regime indicators represent the real-time-safe regime information available at date \(t\), with one regime omitted as the reference category. The interaction tests whether the association of abnormal volume with transition risk differs with conditional volatility.

The formal volume-coefficient hypothesis is \(H_0:\beta_5=0\) against \(H_A:\beta_5\ne0\), where \(H_0\) is the null hypothesis and \(H_A\) is the alternative hypothesis. The principal empirical question, however, is whether adding \(LogAV_t\) improves prediction on the unseen test period, not only whether its p-value is below 0.05.

The regression analysis is planned methodology and has not yet been implemented. Evaluation will use the chronological training/test split rather than random splitting. Reported results will include coefficients, odds ratios, confidence intervals, p-values, predicted-probability changes, ROC-AUC, precision, recall, F1 score, Brier score, log loss, calibration, class balance, and naive benchmarks. Overlapping horizons and serial dependence will be addressed through block-bootstrap inference, time-series-appropriate inference, and/or non-overlapping-horizon robustness samples.

Logistic regression measures predictive association. It does not by itself demonstrate that abnormal volume causes a market transition.

## 8. Planned supplementary statistical analysis

The supplementary two-factor analysis will use continuous abnormal volume as the outcome. The factors will be transition type and a low, medium, or high GARCH-volatility category, together with their interaction. In schematic form,

\[
AV_t=\alpha+\text{transition-type effect}+\text{GARCH-category effect}+\text{interaction}+\varepsilon_t.
\]

Here, \(\alpha\) is the intercept and \(\varepsilon_t\) is the residual, or unexplained, component. GARCH-category thresholds will be calculated from the training period only and then applied unchanged to later observations. Categorising continuous GARCH volatility loses information; for that reason, this analysis is supplementary to the continuous-variable logistic regression rather than the main predictive test.

Welch one-way ANOVA, Kruskal-Wallis tests, Games-Howell post-hoc comparisons, effect sizes, and block-bootstrap or episode-level robustness analyses may also be used where appropriate. ANOVA is retrospective comparison, not prediction. Abnormal-volume differences across regimes that were themselves constructed with abnormal volume will not be treated as independent confirmatory evidence. These analyses are planned methodology and have not yet been implemented.

## 9. Data products, reproducibility, and testing

The analysis will retain documented CSV checkpoints for raw SPY data, raw VIX data, merged market data, engineered features, K-means results, HMM states and probabilities, the transition dataset, statistical-result tables, and regression-result tables. Generated data should be reproducible from source code even if it is not tracked by Git. Fixed random seeds and package versions will be documented to support replication.

The implementation will include tests for the following conditions:

1. Dates are sorted and unique, and required data columns are present.
2. Volume is positive before abnormal-volume analysis, and day \(t\)'s volume is excluded from its own benchmark.
3. Drawdown does not exceed zero apart from numerical tolerance.
4. Training statistics, thresholds, scalers, transformations, and models do not use test observations.
5. Manually constructed state sequences produce the intended transition outcomes.
6. Dates lacking sufficient future observations remain missing.
7. Predictors dated \(t\) contain no information after \(t\), and predictive states do not use future observations.
8. Results are reproducible under fixed random seeds.

## 10. Interpretation

The original K-means analysis is retained as exploratory evidence about similarity in daily observations. The revised HMM is intended to provide the main time-aware regime description. The supplementary ANOVA framework compares historical groups, while logistic regression provides the principal test of whether abnormal volume improves transition prediction.

No individual model, statistical test, or illustrative historical date can establish causation. The value of the revised design lies in its explicit time ordering, separation of regime construction from volume testing, and evaluation on a later unseen period.

# Identifying Market Regimes via Volumetric Signatures

## Overview
This project examines whether unsupervised machine learning methodologies can identify distinct market regimes in U.S. equity markets. Specifically, it applies K-Means clustering to daily S&P 500 returns, volatility, and trading volume to detect regimes and analyzes whether abnormal trading volume signatures reliably precede regime transitions.

## Project Architecture
* **Phase 1: Data Engineering:** Ingesting and scaling historical S&P 500 data (`^GSPC`) via Python.
* **Phase 2: Machine Learning Pipeline:** Deploying unsupervised clustering to mathematically define market regimes without human bias.
* **Phase 3: Regime Analysis:** Evaluating the dates of identified regime shifts against $t-1$ to $t-5$ volumetric behavior.
* **Phase 4: Synthesis:** Formally documenting findings and visualizing transition boundaries.

## Technology Stack
* Python (Pandas, NumPy)
* Scikit-Learn (K-Means Clustering)
* yfinance API
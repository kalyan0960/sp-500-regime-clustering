# Coding Agent Guide

- Read `docs/RESEARCH_DESIGN.md` in full before making a research-related change.
- Preserve Notebooks 01-04 as the original Week 1-5 exploratory K-means baseline. Do not delete, overwrite, or silently revise that progression.
- Obtain an explicit methodological decision before replacing a finalized model or variable.
- Put reusable calculations in `src/`, use pandas DataFrames for analysis, and save documented CSV checkpoints at major stages.
- Prevent look-ahead leakage. Fit scalers, thresholds, transformations, and models on training data only.
- Keep work reproducible with fixed random seeds and documented package versions.
- Do not silently change statistical definitions or research decisions. Use clear names and define mathematical symbols in plain language.
- Distinguish description, association, prediction, and causation. Report failed tests, convergence problems, and limitations honestly.
- Run relevant code and tests before reporting completion.

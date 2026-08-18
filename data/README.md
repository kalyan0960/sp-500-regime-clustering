# Data directory

`raw/` is for source files obtained directly from data providers. `external/` is for supplementary externally sourced inputs, such as VIX data. `processed/` is for reproducible merged datasets and engineered feature checkpoints.

Generated datasets should be reproducible from the research code and are generally ignored by Git. No private, confidential, licensed, or otherwise restricted data should be committed. Data-quality checks and source provenance belong with the code and documented checkpoints.

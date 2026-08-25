"""Safe loading and validation for the research configuration."""

from datetime import date
from pathlib import Path
from typing import Any


def load_research_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate its fixed project constraints."""
    import yaml

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Research configuration must be a mapping.")

    data = config["data"]
    dates = {
        name: date.fromisoformat(data[name])
        for name in ("start_date", "training_end", "test_start", "end_date")
    }
    if not (
        dates["start_date"] <= dates["training_end"] < dates["test_start"] <= dates["end_date"]
    ):
        raise ValueError("Configured dates are not logically ordered.")

    features = config["features"]
    transitions = config["transitions"]
    positive_values = [
        features["abnormal_volume_window"],
        features["primary_drawdown_window"],
        features["robustness_drawdown_window"],
        transitions["primary_horizon"],
        *transitions["robustness_horizons"],
    ]
    if any(not isinstance(value, int) or value <= 0 for value in positive_values):
        raise ValueError("Feature windows and transition horizons must be positive integers.")
    if transitions["primary_horizon"] in transitions["robustness_horizons"]:
        raise ValueError("The primary horizon cannot appear in robustness horizons.")

    repository_root = config_path.resolve().parents[1]
    for directory in config["storage"].values():
        if not (repository_root / directory).is_dir():
            raise ValueError(f"Configured directory does not exist: {directory}")

    return config

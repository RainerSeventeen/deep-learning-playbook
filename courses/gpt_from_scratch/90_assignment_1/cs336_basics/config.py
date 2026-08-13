from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a YAML training configuration."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, Mapping):
        raise ValueError("config must be a YAML mapping")

    required_sections = ("model", "optimizer", "training")
    for section in required_sections:
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"config section '{section}' must be a mapping")

    return dict(config)

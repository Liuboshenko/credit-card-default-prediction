import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import FEATURE_NAMES


def validate_and_preprocess(data: dict) -> pd.DataFrame:
    """Validate input JSON and convert to a single-row DataFrame ordered by FEATURE_NAMES."""
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    try:
        row = {f: float(data[f]) for f in FEATURE_NAMES}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"All feature values must be numeric: {exc}") from exc

    return pd.DataFrame([row], columns=FEATURE_NAMES)


def get_feature_names() -> list:
    return list(FEATURE_NAMES)

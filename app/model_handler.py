import os
import joblib
import numpy as np
from typing import Dict, List, Tuple


class ModelHandler:
    """Load multiple versioned sklearn Pipeline models and run inference."""

    def __init__(self, model_paths: Dict[str, str]):
        self._models: Dict[str, object] = {}
        for version, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model '{version}' not found at '{path}'. "
                    "Run 'python models/train_model.py' first."
                )
            self._models[version] = joblib.load(path)

    def predict(self, version: str, features: np.ndarray) -> Tuple[int, float]:
        """Return (class_label, default_probability) for one sample."""
        if version not in self._models:
            raise ValueError(
                f"Unknown model version '{version}'. "
                f"Available: {self.available_versions}"
            )
        pipeline = self._models[version]
        label = int(pipeline.predict(features)[0])
        probability = float(pipeline.predict_proba(features)[0][1])
        return label, probability

    @property
    def available_versions(self) -> List[str]:
        return list(self._models.keys())

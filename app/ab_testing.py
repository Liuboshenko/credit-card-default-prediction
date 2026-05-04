import random
import threading
from typing import Dict, List


class ABTestingManager:
    """Thread-safe manager for A/B traffic splitting between two model versions."""

    def __init__(self, versions: List[str], split: float = 0.5):
        if len(versions) != 2:
            raise ValueError("Exactly two versions required for A/B testing.")
        self._versions = versions
        self._split = split
        self._lock = threading.Lock()
        self._stats: Dict[str, Dict] = {
            v: {
                'requests': 0,
                'default_count': 0,
                'probability_sum': 0.0,
            }
            for v in versions
        }

    def assign_version(self) -> str:
        """Return 'v1' with probability split, 'v2' otherwise."""
        return self._versions[0] if random.random() < self._split else self._versions[1]

    def record(self, version: str, prediction: int, probability: float) -> None:
        with self._lock:
            s = self._stats[version]
            s['requests'] += 1
            s['default_count'] += prediction
            s['probability_sum'] += probability

    def get_stats(self) -> Dict:
        with self._lock:
            result = {}
            for v, s in self._stats.items():
                n = s['requests']
                result[v] = {
                    'requests': n,
                    'default_rate': round(s['default_count'] / n, 4) if n else None,
                    'avg_probability': round(s['probability_sum'] / n, 4) if n else None,
                }
            return result

    @property
    def versions(self) -> List[str]:
        return list(self._versions)

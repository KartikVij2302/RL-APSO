import numpy as np

# Minimal signal model used by SPSO.
# Matches the simple exponential attenuation used elsewhere in the repo.
ALPHA = 0.01
SOURCE_POWER = 1.0


def measure_signal(position: np.ndarray, source: np.ndarray) -> float:
    d = np.linalg.norm(position - source)
    return float(SOURCE_POWER * np.exp(-ALPHA * d * d))

import numpy as np

# ---- Experiment / model constants ----
ALPHA = 0.01            # signal attenuation factor
SOURCE_POWER = 100.0    # signal amplitude at source
SPEED = 10.0            # constant UAV speed in m/s (as required)
T = 1.0                 # sampling interval (s); displacement per iter = SPEED * T
TERMINATION_DIST = 0.1  # termination threshold (m)

def measure_signal(position: np.ndarray, source: np.ndarray) -> float:
        d = np.linalg.norm(position - source)
        return SOURCE_POWER * np.exp(-ALPHA * d * d)
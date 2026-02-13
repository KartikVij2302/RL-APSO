import torch
import os
import numpy as np
import random

def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # for more deterministic CUDA behaviour (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
import numpy as np
from typing import Callable

def make_correlated_pns(memory_depth: int = 8, corr_strength: float = 0.25) -> Callable:
    """
    Correlated PNS: stealing probability depends on recent multiphoton frequency.
    """
    window = []
    def adapter(photons: np.ndarray, mu_vec=None) -> np.ndarray:
        fwd = photons.copy()
        for i, n in enumerate(fwd):
            recent_multi = sum(1 for x in window if x >= 2)
            steal_prob = min(1.0, corr_strength * (recent_multi / max(1, len(window))))
            if n >= 2 and np.random.rand() < steal_prob:
                fwd[i] = max(n - 1, 1)
            window.append(n)
            if len(window) > memory_depth:
                window.pop(0)
        return fwd
    return adapter

import numpy as np
from typing import Callable

def make_independent_pns(split_keep_at_least_one: bool = True) -> Callable:
    """
    Idealized PNS: for n>=2, Eve takes 1 photon (stores) and forwards n-1 (≥1 if flag set),
    else forwards n as-is.
    """
    def adapter(photons: np.ndarray, mu_vec=None) -> np.ndarray:
        fwd = photons.copy()
        mask = fwd >= 2
        fwd[mask] = np.maximum(fwd[mask] - 1, 1 if split_keep_at_least_one else 0)
        return fwd
    return adapter

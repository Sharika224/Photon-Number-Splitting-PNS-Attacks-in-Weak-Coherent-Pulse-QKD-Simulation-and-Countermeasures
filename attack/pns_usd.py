import numpy as np
from typing import Callable

def make_hybrid_pns_usd(success_prob: float = 0.1) -> Callable:
    """
    Hybrid: PNS + occasional Unambiguous State Discrimination (USD) success
    modeled as extra stealing opportunities without added error.
    """
    def adapter(photons: np.ndarray, mu_vec=None) -> np.ndarray:
        fwd = photons.copy()
        # base PNS
        mask = fwd >= 2
        fwd[mask] = np.maximum(fwd[mask] - 1, 1)
        # USD opportunity: if pulse was single-photon, sometimes Eve learns bit w/o disturbance
        if mu_vec is not None:
            single = (photons == 1)
            usd_hits = single & (np.random.rand(len(photons)) < success_prob)
            # USD doesn't change photon number forwarded; tracking of learned bits is left to analytics.
            # We keep number the same; external metrics can count usd_hits if needed.
        return fwd
    return adapter

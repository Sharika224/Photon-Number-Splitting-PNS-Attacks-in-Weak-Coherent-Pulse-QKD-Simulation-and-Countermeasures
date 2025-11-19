import numpy as np
from ..core.utils import binary_entropy

def gllp_keyrate(q_sift: float, Qs: float, Es: float, f_ec: float,
                 Y1L: float, e1U: float, mu_signal: float) -> float:
    """Asymptotic GLLP rate per signal pulse."""
    P1 = np.exp(-mu_signal) * mu_signal
    Q1 = Y1L * P1
    return q_sift * (Q1 * (1 - binary_entropy(e1U)) - f_ec * Qs * binary_entropy(Es))

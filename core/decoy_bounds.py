import numpy as np

class DecoyEstimates:
    def __init__(self, Y0_lower: float, Y1_lower: float, e1_upper: float):
        self.Y0_lower = Y0_lower
        self.Y1_lower = Y1_lower
        self.e1_upper = e1_upper

def vacuum_weak_bounds(mu_s: float, mu_w: float,
                       Qs: float, Qw: float, Q0: float,
                       Es: float, Ew: float, e0: float = 0.5) -> DecoyEstimates:
    """Standard vacuum+weak (analytical) bounds, asymptotic form."""
    Y0L = Q0
    S = np.exp
    denom = mu_s * mu_w - mu_w ** 2
    if denom <= 0:  # fallback safe-guard
        return DecoyEstimates(Y0L, 0.0, 0.5)
    Y1L = (mu_s / denom) * (Qw * S(mu_w) - (mu_w ** 2 / mu_s ** 2) * Qs * S(mu_s)
                            - ((mu_s ** 2 - mu_w ** 2) / mu_s ** 2) * Y0L)
    Y1L = max(0.0, Y1L)
    e1U_den = max(mu_w * Y1L, 1e-16)
    e1U_num = Ew * Qw * S(mu_w) - e0 * Y0L
    e1U = np.clip(e1U_num / e1U_den, 0.0, 0.5)
    return DecoyEstimates(Y0L, Y1L, e1U)

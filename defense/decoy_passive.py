import numpy as np

def passive_decoy_intensities(n_pulses: int, mu_signal: float, fluct: float) -> np.ndarray:
    """Uniform fluctuation in [(1-fluct), (1+fluct)] * mu_signal; with a small vacuum sprinkle."""
    mu = mu_signal * np.random.uniform(1 - fluct, 1 + fluct, size=n_pulses)
    vac = np.random.rand(n_pulses) < 0.02
    mu[vac] = 0.0
    mu[mu < 0] = 0.0
    return mu

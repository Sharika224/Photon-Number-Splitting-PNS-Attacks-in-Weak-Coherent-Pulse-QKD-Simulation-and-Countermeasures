import numpy as np

class WeakCoherentSource:
    """Poissonian emitter per pulse."""
    def __init__(self, mu: float):
        self.mu = mu

    def emit(self, n_pulses: int) -> np.ndarray:
        return np.random.poisson(self.mu, size=n_pulses)

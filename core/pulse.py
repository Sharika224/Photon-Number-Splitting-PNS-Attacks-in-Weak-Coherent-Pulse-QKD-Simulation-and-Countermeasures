import numpy as np
from dataclasses import dataclass
from .source import WeakCoherentSource

@dataclass
class PulseEngine:
    """Event-level engine building the stream of pulses with (passive) intensity fluctuations."""
    mu_nominal: float
    passive_fluct: float  # e.g., 0.2 -> uniform in [0.8, 1.2] * mu

    def intensities(self, n_pulses: int) -> np.ndarray:
        fluct = np.random.uniform(1 - self.passive_fluct, 1 + self.passive_fluct, size=n_pulses)
        mu_vec = np.clip(self.mu_nominal * fluct, 0.0, None)
        # sprinkle some true vacuum for tomography realism
        idx = np.random.rand(n_pulses) < 0.02
        mu_vec[idx] = 0.0
        return mu_vec

    def emit_photons(self, mu_vec: np.ndarray) -> np.ndarray:
        photons = np.random.poisson(mu_vec)
        return photons

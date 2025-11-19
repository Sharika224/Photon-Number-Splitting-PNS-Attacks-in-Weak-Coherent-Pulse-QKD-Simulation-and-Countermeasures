import numpy as np
from .utils import channel_transmittance

class FiberChannel:
    def __init__(self, distance_km: float, loss_db_per_km: float):
        self.eta = channel_transmittance(distance_km, loss_db_per_km)

    def transmit(self, photons_per_pulse: np.ndarray) -> np.ndarray:
        # binomial thinning per pulse
        return np.random.binomial(photons_per_pulse, self.eta)

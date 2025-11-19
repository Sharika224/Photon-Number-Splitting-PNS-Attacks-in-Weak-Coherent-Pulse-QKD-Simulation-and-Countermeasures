import numpy as np
from dataclasses import dataclass

@dataclass
class AnomalyDetector:
    delta_signal: float = 0.005
    delta_decoy: float = 0.005
    alerts: int = 0

    def check(self, Q_signal_obs: float, Q_signal_ref: float,
              Q_weak_obs: float,   Q_weak_ref: float) -> bool:
        cond = (abs(Q_signal_obs - Q_signal_ref) > self.delta_signal) or \
               (abs(Q_weak_obs   - Q_weak_ref)   > self.delta_decoy)
        if cond:
            self.alerts += 1
            return True
        return False

from dataclasses import dataclass
import numpy as np
import math
from typing import Tuple

def set_seed(seed: int = 1337):
    np.random.seed(seed)

def poisson_pmf(n: np.ndarray, mu: float) -> np.ndarray:
    n = np.asarray(n, dtype=int)
    return np.exp(-mu) * (mu ** n) / np.vectorize(math.factorial)(n)

def binary_entropy(x: float) -> float:
    if x <= 0 or x >= 1:
        return 0.0
    return -x*np.log2(x) - (1-x)*np.log2(1-x)

@dataclass
class SysParams:
    # source / decoys
    mu_signal: float = 0.5
    passive_fluct: float = 0.20  # ±20% around mu_signal (passive decoy)
    # channel & detectors
    distance_km: float = 100.0
    loss_db_per_km: float = 0.2
    detector_eff: float = 0.15
    dark_count: float = 1e-6
    misalign_err: float = 0.015
    # protocol
    sift_prob: float = 0.5
    f_ec: float = 1.16
    # finite-key
    n_pulses: int = 1_000_000
    epsilon_sec: float = 1e-10
    # binning thresholds for passive decoy classification
    weak_threshold: float = 0.35   # as fraction of mu_signal
    signal_threshold: float = 0.85 # as fraction of mu_signal

def channel_transmittance(distance_km: float, loss_db_per_km: float) -> float:
    return 10 ** (-loss_db_per_km * distance_km / 10.0)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def classify_mu(mu_array: np.ndarray, mu_sig: float,
                weak_thr: float, sig_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks: vacuum(~0), weak, signal for passive-decoy binning."""
    vacuum = mu_array < 1e-3
    weak = (mu_array >= 1e-3) & (mu_array < weak_thr * mu_sig)
    signal = mu_array >= sig_thr * mu_sig
    return vacuum, weak, signal

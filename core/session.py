import numpy as np
from dataclasses import dataclass
from typing import Dict
from .utils import SysParams, classify_mu, clamp01, binary_entropy
from .pulse import PulseEngine
from .channel import FiberChannel
from .detector import ThresholdDetector
from .decoy_bounds import vacuum_weak_bounds

@dataclass
class SessionResult:
    Q_signal: float; E_signal: float
    Q_weak: float;   E_weak: float
    Q_vac: float;    E_vac: float
    Y0L: float; Y1L: float; e1U: float
    R_asym: float; R_finite: float
    qber_total: float; clicks_total: int

class QKDSession:
    def __init__(self, P: SysParams):
        self.P = P
        self.pulse = PulseEngine(P.mu_signal, P.passive_fluct)
        self.chan = FiberChannel(P.distance_km, P.loss_db_per_km)
        self.det  = ThresholdDetector(P.detector_eff, P.dark_count)

    def run_once(self, attack_adapter=None) -> SessionResult:
        P = self.P
        mu_vec = self.pulse.intensities(P.n_pulses)
        photons = self.pulse.emit_photons(mu_vec)

        # optional adversary (in-place transform of photons)
        if attack_adapter is not None:
            photons = attack_adapter(photons, mu_vec)

        # channel + detection
        transmitted = self.chan.transmit(photons)
        clicks = self.det.click_vector(transmitted)

        # QBER from misalignment over clicked events
        if clicks.sum() > 0:
            errors = (np.random.rand(len(clicks)) < P.misalign_err) & (clicks == 1)
            qber_total = errors.sum() / clicks.sum()
        else:
            qber_total = 0.0

        # passive-decoy classification
        vac_m, weak_m, sig_m = classify_mu(mu_vec, P.mu_signal, P.weak_threshold, P.signal_threshold)

        def gain_and_error(mask):
            n = max(mask.sum(), 1)
            Q = clicks[mask].mean() if mask.any() else 0.0
            if clicks[mask].sum() > 0:
                errs = (np.random.rand(mask.sum()) < P.misalign_err) & (clicks[mask] == 1)
                E = errs.sum() / clicks[mask].sum()
            else:
                E = 0.0
            return Q, E

        Q0, E0 = gain_and_error(vac_m)
        Qw, Ew = gain_and_error(weak_m)
        Qs, Es = gain_and_error(sig_m)

        # Estimate single-photon terms using signal/weak/vacuum (vacuum+weak bounds)
        mu_w_guess = max(P.mu_signal * 0.25, 1e-3)  # effective weak intensity bucket center (rough)
        est = vacuum_weak_bounds(P.mu_signal, mu_w_guess, Qs, Qw, Q0, Es, Ew)

        # GLLP asymptotic key rate (per *signal* pulse)
        from .utils import binary_entropy
        P1 = np.exp(-P.mu_signal) * P.mu_signal
        Q1 = est.Y1L * P1
        R_asym = P.sift_prob * (Q1 * (1 - binary_entropy(est.e1U)) - P.f_ec * Qs * binary_entropy(Es))

        # Finite-key (simple conservative penalty)
        delta = 3.0 * np.sqrt(np.log(1/P.epsilon_sec) / max(1, P.n_pulses)) / np.log(2)
        R_finite = max(0.0, R_asym - delta)

        return SessionResult(
            Q_signal=Qs, E_signal=Es, Q_weak=Qw, E_weak=Ew, Q_vac=Q0, E_vac=E0,
            Y0L=est.Y0_lower, Y1L=est.Y1_lower, e1U=est.e1_upper,
            R_asym=R_asym, R_finite=R_finite,
            qber_total=qber_total, clicks_total=int(clicks.sum())
        )

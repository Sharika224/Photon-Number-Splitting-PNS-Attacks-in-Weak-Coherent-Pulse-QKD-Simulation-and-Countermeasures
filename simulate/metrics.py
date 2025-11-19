"""
Metrics module to evaluate the performance of hybrid QKD defense simulations.
Implements QBER, SKR, ADP, runtime efficiency, and CRI calculations.
"""

import numpy as np

def compute_qber(clicks: int, errors: int) -> float:
    return errors / clicks if clicks > 0 else 0.0

def compute_skr(Qs, Es, f_ec, Y1L, e1U, mu_signal, q_sift=0.5):
    from ..core.utils import binary_entropy
    P1 = np.exp(-mu_signal) * mu_signal
    Q1 = Y1L * P1
    return q_sift * (Q1 * (1 - binary_entropy(e1U)) - f_ec * Qs * binary_entropy(Es))

def compute_attack_detection_probability(delta_values, threshold=0.005):
    """Simulate Attack Detection Probability (ADP)."""
    if not len(delta_values):
        return 0.0
    delta_values = np.abs(np.array(delta_values))
    detected = np.sum(delta_values > threshold)
    return detected / len(delta_values)

def compute_runtime_efficiency(num_pulses, time_seconds):
    """Efficiency: pulses processed per second."""
    if time_seconds <= 0:
        return 0
    return num_pulses / time_seconds

def compute_resilience_index(skr_attack, skr_ideal):
    """CRI = SKRattack / SKRideal"""
    return skr_attack / skr_ideal if skr_ideal > 0 else 0.0

def summarize_metrics(run_outputs):
    """
    Given list of run dictionaries (from simulate_run),
    summarize mean metrics across configurations.
    """
    skr_asym = [r["results"]["R_asym"] for r in run_outputs]
    skr_fin = [r["results"]["R_finite"] for r in run_outputs]
    qber = [r["results"]["qber_total"] for r in run_outputs]
    cri = [compute_resilience_index(f, a) for f, a in zip(skr_fin, skr_asym)]
    return {
        "mean_R_asym": np.mean(skr_asym),
        "mean_R_finite": np.mean(skr_fin),
        "mean_QBER": np.mean(qber),
        "mean_CRI": np.mean(cri),
    }

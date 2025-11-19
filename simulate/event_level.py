"""
Event-level simulation orchestrator for Weak Coherent Pulse QKD
under Photon-Number Splitting (PNS) attacks and hybrid defenses.

This module integrates all major building blocks:
 - core (photon emission, channel, detector)
 - attack (Eve adapters)
 - defense (decoy-state, finite-key, anomaly detection)
and returns a structured dictionary with computed metrics.
"""

import numpy as np
from dataclasses import asdict
from ..core.utils import SysParams, set_seed
from ..core.session import QKDSession
from ..attack.adversary import make_independent_pns
from ..attack.pns_correlated import make_correlated_pns
from ..attack.pns_usd import make_hybrid_pns_usd
from ..defense.anomaly import AnomalyDetector

def simulate_run(
    attack_type: str = "independent",
    correlation_strength: float = 0.3,
    memory_depth: int = 8,
    success_prob_usd: float = 0.1,
    distance_km: float = 100.0,
    pulses: int = 500_000,
    seed: int = 42
):
    """Run one full QKD session with chosen PNS variant and return metrics."""
    set_seed(seed)
    P = SysParams(distance_km=distance_km, n_pulses=pulses)
    session = QKDSession(P)

    if attack_type == "independent":
        attack_adapter = make_independent_pns()
    elif attack_type == "correlated":
        attack_adapter = make_correlated_pns(memory_depth=memory_depth, corr_strength=correlation_strength)
    elif attack_type == "hybrid_usd":
        attack_adapter = make_hybrid_pns_usd(success_prob=success_prob_usd)
    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

    result = session.run_once(attack_adapter=attack_adapter)

    # anomaly detection based on expected reference rates (approx baseline)
    ad = AnomalyDetector()
    baseline_Qs, baseline_Qw = 0.02, 0.007
    detected = ad.check(result.Q_signal, baseline_Qs, result.Q_weak, baseline_Qw)

    return {
        "params": asdict(P),
        "results": result.__dict__,
        "anomaly_detected": detected,
        "alerts": ad.alerts,
    }

if __name__ == "__main__":
    out = simulate_run(attack_type="correlated", distance_km=150)
    import json
    print(json.dumps(out, indent=2, default=float))

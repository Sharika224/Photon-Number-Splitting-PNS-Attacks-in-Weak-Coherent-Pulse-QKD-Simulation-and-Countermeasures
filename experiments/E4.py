import json
from src.qkd.core.utils import SysParams, set_seed
from src.qkd.core.session import QKDSession
from src.qkd.attack.pns_correlated import make_correlated_pns
from src.qkd.defense.anomaly import AnomalyDetector

if __name__ == "__main__":
    set_seed(2025)
    P = SysParams(n_pulses=300_000)
    sess = QKDSession(P)
    res = sess.run_once(attack_adapter=make_correlated_pns(corr_strength=0.3))
    # reference "no attack" baselines (you can precompute/store from clean run)
    Qs_ref, Qw_ref = 0.02, 0.007
    ad = AnomalyDetector(delta_signal=0.003, delta_decoy=0.002)
    flagged = ad.check(res.Q_signal, Qs_ref, res.Q_weak, Qw_ref)
    print(json.dumps({"session": res.__dict__, "anomaly_flagged": flagged, "alerts": ad.alerts}, indent=2))

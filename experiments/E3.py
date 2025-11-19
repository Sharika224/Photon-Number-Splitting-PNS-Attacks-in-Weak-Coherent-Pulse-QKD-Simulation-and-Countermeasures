import json
from src.qkd.core.utils import SysParams, set_seed
from src.qkd.core.session import QKDSession
from src.qkd.attack.pns_correlated import make_correlated_pns

if __name__ == "__main__":
    set_seed(1337)
    strengths = [0.0, 0.1, 0.25, 0.5]
    results = []
    for s in strengths:
        P = SysParams(n_pulses=200_000)
        sess = QKDSession(P)
        res = sess.run_once(attack_adapter=make_correlated_pns(corr_strength=s))
        results.append({"corr_strength": s, **res.__dict__})
    print(json.dumps(results, indent=2, default=float))

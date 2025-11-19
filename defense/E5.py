import json
from src.qkd.core.utils import SysParams, set_seed
from src.qkd.core.session import QKDSession
from src.qkd.attack.adversary import make_independent_pns

if __name__ == "__main__":
    set_seed(99)
    distances = [50, 100, 150, 200, 250]
    out = []
    for d in distances:
        P = SysParams(distance_km=d, n_pulses=300_000)
        res = QKDSession(P).run_once(attack_adapter=make_independent_pns())
        out.append({"distance_km": d, "R_asym": res.R_asym, "R_finite": res.R_finite,
                    "Q_signal": res.Q_signal, "E_signal": res.E_signal})
    print(json.dumps(out, indent=2, default=float))

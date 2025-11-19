import json
from src.qkd.core.utils import SysParams, set_seed
from src.qkd.core.session import QKDSession
from src.qkd.attack.adversary import make_independent_pns

if __name__ == "__main__":
    set_seed(42)
    # passive decoys are built-in (via P.passive_fluct). Compare two fluct levels.
    configs = [
        ("passive_low", 0.10),
        ("passive_high", 0.25),
    ]
    out = {}
    for name, fluct in configs:
        P = SysParams(passive_fluct=fluct, n_pulses=200_000)
        R = QKDSession(P).run_once(attack_adapter=make_independent_pns())
        out[name] = R.__dict__
    print(json.dumps(out, indent=2, default=float))

import json
from src.qkd.core.utils import SysParams, set_seed
from src.qkd.core.session import QKDSession
from src.qkd.attack.adversary import make_independent_pns

if __name__ == "__main__":
    set_seed(7)
    P = SysParams(distance_km=100.0, n_pulses=1_000_00)  # 100k pulses quick demo
    sess = QKDSession(P)
    res = sess.run_once(attack_adapter=make_independent_pns())
    print(json.dumps(res.__dict__, indent=2, default=float))

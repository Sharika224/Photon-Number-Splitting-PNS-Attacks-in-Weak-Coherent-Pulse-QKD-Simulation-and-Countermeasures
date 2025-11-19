"""
Compare different attack/defense configurations
and generate structured outputs for visualization or benchmarking.
"""

import json
from time import time
import numpy as np
from .event_level_sim import simulate_run
from .metrics import summarize_metrics, compute_resilience_index

def compare_attack_types(distances=(50, 100, 150, 200)):
    results = []
    for d in distances:
        for attack in ["independent", "correlated", "hybrid_usd"]:
            start = time()
            out = simulate_run(attack_type=attack, distance_km=d)
            runtime = time() - start
            res = {
                "distance": d,
                "attack": attack,
                **out["results"],
                "runtime_s": runtime,
                "anomaly": out["anomaly_detected"],
            }
            results.append(res)
    return results

def summarize_comparison(results):
    grouped = {}
    for r in results:
        key = (r["attack"], r["distance"])
        grouped.setdefault(key, []).append(r)
    summary = {}
    for k, runs in grouped.items():
        summary[str(k)] = summarize_metrics([{"results": r} for r in runs])
    return summary

if __name__ == "__main__":
    res = compare_attack_types()
    summary = summarize_comparison(res)
    print(json.dumps(summary, indent=2, default=float))

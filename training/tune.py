"""Tune the rules on the DFL harness. One-at-a-time sweep around the defaults.

    python solution/tune.py            # -> opendata/results/tune.txt (appended as it goes)

Each setting is scored on all 7 damaged DFL matches x 2 damage seeds (1, 2; seed 0 was used for
the v1/v2 comparison). Differences under ~0.01 mean spellF1 are noise: keep the default.
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "solution")); sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import opendata as od, rules_v2 as v2, possession_metrics as pm

SWEEP = {
    "r_contact": [0.9, 1.5, 2.0],
    "margin": [0.25, 1.0],
    "control_frames": [6, 15],
    "control_episodes": [3],
    "lookahead": [2.0, 6.0],
    "episode_gap": [10],
    "dead_share": [0.35, 0.45],
    "blackout_boost": [1.0, 1.3],
    "dead_min": [3.0, 6.0],
    "pass_speed": [3.0, 6.0],
    "pass_max": [2.0, 5.0],
    "restart_clear": [2.0, 5.0],
    "min_poss": [0.5, 2.0],
    "min_dead": [2.0, 5.0],
}
SEEDS = (1, 2)


def configs():
    yield "default", {}
    for k, vals in SWEEP.items():
        for v in vals:
            yield "%s=%s" % (k, v), {k: v}


def main(extra=None):
    cfgs = list(configs()) if extra is None else extra
    out = os.path.join(od.OD, "results", "tune.txt" if extra is None else "tune_final.txt")
    scores = {name: [] for name, _ in cfgs}
    t0 = time.time()
    for m in od.MATCHES:
        M = od.load(m)
        gtc = od.to_codes(od.controlled(M["lab"], M["segs"]), len(M["lab"]))
        for seed in SEEDS:
            P, B, _ = od.damage(M, seed)
            F = None
            for name, params in cfgs:
                pred = v2.label_match(P, B, M["team"], M["pitch"], M["segs"], params=params)
                r = pm.evaluate(gtc, od.to_codes(pred, len(pred)), bootstrap_n=0)
                scores[name].append((r["spell_f1_mean"], r["macro_f1"]))
            del P, B
        with open(out, "a") as fh:
            fh.write("after %s (%.0f min):\n" % (m, (time.time() - t0) / 60))
            base = np.mean([s[0] for s in scores["default"]]) if "default" in scores else 0
            for name, _ in cfgs:
                s = np.array(scores[name])
                fh.write("  %-22s spellF1 %.4f (%+.4f)  macroF1 %.4f\n" % (name, s[:, 0].mean(), s[:, 0].mean() - base, s[:, 1].mean()))
    return scores


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()

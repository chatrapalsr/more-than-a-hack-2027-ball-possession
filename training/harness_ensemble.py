"""Score GRU (D), TCN (E) and their average on the 7 Bundesliga matches, using the saved out-of-fold
probabilities. Rebuilds the 1 s answer key and the rules labels locally (same damage seeds as in training).

    python solution/harness_ensemble.py
"""
import os, sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["MTAH_T"] = "1.0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("", "solution", os.path.join("solution", "kaggle"), "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, p))
import opendata as od, rules_v2 as v2, decode
from common import MATCHES, SEEDS, spell_f1

OUT = os.path.join(ROOT, "solution", "models")
CACHE = os.path.join(ROOT, "solution", "_cache", "harness_rules")
os.makedirs(CACHE, exist_ok=True)
pd_ = np.load(os.path.join(OUT, "d", "track_d_oof.npz"))
pe_ = np.load(os.path.join(OUT, "e", "track_e_oof.npz"))
sd = json.load(open(os.path.join(OUT, "d", "track_d_summary.json")))
se = json.load(open(os.path.join(OUT, "e", "track_e_summary.json")))

rows = []
for m in MATCHES:
    M = od.load(m)
    gt = od.controlled(M["lab"], M["segs"], T=1.0)
    for s in SEEDS:
        k = "%s_%d" % (m, s)
        f = os.path.join(CACHE, k + ".npy")
        if os.path.isfile(f):
            rules = np.load(f)
        else:
            P, B, _ = od.damage(M, s)
            rules = v2.label_match(P, B, M["team"], M["pitch"], M["segs"]).astype(np.int8)
            np.save(f, rules)
            del P, B
        pd, pe = pd_[k].astype(np.float32), pe_[k].astype(np.float32)
        r = dict(match=m, seed=s, rules=spell_f1(gt, rules)[0])
        for name, prob in (("D", pd), ("E", pe), ("DE", (pd + pe) / 2)):
            for mode in ("model", "rules_dead"):
                for pen, mp in ((32.0, 1.0), (64.0, 1.0), (32.0, 2.0)):
                    lab = decode.decode(prob, M["segs"], pen, mp, rules=rules, mode=mode)
                    r["%s|%s|%g|%g" % (name, mode, pen, mp)] = spell_f1(gt, lab)[0]
        rows.append(r)
        print(m, s, "rules %.3f  D %.3f  E %.3f  DE %.3f (model, pen32 mp1)" % (
            r["rules"], r["D|model|32|1"], r["E|model|32|1"], r["DE|model|32|1"]), flush=True)

keys = [k for k in rows[0] if "|" in k] + ["rules"]
print("\nmean spellF1 over %d matches x %d seeds (1 s labels):" % (len(MATCHES), len(SEEDS)))
for k in sorted(keys, key=lambda k: -np.mean([r[k] for r in rows])):
    print("  %-26s %.4f" % (k, np.mean([r[k] for r in rows])))
de, d, e = ([np.mean([r[k] for r in rows if r["match"] == m]) for m in MATCHES]
            for k in ("DE|rules_dead|32|1", "D|rules_dead|32|1", "E|rules_dead|32|1"))
print("\nper match, hybrid pen32 mp1 (DE / D / E):")
for m, a, b, c in zip(MATCHES, de, d, e):
    print("  %s  %.3f  %.3f  %.3f" % (m, a, b, c))
json.dump(rows, open(os.path.join(ROOT, "opendata", "results", "harness_ensemble.json"), "w"), indent=1)

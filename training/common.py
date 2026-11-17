"""Shared by the GPU training scripts: load features, score, choose the decoder."""
import glob, json, os
import numpy as np

MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
SEEDS = (0, 1, 2)


def load_all(feats_dir):
    D = {}
    for m in MATCHES:
        for s in SEEDS:
            d = np.load(os.path.join(feats_dir, "%s_%d.npz" % (m, s)))
            D[(m, s)] = dict(X=d["X"], gt=d["gt"], v2=d["v2"], segs=[tuple(x) for x in d["segs"]],
                             names=list(d["names"]))
    return D


def spell_f1(gt, lab):
    import possession_metrics as pm
    L = np.array(["H", "A", "D", "X"], dtype=object)
    r = pm.evaluate(pm.label_codes(L[np.where(gt < 0, 3, gt)]), pm.label_codes(L[np.where(lab < 0, 3, lab)]), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


def choose_decoder(D, oof, out_path, name):
    """oof[(m,s)] = prob[n,3]. Grid over Viterbi switch penalty and min possession; report vs v2."""
    import decode
    np.savez_compressed(out_path.replace("_summary.json", "_oof.npz"),
                        **{"%s_%d" % k: v.astype(np.float16) for k, v in oof.items()})
    grid = [(mode, pen, mp) for mode in ("model", "rules_dead")
            for pen in (32.0, 64.0, 128.0) for mp in (1.0, 2.0)]
    res = {}
    for mode, pen, mp in grid:
        sc = [spell_f1(D[k]["gt"], decode.decode(oof[k], D[k]["segs"], pen, mp, rules=D[k]["v2"], mode=mode))
              for k in oof]
        res["%s,pen=%g,min_poss=%g" % (mode, pen, mp)] = np.mean(sc, 0).tolist()
        print(name, mode, pen, mp, "spellF1 %.4f macroF1 %.4f" % tuple(np.mean(sc, 0)), flush=True)
    v2 = np.mean([spell_f1(D[k]["gt"], D[k]["v2"]) for k in oof], 0).tolist()
    best = max(res, key=lambda k: res[k][0])
    per_match = {}
    mode = best.split(",")[0]
    pen, mp = [float(x.split("=")[1]) for x in best.split(",")[1:]]
    for m in MATCHES:
        ks = [k for k in oof if k[0] == m]
        per_match[m] = dict(
            model=float(np.mean([spell_f1(D[k]["gt"], decode.decode(oof[k], D[k]["segs"], pen, mp, rules=D[k]["v2"], mode=mode))[0] for k in ks])),
            v2=float(np.mean([spell_f1(D[k]["gt"], D[k]["v2"])[0] for k in ks])))
    summary = dict(method=name, v2_defaults=v2, grid=res, best=best, best_score=res[best], per_match=per_match,
                   penalty=pen, min_poss=mp, mode=mode)
    json.dump(summary, open(out_path, "w"), indent=1)
    print("BEST", name, best, res[best], "| v2 rules", v2, flush=True)
    return summary

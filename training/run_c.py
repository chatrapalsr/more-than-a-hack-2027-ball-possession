"""Model C (dropped): XGBoost (GPU) per-frame H/A/D model, leave-one-match-out, Viterbi decode.
Outputs: track_c_model.json, track_c_summary.json (OOF spellF1 vs v2 rules on the same frames)."""
import glob, os, sys, time
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/kprep.py", recursive=True)[0]))
import kprep
kprep.run()
import numpy as np, xgboost as xgb
import common

t0 = time.time()
D = common.load_all(kprep.FEATS)
names = D[(common.MATCHES[0], 0)]["names"]
STEP = 2                               # train on every 2nd frame (neighbours are near-duplicates)
PARAMS = dict(objective="multi:softprob", num_class=3, tree_method="hist", device="cuda",
              max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=20, eval_metric="mlogloss", seed=0)
ROUNDS = 600


def xy(keys):
    X = np.concatenate([D[k]["X"][::STEP] for k in keys]); y = np.concatenate([D[k]["gt"][::STEP] for k in keys])
    ok = y >= 0
    return X[ok], y[ok]


oof = {}
for m in common.MATCHES:
    tr = [k for k in D if k[0] != m]
    X, y = xy(tr)
    bst = xgb.train(PARAMS, xgb.DMatrix(X, label=y), ROUNDS)
    for k in [k for k in D if k[0] == m]:
        oof[k] = bst.predict(xgb.DMatrix(D[k]["X"]))
    print("fold", m, "done %.0f s" % (time.time() - t0), flush=True)
summ = common.choose_decoder(D, oof, "/kaggle/working/track_c_summary.json", "C-xgb")
X, y = xy(list(D))
bst = xgb.train(PARAMS, xgb.DMatrix(X, label=y), ROUNDS)
bst.save_model("/kaggle/working/track_c_model.json")
imp = bst.get_score(importance_type="gain")
top = sorted(imp.items(), key=lambda kv: -kv[1])[:15]
print("top features:", [(names[int(k[1:])], round(v, 1)) for k, v in top])
print("total %.0f s" % (time.time() - t0))

import shutil
for _d in ("od", "feats", "code"):
    shutil.rmtree("/kaggle/working/" + _d, ignore_errors=True)

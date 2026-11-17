"""Inference for the trained sequence models (GRU = d, TCN = e) on the provided match.

    python solution/predict_learned.py d|e [mode]
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "solution"))
import features, decode, rules_pipeline as v1
from games import GAME, GAMES, half_video_blocks

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def bar_features():
    cache = os.path.join(ROOT, "solution", "_cache", "bar_features.npz")
    P, B, team = v1.load_match()
    segs = [(a, b) for _, a, b in half_video_blocks(GAME)]
    forced = np.zeros(len(B), bool); forced[segs[1][0]:v1.KICKOFF_2ND] = True
    if os.path.isfile(cache):
        d = np.load(cache)
        return d["X"], segs, forced
    X, names, lab = features.build(P, B, team, GAMES[GAME]["pitch"], segs, forced)
    np.savez_compressed(cache, X=X)
    return X, segs, forced


def prob_c(X):
    import xgboost as xgb
    bst = xgb.Booster(); bst.load_model(os.path.join(OUT, "c", "track_c_model.json"))
    return bst.predict(xgb.DMatrix(X))


def prob_d(X, segs):
    import torch, torch.nn as nn
    ck = torch.load(os.path.join(OUT, "d", "track_d_model.pt"), map_location="cpu", weights_only=False)

    class Net(nn.Module):
        def __init__(self, nin, h=128):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(nin, h), nn.GELU(), nn.Dropout(0.1))
            self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * h, 3)

        def forward(self, x):
            return self.out(self.gru(self.inp(x))[0])

    net = Net(ck["nin"]); net.load_state_dict(ck["state"]); net.eval()
    ds, mu, sd = ck["ds"], ck["mu"], ck["sd"]
    prob = np.zeros((len(X), 3), np.float32)
    with torch.no_grad():
        for a, b in segs:
            x = torch.tensor(((X[a:b + 1:ds] - mu) / sd).astype(np.float32))[None]
            p = torch.softmax(net(x)[0], -1).numpy()
            prob[a:b + 1] = np.repeat(p, ds, 0)[:b + 1 - a]
    return prob


def prob_e(X, segs):
    import torch, torch.nn as nn
    ck = torch.load(os.path.join(OUT, "e", "track_e_model.pt"), map_location="cpu", weights_only=False)

    class Block(nn.Module):
        def __init__(self, h, d):
            super().__init__()
            self.conv = nn.Conv1d(h, h, 3, padding=d, dilation=d)
            self.norm = nn.BatchNorm1d(h); self.drop = nn.Dropout(0.1)

        def forward(self, x):
            return x + self.drop(torch.nn.functional.gelu(self.norm(self.conv(x))))

    class Net(nn.Module):
        def __init__(self, nin, h=96):
            super().__init__()
            self.inp = nn.Conv1d(nin, h, 1)
            self.blocks = nn.Sequential(*[Block(h, d) for d in (1, 2, 4, 8, 16, 32, 64, 128)])
            self.out = nn.Conv1d(h, 3, 1)

        def forward(self, x):
            return self.out(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)

    net = Net(ck["nin"]); net.load_state_dict(ck["state"]); net.eval()
    ds, mu, sd = ck["ds"], ck["mu"], ck["sd"]
    prob = np.zeros((len(X), 3), np.float32)
    with torch.no_grad():
        for a, b in segs:
            x = torch.tensor(((X[a:b + 1:ds] - mu) / sd).astype(np.float32))[None]
            p = torch.softmax(net(x)[0], -1).numpy()
            prob[a:b + 1] = np.repeat(p, ds, 0)[:b + 1 - a]
    return prob


def main(track):
    X, segs, forced = bar_features()
    summ = json.load(open(os.path.join(OUT, track, "track_%s_summary.json" % track)))
    prob = prob_c(X) if track == "c" else prob_d(X, segs) if track == "d" else prob_e(X, segs)
    mode = sys.argv[2] if len(sys.argv) > 2 else summ.get("mode", "model")
    rules_csv = os.path.join(ROOT, "evaluation", "results", "rules_v2_tuned.csv")
    letters = np.array([l.split(",")[1].strip() for l in open(rules_csv).readlines()[1:]])
    rules = np.select([letters == "H", letters == "A", letters == "D"], [0, 1, 2], -1).astype(np.int8)
    lab = decode.decode(prob, segs, summ["penalty"], summ["min_poss"], rules=rules, mode=mode)
    lab[forced] = 2
    import rules_v2 as v2
    print("track", track, "mode", mode, "decoder", summ["best"], "|", v2.summary(lab, segs))
    letters = np.array(["H", "A", "D", "X"])[np.where(lab < 0, 3, lab)]
    out = os.path.join(ROOT, "evaluation", "results", "track_%s%s.csv" % (track, "" if mode == "model" else "_" + mode))
    with open(out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, ch) for i, ch in enumerate(letters))
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1])

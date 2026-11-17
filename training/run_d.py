"""GRU model: bidirectional GRU over the per-frame features (GPU), 3 folds by match,
Viterbi decode. Outputs: track_d_model.pt (state + normalisation), track_d_summary.json."""
import glob, os, sys, time
sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/kprep.py", recursive=True)[0]))
os.environ["MTAH_T"] = "1.0"          # control rule: stints < 1 s that do not hold are not possession
import kprep
kprep.run()
import numpy as np, torch, torch.nn as nn
import common

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, flush=True)
t0 = time.time()
D = common.load_all(kprep.FEATS)
DS = 2                     # 12.5 fps
WIN = 600                  # 48 s windows for training
FOLDS = [["J03WMX", "J03WN1"], ["J03WPY", "J03WOH"], ["J03WQQ", "J03WOY", "J03WR9"]]
EPOCHS = 12


class Net(nn.Module):
    def __init__(self, nin, h=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nin, h), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * h, 3)

    def forward(self, x):
        return self.out(self.gru(self.inp(x))[0])


def windows(keys, mu, sd):
    xs, ys = [], []
    for k in keys:
        X = ((D[k]["X"][::DS] - mu) / sd).astype(np.float32); y = D[k]["gt"][::DS]
        for a in range(0, len(X) - WIN, WIN // 2):
            xs.append(X[a:a + WIN]); ys.append(y[a:a + WIN])
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)).long()


def train(keys):
    allX = np.concatenate([D[k]["X"][::DS * 4] for k in keys])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    X, y = windows(keys, mu, sd)
    cnt = np.bincount(y[y >= 0].numpy(), minlength=3)
    w = torch.tensor((cnt.sum() / (3 * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32, device=dev)
    net = Net(X.shape[2]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * ((len(X) + 31) // 32))
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-1)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), 32):
            b = perm[i:i + 32]
            xb, yb = X[b].to(dev), y[b].to(dev)
            loss = lossf(net(xb).reshape(-1, 3), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(b)
        print("  epoch %d loss %.4f (%.0f s)" % (ep, tot / len(X), time.time() - t0), flush=True)
    return net, mu, sd


@torch.no_grad()
def predict(net, mu, sd, k):
    net.eval()
    X = D[k]["X"]; n = len(X)
    prob = np.zeros((n, 3), np.float32)
    for a, b in D[k]["segs"]:
        x = torch.tensor(((X[a:b + 1:DS] - mu) / sd).astype(np.float32))[None].to(dev)
        p = torch.softmax(net(x)[0], -1).cpu().numpy()
        prob[a:b + 1] = np.repeat(p, DS, 0)[:b + 1 - a]
    return prob


oof = {}
for fold in FOLDS:
    tr = [k for k in D if k[0] not in fold]
    net, mu, sd = train(tr)
    for k in [k for k in D if k[0] in fold]:
        oof[k] = predict(net, mu, sd, k)
    print("fold", fold, "done %.0f s" % (time.time() - t0), flush=True)
common.choose_decoder(D, oof, "/kaggle/working/track_d_summary.json", "D-gru")
net, mu, sd = train(list(D))
torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, nin=len(mu), ds=DS), "/kaggle/working/track_d_model.pt")
print("total %.0f s" % (time.time() - t0))

import shutil
for _d in ("od", "feats", "code"):
    shutil.rmtree("/kaggle/working/" + _d, ignore_errors=True)

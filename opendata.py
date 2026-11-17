"""Open-data training and test harness on the DFL/IDSSE Bundesliga matches (CC-BY 4.0, Bassek et al.).

    python solution/opendata.py convert          # raw XML -> opendata/<id>.npz (slow, once)
    python solution/opendata.py eval             # score rules v1-like / v2 on damaged matches

Raw files: opendata/raw/<id>_meta.xml, <id>_tracking.xml, downloaded from the figshare API
(https://api.figshare.com/v2/file/download/<file id>; ids from kloppy's get_IDSSE_url).
Everything under opendata/ is gitignored.

Labels per frame: ball dead -> 2, else ball-owning team (home 0 / away 1).
Damage makes the clean DFL tracking look like BAR-ATH: ball blackouts with BAR-ATH's run lengths
(28% of them starting where the ball reaches a line), and 0-4 players missing in bursts, with the
BAR-ATH distribution of tracked-player counts. Profile from opendata/bar_profile.npz (unlabelled
statistics of our match).
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "solution"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
OD = os.environ.get("MTAH_OD", os.path.join(ROOT, "opendata"))
MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
FPS = 25


# ---------------------------------------------------------------- convert
def convert(m):
    """Stream the DFL position XML line by line (one <Frame> per line), so memory stays small.

    Same fields kloppy's sportec reader uses: X/Y metres from the centre spot, S speed (km/h, converted to m/s),
    on the ball: BallStatus 1 alive / 0 dead, BallPossession 1 home / 2 guest.
    """
    import re
    from array import array
    out = os.path.join(OD, m + ".npz")
    if os.path.isfile(out):
        return
    raw = os.path.join(OD, "raw")
    meta = open(os.path.join(raw, m + "_meta.xml"), encoding="utf-8").read()
    home_id = re.search(r'HomeTeamId="([^"]+)"', meta).group(1)
    guest_id = re.search(r'GuestTeamId="([^"]+)"', meta).group(1)
    fs_re = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
    fr_re = re.compile(r'<Frame N="(\d+)"[^>]*? X="([^"]+)" Y="([^"]+)"[^>]*? S="([^"]+)"'
                       r'(?:[^>]*? BallPossession="(\d)" BallStatus="(\d)")?')
    L, W = 105.0, 68.0
    sets = {}                                         # (section, person) -> arrays
    teams = {}
    cur = None
    with open(os.path.join(raw, m + "_tracking.xml"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<Frame "):
                if cur is None:
                    continue
                g = fr_re.match(line)
                if g is None:
                    continue
                c = cur
                c[0].append(int(g.group(1))); c[1].append(float(g.group(2)))
                c[2].append(float(g.group(3))); c[3].append(float(g.group(4)))
                if c[4] is not None:
                    c[4].append(int(g.group(5) or 0)); c[5].append(int(g.group(6) or 0))
            elif line.startswith("<FrameSet "):
                sec, tid, pid = fs_re.match(line).groups()
                if tid not in (home_id, guest_id, "BALL") or sec not in ("firstHalf", "secondHalf"):
                    cur = None
                    continue
                ball = tid == "BALL"
                cur = (array("i"), array("f"), array("f"), array("f"),
                       array("b") if ball else None, array("b") if ball else None)
                sets[(sec, pid if not ball else "BALL")] = cur
                if not ball:
                    teams[pid] = 0 if tid == home_id else 1
            elif line.startswith("<PitchSize"):
                L = float(re.search(r'X="([^"]+)"', line).group(1))
                W = float(re.search(r'Y="([^"]+)"', line).group(1))
    persons = sorted(teams)
    slot = {p: i for i, p in enumerate(persons)}
    team = np.array([teams[p] for p in persons], dtype=np.int8)
    # frame index: each half spans its ball frames, halves concatenated
    offs, period, total = {}, [], 0
    for k, sec in enumerate(("firstHalf", "secondHalf")):
        N = np.frombuffer(sets[(sec, "BALL")][0], dtype=np.int32)
        offs[sec] = (int(N.min()), total)
        period += [k + 1] * int(N.max() - N.min() + 1)
        total += int(N.max() - N.min() + 1)
    n, S = total, len(persons)
    P = np.full((n, S, 3), np.nan, np.float32)
    B = np.full((n, 2), np.nan, np.float32)
    lab = np.full(n, -1, np.int8)
    for (sec, pid), c in sets.items():
        n0, base = offs[sec]
        idx = np.frombuffer(c[0], dtype=np.int32) - n0 + base
        ok = (idx >= base) & (idx < n)
        idx = idx[ok]
        x = np.frombuffer(c[1], dtype=np.float32)[ok] / L + 0.5
        y = np.frombuffer(c[2], dtype=np.float32)[ok] / W + 0.5
        if pid == "BALL":
            B[idx, 0], B[idx, 1] = x, y
            poss = np.frombuffer(c[4], dtype=np.int8)[ok]; stat = np.frombuffer(c[5], dtype=np.int8)[ok]
            lab[idx] = np.where(stat == 0, 2, np.where(poss == 1, 0, np.where(poss == 2, 1, -1)))
        else:
            s = slot[pid]
            P[idx, s, 0], P[idx, s, 1] = x, y
            P[idx, s, 2] = np.frombuffer(c[3], dtype=np.float32)[ok] / 3.6   # S is km/h
    del sets
    np.savez_compressed(out, P=P, B=B, team=team, lab=lab, period=np.array(period, np.int8),
                        pitch=np.array([L, W]))
    sp = P[:, :, 2]
    print("%s frames %d (%.0f min), players %d (home %d), pitch %.1fx%.1f, dead %.0f%%, "
          "unlabelled %.1f%%, player speed p50 %.2f p99.9 %.1f, players/frame %.1f" % (
              m, n, n / FPS / 60, S, (team == 0).sum(), L, W, 100 * (lab == 2).mean(),
              100 * (lab == -1).mean(), np.nanmedian(sp), np.nanpercentile(sp, 99.9),
              np.isfinite(P[:, :, 0]).sum(1).mean()))


def load(m):
    d = np.load(os.path.join(OD, m + ".npz"))
    lab = d["lab"].copy()
    for i in range(1, len(lab)):                      # alive with no owner: keep the last owner
        if lab[i] == -1:
            lab[i] = lab[i - 1]
    period = d["period"]
    segs = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        segs.append((int(idx[0]), int(idx[-1])))
    return dict(P=d["P"], B=d["B"], team=d["team"], lab=lab, segs=segs, pitch=tuple(d["pitch"]))


# ---------------------------------------------------------------- damage
DEAD_SNAP = 0.6     # share of blackouts that start at a stoppage (ball goes dead)
VIS_SCALE = 0.7     # visible runs shortened so ~41% of frames lose the ball, as in BAR-ATH
# Targets from BAR-ATH: ball missing 41% of graded frames (unlabelled); in the public window 56%
# of missing frames are dead ball -> missing ~55% of dead time and ~31% of live time.


def damage(M, seed, dead_snap=DEAD_SNAP, vis_scale=VIS_SCALE):
    rng = np.random.default_rng(seed)
    prof = np.load(os.path.join(OD, "bar_profile.npz"))
    P, B = M["P"].copy(), M["B"].copy()
    n = len(B)
    hide = np.zeros(n, bool)
    lab = M["lab"]
    for a, b in M["segs"]:
        dead_starts = a + np.flatnonzero(np.diff((lab[a:b + 1] == 2).astype(np.int8)) == 1)
        f = a + int(rng.choice(prof["vis"]) * vis_scale)
        while f <= b:
            g = int(rng.choice(prof["gap"]))
            if len(dead_starts) and rng.random() < dead_snap:
                nxt = dead_starts[dead_starts >= f]       # start at the next stoppage
                if len(nxt) and nxt[0] - f < 60 * FPS:
                    f = int(nxt[0]) - int(rng.integers(0, 2 * FPS))
            hide[f:min(b + 1, f + g)] = True
            f += g + int(rng.choice(prof["vis"]) * vis_scale)
    B[hide] = np.nan
    # players: bursts (median ~5 s) with k missing, k from BAR-ATH's tracked-count distribution
    S = P.shape[1]
    tracked0 = np.isfinite(P[:, :, 0])
    miss_k = np.clip(22 - prof["cnt"], 0, 8)
    gk = goalkeepers(M)
    f = 0
    while f < n:
        ln = int(rng.exponential(7 * FPS)) + FPS
        k = int(rng.choice(miss_k))
        on = np.flatnonzero(tracked0[min(f, n - 1)])
        if k and len(on):
            w = np.where(np.isin(on, gk), 3.0, 1.0); w /= w.sum()
            drop = rng.choice(on, size=min(k, len(on)), replace=False, p=w)
            P[f:f + ln][:, drop] = np.nan
        f += ln
    return P, B, hide


def goalkeepers(M):
    """Slots whose mean x is most extreme per team per half: the keepers."""
    out = []
    for a, b in M["segs"]:
        mx = np.nanmean(M["P"][a:b + 1, :, 0], 0)
        for t in (0, 1):
            idx = np.flatnonzero((M["team"] == t) & np.isfinite(mx))
            if len(idx):
                dev = np.abs(mx[idx] - 0.5)
                out.append(int(idx[np.argmax(dev)]))
    return np.array(out)


# ---------------------------------------------------------------- evaluate
def to_codes(lab, n):
    import possession_metrics as pm
    letters = np.array(["H", "A", "D", "X"], dtype=object)[np.where(lab < 0, 3, lab)]
    return pm.label_codes(letters)


def spells(lab, segs):
    k = 0; fr = 0
    for a, b in segs:
        s = lab[a:b + 1]
        st = np.r_[0, np.flatnonzero(np.diff(s)) + 1]; en = np.r_[st[1:], len(s)]
        for x, y in zip(st, en):
            if s[x] in (0, 1):
                k += 1; fr += y - x
    return k, fr / max(1, k) / FPS


CONTROL_T = float(os.environ.get("MTAH_T", 3.0))   # s; DFL flips owner on any touch. Stints shorter than this that do not hold
                  # are not "controlled" (CHALLENGE.md) -> previous team keeps the ball. T = 3 s
                  # brings DFL to 237 possessions / 13.4 s mean, vs 14.2 s in our public window.


def controlled(lab, segs, T=CONTROL_T):
    lab = lab.copy()
    for a, b in segs:
        s = lab[a:b + 1]
        for _ in range(3):
            st = np.r_[0, np.flatnonzero(np.diff(s)) + 1]; en = np.r_[st[1:], len(s)]
            ch = False
            for i in range(1, len(st)):
                x, e = st[i], en[i]
                if s[x] in (0, 1) and e - x < T * FPS and s[x - 1] in (0, 1) and s[x - 1] != s[x]:
                    s[x:e] = s[x - 1]; ch = True
            if not ch:
                break
    return lab


def baseline(M, P, B, window=50):
    """starter/baseline.py: team of the nearest player when the ball is seen, majority vote
    over +-window frames, D where nobody was seen in the window."""
    L, W = M["pitch"]
    dist = np.hypot(P[:, :, 0] * L - B[:, None, 0] * L, P[:, :, 1] * W - B[:, None, 1] * W)
    dist[~np.isfinite(dist)] = np.inf
    best = dist.argmin(1)
    codes = np.where(np.isfinite(dist.min(1)), M["team"][best], -1)
    h = np.r_[0, np.cumsum(codes == 0)]; a = np.r_[0, np.cumsum(codes == 1)]
    i = np.arange(len(codes)); lo = np.maximum(0, i - window); hi = np.minimum(len(codes), i + window + 1)
    hh, aa = h[hi] - h[lo], a[hi] - a[lo]
    return np.where(hh + aa == 0, 2, np.where(hh >= aa, 0, 1)).astype(np.int8)


METHODS = {
    "baseline_smooth50": None,
    "v1-like": dict(pass_speed=1e9, blackout_boost=1.0, out_rule=False, wall_rule=False, restart_clear=1e9),
    "v2 (changes 1-5)": dict(out_rule=False, wall_rule=False),
    "v2 + out-of-play": dict(wall_rule=False),
    "v2 + wall": dict(out_rule=False),
    "v2 + out + wall": dict(),
    "v2 all, no pass rule": dict(pass_speed=1e9),
}


def evaluate(seeds=(0,), methods=None, verbose=True):
    import possession_metrics as pm
    import rules_v2 as v2
    methods = methods or METHODS
    rows = []
    for m in MATCHES:
        M = load(m)
        gt = controlled(M["lab"], M["segs"])
        gtc = to_codes(gt, len(gt))
        for seed in seeds:
            P, B, hide = damage(M, seed)
            for name, params in methods.items():
                pred = baseline(M, P, B) if params is None else \
                    v2.label_match(P, B, M["team"], M["pitch"], M["segs"], params=params)
                r = pm.evaluate(gtc, to_codes(pred, len(pred)), bootstrap_n=0)
                k, mean = spells(pred, M["segs"])
                rows.append((m, seed, name, r["macro_f1"], r["spell_f1_mean"], k, mean))
                if verbose:
                    print("%s seed %d  %-22s macroF1 %.3f  spellF1 %.3f  poss %4d  mean %5.1f s" % rows[-1], flush=True)
            del P, B
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "convert":
        for m in (sys.argv[2:] or MATCHES):
            convert(m)
    elif cmd == "labels":
        for m in MATCHES:
            M = load(m)
            k, mean = spells(controlled(M["lab"], M["segs"]), M["segs"])
            print("%s controlled possessions %d, mean %.1f s, dead %.0f%%" % (m, k, mean, 100 * (M["lab"] == 2).mean()))
    elif cmd == "eval":
        seeds = tuple(int(s) for s in sys.argv[2:]) or (0,)
        rows = evaluate(seeds)
        print("\nmean over %d matches x %d seeds:" % (len(MATCHES), len(seeds)))
        for name in METHODS:
            r = np.array([x[3:] for x in rows if x[2] == name], dtype=float)
            print("  %-22s macroF1 %.3f  spellF1 %.3f (match range %.3f-%.3f)  poss %4.0f  mean %5.1f s" % (
                name, r[:, 0].mean(), r[:, 1].mean(), r[:, 1].min(), r[:, 1].max(), r[:, 2].mean(), r[:, 3].mean()))

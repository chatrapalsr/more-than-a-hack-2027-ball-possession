"""First rule-based version: rule-based possession from the tactical ATD. numpy only.

    python solution/rules_pipeline.py --out evaluation/results/rules_v1.csv

Prints aggregates only. No labels, no ground truth, no frame-specific constants except the
published half boundaries from games.py / FAQ.md.
"""
import argparse, os, sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths, source_frames, half_video_blocks  # noqa: E402

CACHE = os.path.join(ROOT, "solution", "_cache")
FPS = 25
# Settings: football-logic defaults. Tune ONLY on other matches (the DFL harness, opendata.py).
R_CONTACT = 1.2        # m, ball this close to a player counts as contact
MARGIN = 0.5           # m, and this much closer than the nearest opponent
CONTROL_FRAMES = 10    # contact frames by the new team ...
CONTROL_WINDOW = 38    # ... inside this many frames (1.5 s) = control
EPISODE_GAP = 5        # frames without contact that separate two contact episodes
SPEED_SMOOTH = 75      # frames (3 s) rolling mean of median player speed
DEAD_MIN = 100         # frames (4 s) of slow players to call dead
MIN_POSS = 25          # frames (1 s) minimum possession run
MIN_DEAD = 75          # frames (3 s) minimum dead run
DEAD_SHARE_PRIOR = 0.40  # football prior: effective playing time is ~55-60% of a match
KICKOFF_2ND = 113742   # published in FAQ.md: graded from 110808, all D until here


def load_match():
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "tactical.npz")
    if os.path.isfile(path):
        d = np.load(path)
        return d["P"], d["B"], d["team"]
    paths = asset_paths()
    n = source_frames("tactical")
    root = ET.parse(paths["atd_meta"]).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team = np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], dtype=np.int8)
    S = len(team)
    P = np.full((n, S, 3), np.nan, dtype=np.float32)
    B = np.full((n, 2), np.nan, dtype=np.float32)
    with open(paths["atd_raw"]) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            for s, g in enumerate(parts[1].split(";")[:S]):
                if g and g[0] != "N":
                    v = g.split(",")
                    P[i, s, :len(v[:3])] = [float(x) for x in v[:3]]
            b = parts[2].split(",")
            if b[0] and b[0][0] != "N":
                B[i] = [float(b[0]), float(b[1])]
    np.savez_compressed(path, P=P, B=B, team=team)
    return P, B, team


def runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    st = np.flatnonzero(d == 1)
    return st, np.flatnonzero(d == -1) - st


def rolling_mean(x, w):
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x)); lo = np.maximum(0, i - w // 2); hi = np.minimum(len(x), i + w // 2 + 1)
    return (c[hi] - c[lo]) / np.maximum(1, k[hi] - k[lo])


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins)
    mid = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * mid) / np.maximum(1, w0)
    m1 = (np.sum(h * mid) - np.cumsum(h * mid)) / np.maximum(1, w1)
    return mid[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def build(verbose=True):
    P, B, team = load_match()
    n = len(B)
    L, W = GAMES[GAME]["pitch"]
    pv = np.isfinite(P[:, :, 0]); bv = np.isfinite(B[:, 0])
    bx, by = B[:, 0] * L, B[:, 1] * W
    dist = np.hypot(P[:, :, 0] * L - bx[:, None], P[:, :, 1] * W - by[:, None])
    dist[~pv] = np.inf
    dh = np.where(team == 0, dist, np.inf).min(1)
    da = np.where(team == 1, dist, np.inf).min(1)
    contact = np.full(n, -1, dtype=np.int8)
    contact[bv & (dh < R_CONTACT) & (da - dh > MARGIN)] = 0
    contact[bv & (da < R_CONTACT) & (dh - da > MARGIN)] = 1

    sp = np.where(pv, P[:, :, 2], np.nan)
    sp.sort(axis=1)                                   # NaN last; median of the finite part
    cnt = pv.sum(1)
    med = np.where(cnt > 0, sp[np.arange(n), np.maximum(0, (cnt - 1) // 2)], np.nan)
    v = rolling_mean(med, SPEED_SMOOTH)

    blocks = half_video_blocks(GAME)
    graded = np.zeros(n, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    # Otsu split the speed histogram at 2.15 m/s and called 74% of the match dead, which no
    # football match is. Use the quantile implied by typical effective playing time instead.
    thr = np.quantile(v[graded & np.isfinite(v)], DEAD_SHARE_PRIOR)
    slow = v < thr
    out_of_pitch = bv & ((B[:, 0] < 0) | (B[:, 0] > 1) | (B[:, 1] < 0) | (B[:, 1] > 1))
    dead = np.zeros(n, bool)
    st, ln = runs(slow | out_of_pitch)
    for s, l in zip(st, ln):
        if l >= DEAD_MIN:
            dead[s:s + l] = True
    dead[blocks[1][1]:KICKOFF_2ND] = True

    lab = np.full(n, -1, dtype=np.int8)               # 0 H, 1 A, 2 D, -1 unknown
    n_switch = 0
    for _, a, b in blocks:
        cur, seg_start = -1, a
        pend_start, pend_frames, episodes, last_other = None, [], 0, -10 ** 9
        for f in range(a, b + 1):
            if dead[f]:
                lab[f] = 2
                cur, seg_start = -1, f + 1
                pend_start, pend_frames, episodes = None, [], 0
                continue
            c = contact[f]
            if c >= 0:
                if cur == -1:
                    cur = c
                    lab[seg_start:f] = cur            # restart belongs to the first team in contact
                elif c == cur:
                    pend_start, pend_frames, episodes = None, [], 0
                else:
                    if pend_start is None:
                        pend_start, episodes = f, 1
                    elif f - last_other > EPISODE_GAP:
                        episodes += 1
                    last_other = f
                    pend_frames.append(f)
                    while pend_frames and f - pend_frames[0] > CONTROL_WINDOW:
                        pend_frames.pop(0)
                    if len(pend_frames) >= CONTROL_FRAMES or episodes >= 2:
                        cur = c
                        lab[pend_start:f] = cur       # back-date to the first touch
                        n_switch += 1
                        pend_start, pend_frames, episodes = None, [], 0
            lab[f] = cur
        seg = lab[a:b + 1]                            # live stretches that never saw a contact
        for i in range(1, len(seg)):
            if seg[i] == -1:
                seg[i] = seg[i - 1]
        for i in range(len(seg) - 2, -1, -1):
            if seg[i] == -1:
                seg[i] = seg[i + 1]

    for _, a, b in blocks:                            # minimum durations
        seg = lab[a:b + 1]
        for _ in range(5):
            ch = np.flatnonzero(np.diff(seg)) + 1
            bounds = np.concatenate(([0], ch, [len(seg)]))
            changed = False
            for s, e in zip(bounds[:-1], bounds[1:]):
                need = MIN_DEAD if seg[s] == 2 else MIN_POSS
                if e - s < need and s > 0:
                    seg[s:e] = seg[s - 1]
                    changed = True
            if not changed:
                break

    letters = np.full(n, "X", dtype="<U1")
    for code, ch_ in ((0, "H"), (1, "A"), (2, "D")):
        letters[graded & (lab == code)] = ch_
    if verbose:
        g = letters[graded]
        poss = 0; frames_poss = 0
        for _, a, b in blocks:
            seg = letters[a:b + 1]
            ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
            bounds = np.concatenate(([0], ch, [len(seg)]))
            for s, e in zip(bounds[:-1], bounds[1:]):
                if seg[s] in "HA":
                    poss += 1; frames_poss += e - s
        print("speed threshold %.2f m/s" % thr)
        print("shares on graded frames: H %.1f%%  A %.1f%%  D %.1f%%  X %.1f%%" % tuple(
            100 * (g == c).mean() for c in "HADX"))
        print("possessions %d, mean length %.1f s, control switches %d" % (
            poss, frames_poss / max(1, poss) / FPS, n_switch))
    return letters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "rules_v1.csv"))
    args = ap.parse_args()
    letters = build()
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, c) for i, c in enumerate(letters))
    print("wrote", os.path.relpath(args.out, ROOT))

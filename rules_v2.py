"""Rule-based possession labels (H / A / D). numpy only.

    python solution/rules_v2.py                      # BAR-ATH -> evaluation/results/rules_v2.csv

`label_match()` takes any match in the common format, so the DFL training matches go through
exactly the same code.

Additions over the first version (rules_pipeline.py):
 1. Ball travelling between players overrides slow players, so slow build-up is not called dead.
 2. Ball missing is a dead-ball clue: the slowness threshold is relaxed inside blackouts.
 3. Control test looks ahead a fixed window (LOOKAHEAD) and also applies at restarts.
 4. Restart owner: the team with a player at the ball while every opponent stands off.
 5. Blackout with a different team on each side and no stoppage: split where the players'
    collective movement turns around, else at the midpoint.
"""
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "solution"))

FPS = 25
DEFAULTS = dict(
    # touches
    r_contact=0.9,         # m (tuned on the DFL matches; was 1.2) ball this close to a player = touch ...
    margin=0.5,            # m, ... and this much closer than the nearest opponent
    episode_gap=5,         # frames without a touch that separate two touches
    # control
    control_frames=10,     # touch frames by the new team inside the look-ahead window = control
    control_episodes=2,    # or this many separate touches (a completed pass)
    lookahead=4.0,         # s, how far ahead we look for the new team's control
    # dead ball
    speed_smooth=3.0,      # s, rolling mean of the middle player's speed
    dead_share=0.40,       # prior: effective playing time ~55-60%, so slowest 40% = slow
    blackout_boost=1.3,    # (tuned; was 1.15) threshold x this while the ball is missing (change 2)
    dead_min=4.0,          # s of slowness needed to call dead
    pass_speed=4.0,        # m/s, a pass: ball reaches this speed between two touches ...
    pass_max=3.0,          # s, ... by the same team, at most this far apart (change 1)
    live_window=1.0,       # s, frames this close to a completed pass are live
    out_rule=True,         # user idea: ball fully over a line = dead at once
    out_margin=0.3,        # m beyond the line (ball radius 0.11 m + tracking noise)
    wall_rule=True,        # user idea: a defensive wall = free kick = dead
    wall_n=3,              # players in the wall
    # restarts (change 4)
    restart_near=1.5,      # m, taker at the ball
    restart_clear=3.0,     # m, nearest opponent at least this far
    restart_look=2.0,      # s before the restart to look for the taker
    # clean-up
    min_poss=2.0,          # s (tuned; was 1.0)
    min_dead=3.0,          # s
)


def runs(mask):
    m = np.concatenate(([0], np.asarray(mask).astype(np.int8), [0]))
    d = np.diff(m)
    st = np.flatnonzero(d == 1)
    return st, np.flatnonzero(d == -1) - st


def rolling_mean(x, w):
    w = max(1, int(w))
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x)); lo = np.maximum(0, i - w // 2); hi = np.minimum(len(x), i + w // 2 + 1)
    return np.where(k[hi] - k[lo] > 0, (c[hi] - c[lo]) / np.maximum(1, k[hi] - k[lo]), np.nan)


def dilate(mask, w):
    """True wherever a True lies within +-w frames."""
    c = np.concatenate(([0], np.cumsum(mask)))
    i = np.arange(len(mask)); lo = np.maximum(0, i - w); hi = np.minimum(len(mask), i + w + 1)
    return (c[hi] - c[lo]) > 0


def features(P, B, team, pitch):
    """Per-frame quantities in metres. P[n,S,3] = x,y (0-1) and speed (m/s); B[n,2] ball (0-1)."""
    L, W = pitch
    n = len(B)
    pv = np.isfinite(P[:, :, 0]); bv = np.isfinite(B[:, 0])
    bx, by = B[:, 0] * L, B[:, 1] * W
    dist = np.hypot(P[:, :, 0] * L - bx[:, None], P[:, :, 1] * W - by[:, None])
    dist[~pv] = np.inf
    dh = np.where(team == 0, dist, np.inf).min(1)
    da = np.where(team == 1, dist, np.inf).min(1)
    sp = np.where(pv, P[:, :, 2], np.nan)
    sp.sort(axis=1)                                   # NaN last; median of the finite part
    cnt = pv.sum(1)
    med = np.where(cnt > 0, sp[np.arange(n), np.maximum(0, (cnt - 1) // 2)], np.nan)
    # ball speed from consecutive visible frames, lightly smoothed (tracking jitter)
    step = np.full(n, np.nan)
    both = bv[1:] & bv[:-1]
    step[1:][both] = np.hypot(np.diff(bx), np.diff(by))[both] * FPS
    bspeed = rolling_mean(step, 5)
    bspeed[~bv] = np.nan
    # mean x velocity of all players (m/s): the whole block turning round marks a turnover
    xs = np.where(pv, P[:, :, 0] * L, np.nan)
    cx = np.array([np.nan] * n)
    okc = cnt > 0
    cx[okc] = np.nansum(xs[okc], 1) / cnt[okc]
    vx = np.r_[np.nan, np.diff(cx) * FPS]
    vx[np.abs(vx) > 5] = np.nan                       # slot swaps make jumps; drop them
    inside = bv & (B[:, 0] >= 0) & (B[:, 0] <= 1) & (B[:, 1] >= 0) & (B[:, 1] <= 1)
    # ball fully over a line: centre beyond it by more than the ball radius + tracking noise
    over = np.maximum.reduce([-bx, bx - L, -by, by - W])
    return dict(n=n, bv=bv, inside=inside, dh=dh, da=da, med=med, bspeed=bspeed, vx=vx,
                over=np.where(bv, over, -np.inf), dist=dist, P=P, team=team, pitch=pitch, bx=bx)


def wall_frames(F, p):
    """Frames with a defensive wall: >= wall_n still players of one team, each within 2 m of
    another, 7-12 m from a visible ball, ball within 35 m of a goal line."""
    L, W = F["pitch"]
    P, dist, team = F["P"], F["dist"], F["team"]
    n = F["n"]
    still = np.isfinite(P[:, :, 2]) & (P[:, :, 2] < 1.0)
    ring = still & (dist > 7) & (dist < 12)
    ballx = np.where(F["bv"], np.nan_to_num(F["bx"]), -1e9)
    near_goal = F["bv"] & ((ballx < 35) | (ballx > L - 35))
    out = np.zeros(n, bool)
    for t in (0, 1):
        cand = ring & (team == t)[None, :]
        rows = np.flatnonzero(near_goal & (cand.sum(1) >= p["wall_n"]))
        for i in rows:
            idx = np.flatnonzero(cand[i])
            xy = np.stack([P[i, idx, 0] * L, P[i, idx, 1] * W], 1)
            d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
            np.fill_diagonal(d, np.inf)
            if ((d < 2.0).any(1)).sum() >= p["wall_n"]:
                out[i] = True
    return out


def dead_mask(F, segs, p, forced_dead=None):
    n = F["n"]
    v = rolling_mean(F["med"], p["speed_smooth"] * FPS)
    graded = np.zeros(n, bool)
    for a, b in segs:
        graded[a:b + 1] = True
    thr = np.nanquantile(v[graded], p["dead_share"])
    thr_f = np.where(F["bv"], thr, thr * p["blackout_boost"])          # change 2
    slow = v < thr_f
    # change 1: a completed pass between teammates (touch, ball travels fast inside the pitch,
    # same team touches again soon, nobody else in between) -> clearly live
    c = contact_codes(F, p)
    eps = episodes(c, 0, n - 1, p["episode_gap"])
    live_ev = np.zeros(n, bool)
    for (t1, s1, e1, _), (t2, s2, e2, _) in zip(eps[:-1], eps[1:]):
        if t1 != t2 or s2 - e1 < 3 or s2 - e1 > p["pass_max"] * FPS:
            continue
        fl = slice(e1, s2 + 1)
        if F["inside"][fl].all() and np.nanmax(F["bspeed"][fl]) > p["pass_speed"]:
            live_ev[s1:e2 + 1] = True
    live_ev = dilate(live_ev, int(p["live_window"] * FPS))
    out = F["bv"] & ~F["inside"]
    cand = (slow & ~live_ev) | out
    dead = np.zeros(n, bool)
    st, ln = runs(cand)
    for s, l in zip(st, ln):
        if l >= p["dead_min"] * FPS:
            dead[s:s + l] = True
    if p["out_rule"]:
        # ball seen fully over a line -> dead at once, and it stays dead while the ball is out,
        # missing or the players are slow, until a completed pass or the stretch ends
        over = F["over"] > p["out_margin"]
        keep = (over | ~F["bv"] | slow | out) & ~live_ev
        st, ln = runs(keep)
        for s, l in zip(st, ln):
            if over[s:s + l].any():
                f0 = s + int(np.argmax(over[s:s + l]))
                dead[f0:s + l] = True
    if p["wall_rule"]:
        wall = wall_frames(F, p)
        st, ln = runs(wall)
        for s, l in zip(st, ln):
            if l >= FPS:
                # the wall stands from the stoppage until the kick: extend over the slow run
                keep = (slow | ~F["bv"])
                a = s
                while a > 0 and keep[a - 1]:
                    a -= 1
                dead[a:s + l] = True
    if forced_dead is not None:
        dead |= forced_dead
    return dead & graded, thr


def contact_codes(F, p):
    c = np.full(F["n"], -1, dtype=np.int8)
    dh, da, bv = F["dh"], F["da"], F["bv"]
    c[bv & (dh < p["r_contact"]) & (da - dh > p["margin"])] = 0
    c[bv & (da < p["r_contact"]) & (dh - da > p["margin"])] = 1
    return c


def episodes(c, a, b, gap):
    """Touch episodes inside [a, b]: list of (team, first, last, n_frames)."""
    eps = []
    for f in range(a, b + 1):
        t = c[f]
        if t < 0:
            continue
        if eps and eps[-1][0] == t and f - eps[-1][2] <= gap:
            e = eps[-1]; eps[-1] = (t, e[1], f, e[3] + 1)
        else:
            eps.append((int(t), f, f, 1))
    return eps


def controls(eps, i, team, horizon, p):
    """Does `team` show control starting at episode i, before the other team touches, within horizon?"""
    frames, count = 0, 0
    start = eps[i][1]
    for t, s, e, k in eps[i:]:
        if s - start > horizon:
            break
        if t != team:
            return False
        frames += k; count += 1
        if frames >= p["control_frames"] or count >= p["control_episodes"]:
            return True
    return False


def restart_owner(F, start, p):
    """Change 4: the team with a player at the ball while every opponent stands off."""
    lo = max(0, start - int(p["restart_look"] * FPS)); hi = start + FPS // 2
    dh, da, bv = F["dh"][lo:hi], F["da"][lo:hi], F["bv"][lo:hi]
    h = (bv & (dh < p["restart_near"]) & (da > p["restart_clear"])).sum()
    a = (bv & (da < p["restart_near"]) & (dh > p["restart_clear"])).sum()
    if max(h, a) >= 5:
        return 0 if h > a else 1
    return -1


def live_stretch(F, c, lab, a, b, p):
    """Label one live stretch [a, b] (no dead frames inside) as 0/1."""
    eps = episodes(c, a, b, p["episode_gap"])
    horizon = int(p["lookahead"] * FPS)
    cur = restart_owner(F, a, p)
    first_i = 0
    if cur < 0:                                       # change 3: restart owner must show control too
        for i, e in enumerate(eps):
            if controls(eps, i, e[0], horizon, p):
                cur, first_i = e[0], i
                break
        if cur < 0:
            cur = eps[0][0] if eps else -1
    lab[a:b + 1] = cur
    switches = []
    i = first_i
    while i < len(eps):
        t, s, e, k = eps[i]
        if t != cur and controls(eps, i, t, horizon, p):
            switches.append((s, t))                   # back-dated to the first touch
            cur = t
        i += 1
    for s, t in switches:
        lab[s:b + 1] = t
    return len(switches)


def split_blackouts(F, lab, dead, segs, p):
    """Change 5: ball missing, no stoppage, different team on each side -> split at turnaround."""
    n_split = 0
    for a, b in segs:
        st, ln = runs(~F["bv"][a:b + 1])
        for s, l in zip(st + a, ln):
            e = s + l
            if l < FPS or s == a or e > b or dead[s:e].any():
                continue
            before, after = lab[s - 1], lab[e] if e <= b else -1
            if before < 0 or after < 0 or before == after:
                continue
            # first touch after the blackout was back-dated to e; the true change is inside.
            vx = rolling_mean(F["vx"][s:e], 2 * FPS)
            if np.isfinite(vx).sum() > FPS:
                d = np.abs(np.diff(np.nan_to_num(vx)))
                cut = s + 1 + int(np.argmax(d))
            else:
                cut = s + l // 2
            lab[s:cut] = before
            lab[cut:e] = after
            n_split += 1
    return n_split


def min_runs(lab, segs, p):
    for a, b in segs:
        seg = lab[a:b + 1]
        for _ in range(5):
            ch = np.flatnonzero(np.diff(seg)) + 1
            bounds = np.concatenate(([0], ch, [len(seg)]))
            changed = False
            for s, e in zip(bounds[:-1], bounds[1:]):
                need = (p["min_dead"] if seg[s] == 2 else p["min_poss"]) * FPS
                if e - s < need and s > 0:
                    seg[s:e] = seg[s - 1]
                    changed = True
            if not changed:
                break


def label_match(P, B, team, pitch, segs, forced_dead=None, params=None, verbose=False):
    """Returns codes[n]: 0 home, 1 away, 2 dead, -1 not graded. segs = [(first, last)] per half."""
    p = dict(DEFAULTS, **(params or {}))
    F = features(P, B, team, pitch)
    dead, thr = dead_mask(F, segs, p, forced_dead)
    c = contact_codes(F, p)
    lab = np.full(F["n"], -1, dtype=np.int8)
    n_sw = 0
    for a, b in segs:
        lab[a:b + 1][dead[a:b + 1]] = 2
        st, ln = runs(~dead[a:b + 1])
        for s, l in zip(st + a, ln):
            n_sw += live_stretch(F, c, lab, s, s + l - 1, p)
    # a live stretch with no touch at all keeps the neighbouring team
    for a, b in segs:
        seg = lab[a:b + 1]
        for i in range(1, len(seg)):
            if seg[i] == -1:
                seg[i] = seg[i - 1]
        for i in range(len(seg) - 2, -1, -1):
            if seg[i] == -1:
                seg[i] = seg[i + 1]
        seg[seg == -1] = 2
    n_split = split_blackouts(F, lab, dead, segs, p)
    min_runs(lab, segs, p)
    if verbose:
        print("speed threshold %.2f m/s, control switches %d, blackout splits %d" % (thr, n_sw, n_split))
    return lab


def summary(lab, segs):
    g = np.concatenate([lab[a:b + 1] for a, b in segs])
    poss = frames = 0
    for a, b in segs:
        seg = lab[a:b + 1]
        st = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]; en = np.r_[st[1:], len(seg)]
        for s, e in zip(st, en):
            if seg[s] in (0, 1):
                poss += 1; frames += e - s
    return "H %.1f%%  A %.1f%%  D %.1f%% | possessions %d, mean %.1f s" % (
        100 * (g == 0).mean(), 100 * (g == 1).mean(), 100 * (g == 2).mean(), poss, frames / max(1, poss) / FPS)


def main():
    from games import GAME, GAMES, half_video_blocks
    import rules_pipeline as v1
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "rules_v2.csv"))
    args = ap.parse_args()
    P, B, team = v1.load_match()
    segs = [(a, b) for _, a, b in half_video_blocks(GAME)]
    forced = np.zeros(len(B), bool)
    forced[segs[1][0]:v1.KICKOFF_2ND] = True          # published in FAQ.md: all D before kick-off
    lab = label_match(P, B, team, GAMES[GAME]["pitch"], segs, forced, verbose=True)
    print(summary(lab, segs))
    letters = np.array(["H", "A", "D", "X"])[np.where(lab < 0, 3, lab)]
    with open(args.out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, ch) for i, ch in enumerate(letters))
    print("wrote", os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()

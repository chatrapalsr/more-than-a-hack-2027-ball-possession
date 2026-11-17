"""Per-frame feature matrix for the learned tracks (C: gradient boosting, D: sequence model).

Same function on the DFL training matches and on BAR-ATH, so train and inference match.
All features are computed without labels. float32, NaN-free (missing -> sentinel + flag).
"""
import numpy as np
import rules_v2 as v2

FPS = 25


def _roll(x, w):
    return v2.rolling_mean(x.astype(np.float64), w).astype(np.float32)


def _time_since(mask):
    """Seconds since mask was last True (forward), capped at 30."""
    n = len(mask); out = np.empty(n, np.float32); last = -10 ** 9
    for i in range(n):
        if mask[i]:
            last = i
        out[i] = min(30.0, (i - last) / FPS)
    return out


def build(P, B, team, pitch, segs, forced_dead=None):
    L, W = pitch
    F = v2.features(P, B, team, pitch)
    p = dict(v2.DEFAULTS)
    n = F["n"]
    dead, thr = v2.dead_mask(F, segs, p, forced_dead)
    lab = v2.label_match(P, B, team, pitch, segs, forced_dead)
    c = v2.contact_codes(F, p)
    bv = F["bv"]
    cols = {}
    dh = np.minimum(F["dh"], 30).astype(np.float32); da = np.minimum(F["da"], 30).astype(np.float32)
    dh[~bv] = 30; da[~bv] = 30
    cols["ball_visible"] = bv.astype(np.float32)
    cols["d_home"] = dh; cols["d_away"] = da
    cols["d_min"] = np.minimum(dh, da); cols["d_diff"] = da - dh
    cols["contact_h"] = (c == 0).astype(np.float32); cols["contact_a"] = (c == 1).astype(np.float32)
    bsp = np.nan_to_num(F["bspeed"], nan=-1).astype(np.float32)
    cols["ball_speed"] = np.minimum(bsp, 40)
    bx = np.where(bv, B[:, 0], -1).astype(np.float32); by = np.where(bv, B[:, 1], -1).astype(np.float32)
    cols["ball_x"] = bx; cols["ball_y"] = by
    cols["ball_over"] = np.clip(np.where(bv, F["over"], -40), -40, 10).astype(np.float32)
    cols["since_seen"] = _time_since(bv); cols["until_seen"] = _time_since(bv[::-1])[::-1].copy()
    med = np.nan_to_num(F["med"], nan=np.nanmedian(F["med"])).astype(np.float32)
    cols["speed_med"] = med
    cols["speed_rel"] = (med / max(thr, 0.1)).astype(np.float32)
    pv = np.isfinite(P[:, :, 0])
    sp = np.where(pv, P[:, :, 2], np.nan)
    cnt = pv.sum(1).astype(np.float32)
    cols["n_players"] = cnt
    cols["frac_slow"] = (np.nansum(sp < 1.0, 1) / np.maximum(cnt, 1)).astype(np.float32)
    for t, nm in ((0, "h"), (1, "a")):
        m = pv & (team == t)[None, :]
        k = np.maximum(m.sum(1), 1)
        cx = np.where(m, P[:, :, 0], 0).sum(1) / k
        cols["cx_" + nm] = cx.astype(np.float32)
        cols["sx_" + nm] = np.sqrt(np.maximum(0, np.where(m, P[:, :, 0] ** 2, 0).sum(1) / k - cx ** 2)).astype(np.float32)
        cols["vx_" + nm] = np.r_[0, np.clip(np.diff(cx) * L * FPS, -5, 5)].astype(np.float32)
    cols["vx_all"] = np.nan_to_num(F["vx"]).astype(np.float32)
    cols["v2_h"] = (lab == 0).astype(np.float32); cols["v2_a"] = (lab == 1).astype(np.float32)
    cols["v2_d"] = (lab == 2).astype(np.float32); cols["v2_deadmask"] = dead.astype(np.float32)
    # rolling context
    for w, tag in ((25, "1s"), (75, "3s"), (250, "10s")):
        for k in ("contact_h", "contact_a", "ball_visible", "speed_med", "frac_slow", "d_diff", "vx_h", "vx_a"):
            cols["%s_r%s" % (k, tag)] = _roll(cols[k], w)
    names = list(cols)
    X = np.stack([np.nan_to_num(cols[k]) for k in names], 1).astype(np.float32)
    return X, names, lab

"""Tracking-damage profile of the provided match (ball visibility runs, players tracked per frame).
Used only to degrade the clean DFL tracking so training inputs look like ours. No labels involved.
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "solution"))
import rules_pipeline as v1, rules_v2 as v2
from games import GAME, half_video_blocks
P, B, team = v1.load_match()
segs = [(a, b) for _, a, b in half_video_blocks(GAME)]
vis, gap, starts_line = [], [], []
for a, b in segs:
    bv = np.isfinite(B[a:b + 1, 0])
    st, ln = v2.runs(bv); vis += list(ln)
    st, ln = v2.runs(~bv); gap += list(ln)
    for s in st:
        if s > 0:
            x, y = B[a + s - 1]
            starts_line.append(min(x, 1 - x) * 105.3 < 1.5 or min(y, 1 - y) * 68 < 1.5)
vis, gap = np.array(vis), np.array(gap)
print("visible runs: n %d, median %.1f s, p25 %.1f, p75 %.1f" % (len(vis), np.median(vis) / 25, *np.percentile(vis, [25, 75]) / 25))
print("missing runs: n %d, median %.1f s, p25 %.1f, p75 %.1f; frac of missing frames in runs >1 s: %.2f" % (
    len(gap), np.median(gap) / 25, *np.percentile(gap, [25, 75]) / 25, gap[gap > 25].sum() / gap.sum()))
print("long blackouts (>1 s): n %d, median %.1f s" % ((gap > 25).sum(), np.median(gap[gap > 25]) / 25))
print("blackouts starting within 1.5 m of a line: %.0f%%" % (100 * np.mean(starts_line)))
g = np.concatenate([np.arange(a, b + 1) for a, b in segs])
cnt = np.isfinite(P[g, :, 0]).sum(1)
print("players tracked: mean %.1f; distribution 22+:%.0f%% 21:%.0f%% 20:%.0f%% 19:%.0f%% <=18:%.0f%%" % (
    cnt.mean(), 100 * (cnt >= 22).mean(), 100 * (cnt == 21).mean(), 100 * (cnt == 20).mean(), 100 * (cnt == 19).mean(), 100 * (cnt <= 18).mean()))
np.savez(ROOT + r"\opendata\bar_profile.npz", vis=vis, gap=gap, line_frac=np.mean(starts_line), cnt=cnt)

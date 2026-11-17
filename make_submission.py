"""Regenerate the submission file from the provided tracking data.

Run from the challenge repository root, with this repository cloned into ./solution:

    python solution/make_submission.py

Steps: rule-based labels -> per-frame features -> GRU and TCN probabilities (averaged) ->
Viterbi decode, dead ball wherever the rules or the models call it -> CSV.
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import rules_pipeline as v1, rules_v2 as v2, decode, predict_learned as pl
from games import GAME, GAMES, half_video_blocks

TEAM = "Python in my Backend"
PENALTY, MIN_POSS = 32.0, 1.0


def main():
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "union"
    P, B, team = v1.load_match()
    segs = [(a, b) for _, a, b in half_video_blocks(GAME)]
    forced = np.zeros(len(B), bool)
    forced[segs[1][0]:v1.KICKOFF_2ND] = True                  # FAQ: everything before the 2nd-half kick-off is D
    print("rules ...", flush=True)
    rules = v2.label_match(P, B, team, GAMES[GAME]["pitch"], segs, forced).astype(np.int8)
    del P, B
    print("features + models ...", flush=True)
    X, segs, forced = pl.bar_features()
    prob = (pl.prob_d(X, segs) + pl.prob_e(X, segs)) / 2
    lab = decode.decode(prob, segs, PENALTY, MIN_POSS, rules=rules, mode=mode)
    lab[forced] = 2
    print(v2.summary(lab, segs))
    letters = np.array(["H", "A", "D", "X"])[np.where(lab < 0, 3, lab)]
    os.makedirs(os.path.join(HERE, "submission"), exist_ok=True)
    out = os.path.join(HERE, "submission", TEAM + ("" if mode == "union" else " (%s)" % mode) + ".csv")
    with open(out, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, c) for i, c in enumerate(letters))
    print("wrote", out)


if __name__ == "__main__":
    main()

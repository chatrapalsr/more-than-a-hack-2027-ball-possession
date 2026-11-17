"""Turn per-frame class probabilities (H, A, D) into labels: Viterbi with a switch penalty, then
minimum run lengths. Shared by Tracks C and D."""
import numpy as np


def viterbi(logp, segs, penalty):
    """logp[n,3]. Each change of state costs `penalty` (in log units). Decoded per segment."""
    n = len(logp)
    out = np.full(n, -1, np.int8)
    trans = np.full((3, 3), -penalty); np.fill_diagonal(trans, 0.0)
    for a, b in segs:
        lp = logp[a:b + 1]
        m = len(lp)
        back = np.zeros((m, 3), np.int8)
        score = lp[0].copy()
        for i in range(1, m):
            cand = score[:, None] + trans            # from x to y
            back[i] = cand.argmax(0)
            score = cand.max(0) + lp[i]
        s = int(score.argmax())
        path = np.empty(m, np.int8)
        for i in range(m - 1, -1, -1):
            path[i] = s; s = back[i, s]
        out[a:b + 1] = path
    return out


def decode(prob, segs, penalty=8.0, min_poss=1.0, min_dead=3.0, rules=None, mode="model"):
    """mode 'model': the model alone. 'union': dead where the model OR the rules say dead.
    'rules_dead': dead exactly where the rules say dead; the model decides H vs A elsewhere."""
    import rules_v2 as v2
    prob = np.asarray(prob, np.float32)
    if mode != "model" and rules is not None:
        prob = prob.copy()
        rd = rules == 2
        if mode == "rules_dead":
            prob[~rd, 2] = 1e-6                      # model may not call dead outside rules' dead
        prob[rd] = (1e-6, 1e-6, 1.0)
    lab = viterbi(np.log(np.clip(prob, 1e-6, 1)), segs, penalty)
    v2.min_runs(lab, segs, dict(min_poss=min_poss, min_dead=min_dead))
    return lab

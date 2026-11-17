"""Round 2: combine the round-1 winners and push one step further (opendata/results/tune_final.txt)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune
best = dict(r_contact=0.9, min_poss=2.0, blackout_boost=1.3, control_frames=15)
cfgs = [("default", {}), ("combined", best)]
for k, v in (("r_contact", 0.7), ("r_contact", 1.0), ("min_poss", 3.0), ("min_poss", 1.5),
             ("blackout_boost", 1.5), ("control_frames", 20), ("control_frames", 10), ("dead_share", 0.45)):
    cfgs.append(("combined, %s=%s" % (k, v), dict(best, **{k: v})))
tune.main(cfgs)

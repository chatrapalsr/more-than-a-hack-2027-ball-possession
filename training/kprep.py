"""Shared preparation for the GPU notebook: find the code, download the 7 DFL matches from figshare,
convert them, damage them (3 seeds) and build feature matrices, using all CPU cores.

Writes /kaggle/working/feats/<match>_<seed>.npz with X, gt (controlled DFL labels), v2 (rules
labels), segs. No BAR-ATH match data is used here, only bar_profile.npz (run-length statistics).
"""
import glob, os, shutil, sys, time, urllib.request
from multiprocessing import Pool

WORK = "/kaggle/working"
OD = os.path.join(WORK, "od")
FEATS = os.path.join(WORK, "feats")
SEEDS = (0, 1, 2)
FILES = {  # figshare file ids (kloppy get_IDSSE_url), meta / tracking
    "J03WMX": (51643475, 51643514), "J03WN1": (51643472, 51643517), "J03WPY": (51643487, 51643526),
    "J03WOH": (51643478, 51643520), "J03WQQ": (51643484, 51643529), "J03WOY": (51643481, 51643523),
    "J03WR9": (51643490, 51643532),
}


def setup():
    src = os.path.dirname(glob.glob("/kaggle/input/**/rules_v2.py", recursive=True)[0])
    code = os.path.join(WORK, "code")
    if not os.path.isdir(code):
        shutil.copytree(src, code)
    os.makedirs(os.path.join(OD, "raw"), exist_ok=True); os.makedirs(FEATS, exist_ok=True)
    shutil.copy(os.path.join(code, "bar_profile.npz"), OD)
    os.environ["MTAH_OD"] = OD
    sys.path.insert(0, code)
    return code


def _download(args):
    m, kind, fid = args
    out = os.path.join(OD, "raw", "%s_%s.xml" % (m, kind))
    if not os.path.isfile(out):
        for attempt in range(5):
            try:
                urllib.request.urlretrieve("https://api.figshare.com/v2/file/download/%d" % fid, out + ".part")
                os.replace(out + ".part", out)
                break
            except Exception as e:
                print("retry", m, kind, e, flush=True); time.sleep(5)
    return out


def _convert(m):
    import opendata as od
    od.convert(m)
    os.remove(os.path.join(OD, "raw", m + "_tracking.xml"))      # free disk
    return m


def _feats(args):
    m, seed = args
    import numpy as np, opendata as od, features
    out = os.path.join(FEATS, "%s_%d.npz" % (m, seed))
    if os.path.isfile(out):
        return out
    M = od.load(m)
    P, B, _ = od.damage(M, seed)
    X, names, lab = features.build(P, B, M["team"], M["pitch"], M["segs"])
    gt = od.controlled(M["lab"], M["segs"])
    np.savez_compressed(out, X=X, gt=gt, v2=lab, segs=np.array(M["segs"]), names=np.array(names))
    return out


def run():
    t = time.time()
    setup()
    jobs = [(m, k, fid) for m, (a, b) in FILES.items() for k, fid in (("meta", a), ("tracking", b))]
    with Pool(8) as pool:
        pool.map(_download, jobs)
    print("downloaded in %.0f s" % (time.time() - t), flush=True)
    with Pool(os.cpu_count()) as pool:
        pool.map(_convert, list(FILES))
    print("converted in %.0f s" % (time.time() - t), flush=True)
    with Pool(os.cpu_count()) as pool:
        pool.map(_feats, [(m, s) for m in FILES for s in SEEDS])
    print("features in %.0f s" % (time.time() - t), flush=True)

"""Harris parameter study (Stage 2): blockSize, ksize and k, one at a time.

Each parameter is varied while the other two stay at their chosen values
(BLOCK_FRAC 0.005, KSIZE 3, K 0.04). For every setting, the full corner pipeline
runs on all photos and the corner error is measured against the annotations.
(THRES_FRAC was studied earlier, see the notes.)

Run from the project root:  python experiments/param_study.py
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stage2_corners
from stage1_preprocess import preprocess
from stage2_corners import harris_response, find_candidates, select_corners
from evaluate import corner_error_pct

STUDY = {
    "BLOCK_FRAC": [0.002, 0.003, 0.005, 0.008, 0.012],   # 2, 3, 5, 8, 12 px at 1000 px
    "KSIZE": [3, 5, 7],
    "K": [0.04, 0.05, 0.06],
}


def load():
    """Preprocessed photos with their annotated corners (preprocessing is done once)."""
    photos = []
    for path in sorted(glob.glob("data/*.jpg")):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(f"data/{name}.json") as f:
            true = json.load(f)["corners"]
        gray, scale = preprocess(cv2.imread(path))
        photos.append((name, gray, scale, true))
    return photos


def run(photos):
    """Corner error (%) per photo, number of candidates per photo, mean time per photo."""
    errors, counts, t0 = [], [], time.time()
    for name, gray, scale, true in photos:
        R = harris_response(gray)
        cands = find_candidates(R)
        corners = select_corners(cands, R, gray)
        counts.append(len(cands))
        errors.append(None if corners is None else corner_error_pct(corners / scale, true))
    return errors, counts, (time.time() - t0) / len(photos)


def main():
    photos = load()
    names = [n.split("_")[-1] for n, *_ in photos]
    chosen = {p: getattr(stage2_corners, p) for p in STUDY}
    print(f"{len(photos)} photos; chosen values: {chosen}\n")

    for param, values in STUDY.items():
        print(f"--- {param} ---")
        print(f"{'value':>7} | {'mean err %':>10} | {'< 2 %':>5} | {'not found':>9} | "
              f"{'cands min-max':>13} | {'s/photo':>7} | worst photos")
        for v in values:
            setattr(stage2_corners, param, v)
            errors, counts, dt = run(photos)
            found = [e for e in errors if e is not None]
            worst = sorted(((e, n) for e, n in zip(errors, names) if e is not None), reverse=True)[:3]
            mark = "  <- chosen" if v == chosen[param] else ""
            print(f"{v:>7} | {np.mean(found) if found else float('nan'):>10.2f} | "
                  f"{sum(e < 2 for e in found):>2}/{len(errors):<2} | {errors.count(None):>9} | "
                  f"{min(counts):>5}-{max(counts):<7} | {dt:>7.3f} | "
                  + ", ".join(f"{n} {e:.1f}" for e, n in worst) + mark)
        setattr(stage2_corners, param, chosen[param])     # back to the chosen value
        print()


if __name__ == "__main__":
    main()
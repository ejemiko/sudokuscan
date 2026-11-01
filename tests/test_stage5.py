"""Stage 5 test: digit recognition on all photos (H4 rectification, detected corners).

Reports
  1. cell accuracy per photo and overall, empty-cell F1, time per photo
  2. effect of cell preparation: filled cells read from the raw grey cell vs the prepared cell
  3. Tesseract page-segmentation modes (--psm 10, 8, 13) on the prepared cells
  4. the most frequent confusions, with example crops
Debug images: out/tests/stage5/

Run from the project root:  python tests/test_stage5.py
"""
import glob
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage1_preprocess import preprocess
from stage2_corners import harris_response, find_candidates, select_corners
from stage3_rectify import rectify
from stage5_recognize import (recognize, binarise, split_cells, clean_cell,
                              ink_fraction, prepare, read_digit)

OUT_DIR = "out/tests/stage5"
WHITELIST = "-c tessedit_char_whitelist=123456789"
PSMS = [10, 8, 13]


def load_photos():
    """(name, truth grid, H4 image) for every photo whose corners are found."""
    photos = []
    for path in sorted(glob.glob("data/*.jpg")):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(f"data/{name}.json") as f:
            truth = json.load(f)["grid"]
        img = cv2.imread(path)
        gray, scale = preprocess(img)
        R = harris_response(gray)
        corners = select_corners(find_candidates(R), R, gray)
        if corners is None:
            print(f"{name}: corners not found, skipped")
            continue
        photos.append((name, truth, rectify(img, corners / scale)["images"]["H4"]))
    return photos


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    photos = load_photos()

    # 1) full Stage 5 on every photo
    print(f"\n{'photo':<16} {'cells ok':>8} {'acc %':>6} {'s':>5}")
    all_true, all_pred = [], []
    for name, truth, h4 in photos:
        t0 = time.time()
        grid, debug = recognize(h4)
        dt = time.time() - t0
        cv2.imwrite(f"{OUT_DIR}/{name}_debug.png", debug)
        ok = sum(grid[r][c] == truth[r][c] for r in range(9) for c in range(9))
        print(f"{name:<16} {ok:>5}/81 {ok / 81 * 100:>6.1f} {dt:>5.1f}")
        all_true += [d for row in truth for d in row]
        all_pred += [d for row in grid for d in row]
    t = np.array(all_true)
    p = np.array(all_pred)
    acc = (t == p).mean()
    tp = ((t == 0) & (p == 0)).sum()            # empty-cell F1: "empty" is the positive class
    prec = tp / max(1, (p == 0).sum())
    rec = tp / max(1, (t == 0).sum())
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"\noverall cell accuracy: {acc * 100:.1f} %   ({(t == p).sum()}/{len(t)})")
    print(f"empty-cell F1: {f1:.3f}   (precision {prec:.3f}, recall {rec:.3f})")
    print(f"filled-cell accuracy: {(t == p)[t > 0].mean() * 100:.1f} %")

    # 2) + 3) filled cells only: raw vs prepared, and psm modes
    print("\nreading filled cells again for the preparation and psm comparison (~1-2 min)...")
    raw_ok = 0
    psm_ok = {m: 0 for m in PSMS}
    n_filled = 0
    examples = {}                                # (true, pred) -> prepared crop
    for name, truth, h4 in photos:
        gray_cells = split_cells(cv2.cvtColor(h4, cv2.COLOR_BGR2GRAY))
        bw_cells = split_cells(binarise(h4))
        for k in range(81):
            true = truth[k // 9][k % 9]
            if true == 0:
                continue
            n_filled += 1
            raw_ok += int(read_digit(gray_cells[k], f"--psm 10 {WHITELIST}") == true)
            clean = clean_cell(bw_cells[k])
            if ink_fraction(clean) == 0:
                continue                         # nothing left to read
            prepared = prepare(clean)
            for m in PSMS:
                pred = read_digit(prepared, f"--psm {m} {WHITELIST}")
                psm_ok[m] += int(pred == true)
                if m == 8 and pred != true and (true, pred) not in examples:
                    examples[(true, pred)] = prepared
    print(f"\nfilled cells ({n_filled}), --psm 10:")
    print(f"  without preparation (raw grey cell): {raw_ok / n_filled * 100:.1f} %")
    print(f"  with preparation:                    {psm_ok[10] / n_filled * 100:.1f} %")
    print("\npsm modes on prepared filled cells:")
    for m in PSMS:
        print(f"  --psm {m:<2}: {psm_ok[m] / n_filled * 100:.1f} %")

    # 4) confusions (true digit -> predicted, 0 = nothing read / judged empty)
    conf = Counter((a, b) for a, b in zip(all_true, all_pred) if a != b)
    print("\nmost frequent confusions (true -> predicted, count):")
    for (a, b), n in conf.most_common(8):
        print(f"  {a} -> {b}: {n}")
        if (a, b) in examples:
            cv2.imwrite(f"{OUT_DIR}/confusion_{a}_as_{b}.png", examples[(a, b)])
    print(f"\nDebug images and confusion crops: {OUT_DIR}/")


if __name__ == "__main__":
    main()
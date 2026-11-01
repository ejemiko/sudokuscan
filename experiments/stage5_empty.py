"""Stage 5 empty-cell experiment: choose EMPTY_FRAC.

For every cell of every photo (H4 rectification, 4 px margin, remnants cleaned)
compute the share of the cell covered by the largest ink component, and compare
empty cells (0 in the annotation) with filled cells (1-9).

Run from the project root:  python experiments/stage5_empty.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage1_preprocess import preprocess
from stage2_corners import harris_response, find_candidates, select_corners
from stage3_rectify import rectify
from stage5_recognize import binarise, split_cells, clean_cell, ink_fraction

THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]


def main():
    empty, filled = [], []          # (fraction, photo, row, col)
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
        h4 = rectify(img, corners / scale)["images"]["H4"]
        for k, cell in enumerate(split_cells(binarise(h4))):
            r, c = k // 9, k % 9
            frac = ink_fraction(clean_cell(cell))
            item = (frac, name.split("_")[-1], r, c)
            (empty if truth[r][c] == 0 else filled).append(item)

    e = np.array([x[0] for x in empty])
    f = np.array([x[0] for x in filled])
    print(f"\ncells: {len(e)} empty, {len(f)} filled\n")
    print(f"empty  cells: max {e.max() * 100:.2f} %   99th pct {np.percentile(e, 99) * 100:.2f} %   "
          f"exactly 0: {(e == 0).mean() * 100:.1f} %")
    print(f"filled cells: min {f.min() * 100:.2f} %   1st pct {np.percentile(f, 1) * 100:.2f} %\n")

    print(f"{'EMPTY_FRAC':>10} | {'empty seen as filled':>20} | {'filled seen as empty':>20}")
    for t in THRESHOLDS:
        print(f"{t * 100:>9.1f}% | {int((e >= t).sum()):>20} | {int((f < t).sum()):>20}")

    print("\nhighest empty cells (photo, row, col):",
          ", ".join(f"{p} r{r}c{c} {v * 100:.2f}%" for v, p, r, c in sorted(empty, reverse=True)[:5]))
    print("lowest filled cells (photo, row, col):",
          ", ".join(f"{p} r{r}c{c} {v * 100:.2f}%" for v, p, r, c in sorted(filled)[:5]))


if __name__ == "__main__":
    main()
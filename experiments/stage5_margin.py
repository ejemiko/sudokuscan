"""Stage 5 margin experiment: how much of each cell border must be cropped
so that grid lines do not reach the OCR engine?

For every photo: run Stages 1-3, take the H4 rectification, binarise it,
split it into 81 cells of 50x50 px and measure the ink fraction inside the
cropped cell for several margins. Empty cells (0 in the annotation) should
have ~0 ink once the grid lines are cut away.

Run from the project root:  python experiments/stage5_margin.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

# make the stage modules in the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage1_preprocess import preprocess
from stage2_corners import harris_response, find_candidates, select_corners
from stage3_rectify import rectify

CELL = 50                          # 450 / 9
MARGINS = [0, 2, 4, 6, 8, 10, 12]  # pixels cut from every side of a cell
SHOW_MARGINS = [4, 6, 8]           # drawn side by side in the figures
SHOW_PHOTOS = ["01", "06", "15"]   # frontal, most oblique, shadow
OUT_DIR = "out/experiments/stage5_margin"


def binarise(rect_img):
    """Ink = 255, paper = 0. Adaptive threshold copes with the shadow in photo 15.
    (Only for this measurement; the real Stage 5 binarisation is chosen later.)"""
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY) if rect_img.ndim == 3 else rect_img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def cell_crop(bw, r, c, m):
    """Cell (r, c) with m pixels removed from each side."""
    return bw[r * CELL + m:(r + 1) * CELL - m, c * CELL + m:(c + 1) * CELL - m]


def border_touch(crop):
    """True if any ink touches the edge of the crop (sign of a grid-line remnant)."""
    return bool(crop[0].any() or crop[-1].any() or crop[:, 0].any() or crop[:, -1].any())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    empty_ink = {m: [] for m in MARGINS}
    empty_touch = {m: [] for m in MARGINS}
    filled_ink = {m: [] for m in MARGINS}

    for path in sorted(glob.glob("data/*.jpg")):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(f"data/{name}.json") as f:
            truth = np.array(json.load(f)["grid"])

        # Stages 1-3, same calls as sudoku_scan.py
        img = cv2.imread(path)
        gray, scale = preprocess(img)
        R = harris_response(gray)
        corners = select_corners(find_candidates(R), R, gray)
        if corners is None:
            print(f"{name}: corners not found, skipped")
            continue
        rect = rectify(img, corners / scale)
        h4 = rect["images"]["H4"]
        bw = binarise(h4)

        for r in range(9):
            for c in range(9):
                for m in MARGINS:
                    crop = cell_crop(bw, r, c, m)
                    ink = (crop > 0).mean()
                    if truth[r, c] == 0:
                        empty_ink[m].append(ink)
                        empty_touch[m].append(border_touch(crop))
                    else:
                        filled_ink[m].append(ink)

        # figure: green boxes = what would be sent to OCR, for 3 margins
        if name.split("_")[-1] in SHOW_PHOTOS:
            panels = []
            for m in SHOW_MARGINS:
                vis = h4.copy() if h4.ndim == 3 else cv2.cvtColor(h4, cv2.COLOR_GRAY2BGR)
                for r in range(9):
                    for c in range(9):
                        cv2.rectangle(vis, (c * CELL + m, r * CELL + m),
                                      ((c + 1) * CELL - m - 1, (r + 1) * CELL - m - 1),
                                      (0, 200, 0), 1)
                cv2.putText(vis, f"margin {m}px", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                panels.append(vis)
            cv2.imwrite(f"{OUT_DIR}/{name}_margins.png", np.hstack(panels))

    n_empty = len(empty_ink[MARGINS[0]])
    n_filled = len(filled_ink[MARGINS[0]])
    print(f"\ncells: {n_empty} empty, {n_filled} filled (from the annotations)\n")
    print(f"{'margin':>6} | {'empty: ink mean':>15} {'ink 95th pct':>13} {'touch edge':>11} | {'filled: ink mean':>16}")
    for m in MARGINS:
        print(f"{m:>4}px | {np.mean(empty_ink[m]) * 100:>14.2f}% "
              f"{np.percentile(empty_ink[m], 95) * 100:>12.2f}% "
              f"{np.mean(empty_touch[m]) * 100:>10.1f}% | "
              f"{np.mean(filled_ink[m]) * 100:>15.2f}%")
    print(f"\nFigures: {OUT_DIR}/  (green boxes = what would reach the OCR)")


if __name__ == "__main__":
    main()
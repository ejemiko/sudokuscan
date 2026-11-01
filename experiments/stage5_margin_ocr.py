"""Stage 5 margin experiment, part 2: does the margin change OCR accuracy?

For every filled cell (1-9 in the annotation), crop the cell with several
margins and read it with Tesseract (single character, digits 1-9 only).
Preparation is deliberately minimal here (binarise, white padding, upscale);
no border-remnant cleaning yet. Empty cells are not part of this test.

Run from the project root:  python experiments/stage5_margin_ocr.py
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import pytesseract

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage1_preprocess import preprocess
from stage2_corners import harris_response, find_candidates, select_corners
from stage3_rectify import rectify
from stage5_margin import binarise, cell_crop

MARGINS = [2, 4, 6, 8, 10]
CONFIG = "--psm 10 -c tessedit_char_whitelist=123456789"

# puzzle source of every photo (see claude notes: A = KrazyDad, B = Mavo Life, C = 7sudoku)
SOURCE = {}
for n in ["03", "06", "09", "12", "15"]:
    SOURCE[n] = "KrazyDad"
for n in ["01", "04", "07", "10", "13"]:
    SOURCE[n] = "Mavo Life"
for n in ["02", "05", "08", "11", "14"]:
    SOURCE[n] = "7sudoku"
SOURCES = ["KrazyDad", "Mavo Life", "7sudoku"]


def ocr_cell(crop):
    """crop: binary cell, ink = 255. Returns the digit read by Tesseract, 0 if nothing."""
    digit = 255 - crop                                   # black digit on white paper
    digit = cv2.copyMakeBorder(digit, 10, 10, 10, 10,
                               cv2.BORDER_CONSTANT, value=255)
    digit = cv2.resize(digit, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    _, digit = cv2.threshold(digit, 127, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(digit, config=CONFIG).strip()
    return int(text[0]) if text[:1].isdigit() else 0


def main():
    start = time.time()

    # 1) rectify every photo once and keep the binary H4 image
    photos = []
    for path in sorted(glob.glob("data/*.jpg")):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(f"data/{name}.json") as f:
            truth = np.array(json.load(f)["grid"])
        img = cv2.imread(path)
        gray, scale = preprocess(img)
        R = harris_response(gray)
        corners = select_corners(find_candidates(R), R, gray)
        if corners is None:
            print(f"{name}: corners not found, skipped")
            continue
        rect = rectify(img, corners / scale)
        photos.append((name, truth, binarise(rect["images"]["H4"])))

    # 2) OCR every filled cell for every margin
    correct = {m: {s: 0 for s in SOURCES} for m in MARGINS}
    total = {m: {s: 0 for s in SOURCES} for m in MARGINS}
    no_output = {m: 0 for m in MARGINS}
    for m in MARGINS:
        for name, truth, bw in photos:
            src = SOURCE[name.split("_")[-1]]
            for r in range(9):
                for c in range(9):
                    if truth[r, c] == 0:
                        continue
                    pred = ocr_cell(cell_crop(bw, r, c, m))
                    total[m][src] += 1
                    correct[m][src] += int(pred == truth[r, c])
                    no_output[m] += int(pred == 0)
        print(f"margin {m}px done ({time.time() - start:.0f} s)")

    # 3) table
    print(f"\nOCR accuracy on filled cells (minimal preparation, no remnant cleaning)\n")
    print(f"{'margin':>6} | {'all':>7} | " + " | ".join(f"{s:>9}" for s in SOURCES) + f" | {'no output':>9}")
    for m in MARGINS:
        all_c = sum(correct[m].values())
        all_t = sum(total[m].values())
        per_src = " | ".join(f"{correct[m][s] / total[m][s] * 100:>8.1f}%" for s in SOURCES)
        print(f"{m:>4}px | {all_c / all_t * 100:>6.1f}% | {per_src} | {no_output[m]:>9}")
    print(f"\ncells per source: " + ", ".join(f"{s} {total[MARGINS[0]][s]}" for s in SOURCES))
    print(f"total time: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
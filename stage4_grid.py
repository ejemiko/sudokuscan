import glob
import json
from pathlib import Path

import cv2
import numpy as np

from stage5_recognize import binarise


def grid_deviation(bw):
    row_profile = bw.sum(axis=1)
    col_profile = bw.sum(axis=0)

    deviations = []

    for profile in [row_profile, col_profile]:
        for k in range(9):
            ideal = k * 50
            lo = max(0, ideal - 20)
            hi = min(len(profile), ideal + 21)

            position = lo + np.argmax(profile[lo:hi])
            deviations.append(abs(position - ideal))

    return np.mean(deviations), np.max(deviations)


print(
    f"{'Fotoğraf':<25}"
    f"{'A3 ort/max':>16}"
    f"{'A4 ort/max':>16}"
    f"{'H4 ort/max':>16}"
)

for path in sorted(glob.glob("results/*_result.json")):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "corners" not in data:
        continue

    root = Path(path).stem.removesuffix("_result")
    values = []

    for model in ["A3", "A4", "H4"]:
        img = cv2.imread(f"results/{root}_rect_{model}.png")

        if img is None:
            values.append("eksik")
            continue

        bw = binarise(img)
        mean_dev, max_dev = grid_deviation(bw)
        values.append(f"{mean_dev:.2f}/{max_dev:.0f}")

    print(
        f"{data['image']:<25}"
        f"{values[0]:>16}"
        f"{values[1]:>16}"
        f"{values[2]:>16}"
    )
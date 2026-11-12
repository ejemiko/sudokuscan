import glob
import json
from pathlib import Path

import cv2
import numpy as np

from stage5_recognize import recognize


models = ["A3", "A4", "H4"]
scores = {model: [] for model in models}

print(f"{'Fotoğraf':<25} {'A3 (%)':>10} {'A4 (%)':>10} {'H4 (%)':>10}")

for path in sorted(glob.glob("results/*_result.json")):
    root = Path(path).stem.removesuffix("_result")

    with open(f"data/{root}.json", "r", encoding="utf-8") as f:
        truth = np.array(json.load(f)["grid"])

    values = []

    for model in models:
        image_path = f"results/{root}_rect_{model}.png"
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(image_path)

        grid, _ = recognize(img)
        accuracy = np.count_nonzero(np.array(grid) == truth) / 81 * 100

        scores[model].append(accuracy)
        values.append(accuracy)

    print(
        f"{root:<25}"
        f" {values[0]:>10.2f}"
        f" {values[1]:>10.2f}"
        f" {values[2]:>10.2f}"
    )

print(
    f"{'Ortalama (%)':<25}"
    f" {np.mean(scores['A3']):>10.2f}"
    f" {np.mean(scores['A4']):>10.2f}"
    f" {np.mean(scores['H4']):>10.2f}"
)
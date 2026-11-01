import glob
import json
import os
import matplotlib.pyplot as plt

import numpy as np

rows = []

for path in sorted(glob.glob("results/*_result.json")):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "corners" not in data:
        continue

    corners = np.array(data["corners"], dtype=float)
    tl, tr, br, bl = corners

    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    skew = max(
        max(top, bottom) / min(top, bottom),
        max(left, right) / min(left, right),
    )

    rows.append((
        data["image"],
        skew,
        data["a3_corner_err_px"],
        data["a4_rms_px"],
    ))

rows.sort(key=lambda row: row[1])

print(f"{'Fotoğraf':<25} {'Eğiklik':>10} {'A3 hata (px)':>14} {'A4 RMS (px)':>14}")
for name, skew, a3_error, a4_rms in rows:
    print(f"{name:<25} {skew:>10.3f} {a3_error:>14.3f} {a4_rms:>14.3f}")
    
skews = [row[1] for row in rows]
a3_errors = [row[2] for row in rows]
a4_rms_values = [row[3] for row in rows]

plt.figure(figsize=(9, 6))

plt.scatter(skews, a3_errors, label="A3 corner error")
plt.scatter(skews, a4_rms_values, label="A4 RMS")

for name, skew, a3_error, _ in rows:
    if skew > 1.05:
        plt.annotate(
            name[-6:-4],
            (skew, a3_error),
            xytext=(5, 5),
            textcoords="offset points",
        )

plt.xlabel("Obliqueness (opposite-side length ratio)")
plt.ylabel("Residual (px)")
plt.title("Affine residuals vs. obliqueness")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

os.makedirs("out/stage4", exist_ok=True)
plt.savefig("out/stage4/residual_vs_obliqueness.png", dpi=150)
plt.close()
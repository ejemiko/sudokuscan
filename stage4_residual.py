import glob
import json

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
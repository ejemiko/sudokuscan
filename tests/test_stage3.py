"""Stage 3 test: A3, A4 and H4 rectification.

Run from the project root:   python3 tests/test_stage3.py
Uses the ground-truth corners from data/*.json, so Stage 3 is tested on its own,
independent of Stage 2. Writes the three rectifications side by side to out/tests/stage3/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage3_rectify import rectify, apply_affine, apply_homography, TARGET, SIZE

DATA = Path("data")
OUT = Path("out/tests/stage3")
OUT.mkdir(parents=True, exist_ok=True)

passed, failed = 0, 0


def check(condition, message):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {message}")


def least_squares_affine(src, dst):
    """Affine by plain least squares (Lecture 6, slides 176-178): solve A t = b for the 6 unknowns."""
    rows, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0]); b.append(u)
        rows.append([0, 0, 0, x, y, 1]); b.append(v)
    t, *_ = np.linalg.lstsq(np.array(rows, float), np.array(b, float), rcond=None)
    return t.reshape(2, 3)


print(f"{'image':<16}{'cond':<10}{'A3 BR err px':>14}{'A4 RMS px':>12}{'H4 max px':>12}")
for path in sorted(DATA.glob("*.jpg")):
    gt = json.loads(path.with_suffix(".json").read_text())
    if not gt.get("corners"):
        print(f"{path.stem:<16} no annotated corners, skipped")
        continue
    img = cv2.imread(str(path))
    src = np.array(gt["corners"], dtype=np.float64)
    r = rectify(img, src)

    # 1. Output images are 450x450
    for key in ("A3", "A4", "H4"):
        check(r["images"][key].shape[:2] == (SIZE, SIZE), f"{path.name} {key}: wrong size")

    # 2. A3 maps TL, TR, BL exactly
    a3_err = np.abs(apply_affine(r["matrices"]["A3"], src[[0, 1, 3]]) - TARGET[[0, 1, 3]]).max()
    check(a3_err < 1e-3, f"{path.name}: A3 does not map TL, TR, BL exactly ({a3_err:.4f} px)")

    # 3. H4 maps all four corners exactly
    h4_err = np.abs(apply_homography(r["matrices"]["H4"], src) - TARGET).max()
    check(h4_err < 1e-3, f"{path.name}: H4 does not map the corners exactly ({h4_err:.4f} px)")

    # 4. A4 is the least-squares affine: compare with a direct least-squares solution
    A_ls = least_squares_affine(src, TARGET)
    rms_ls = np.sqrt(np.mean(np.sum((apply_affine(A_ls, src) - TARGET) ** 2, axis=1)))
    check(abs(r["a4_rms_px"] - rms_ls) < 0.05,
          f"{path.name}: A4 RMS {r['a4_rms_px']:.3f} differs from least squares {rms_ls:.3f}")

    # 5. A4 can never be worse than A3 at the BR corner... and A4 RMS should be smaller than A3's error
    check(r["a4_rms_px"] <= r["a3_corner_err_px"] + 1e-6,
          f"{path.name}: A4 RMS larger than A3 corner error")

    cond = "+".join(gt.get("conditions", []))
    print(f"{path.stem:<16}{cond:<10}{r['a3_corner_err_px']:>14.2f}{r['a4_rms_px']:>12.2f}{h4_err:>12.1e}")

    # Side by side A3 | A4 | H4 with the ideal 50 px grid drawn on top
    tiles = []
    for key in ("A3", "A4", "H4"):
        t = r["images"][key].copy()
        for k in range(10):
            cv2.line(t, (k * 50, 0), (k * 50, SIZE), (0, 0, 255), 1)
            cv2.line(t, (0, k * 50), (SIZE, k * 50), (0, 0, 255), 1)
        cv2.putText(t, key, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        tiles.append(t)
    gap = np.full((SIZE, 8, 3), 255, np.uint8)
    cv2.imwrite(str(OUT / f"{path.stem}_A3_A4_H4.png"), np.hstack([tiles[0], gap, tiles[1], gap, tiles[2]]))

print(f"\n{passed} checks passed, {failed} failed")
print(f"Side-by-side images (red = ideal grid every 50 px): {OUT}/")
sys.exit(1 if failed else 0)
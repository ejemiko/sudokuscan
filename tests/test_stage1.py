"""Stage 1 test: checks the preprocessing output on every image in data/.

Run from the project root:   python3 tests/test_stage1.py
Writes before/after images to out/tests/stage1/ for visual inspection.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage1_preprocess import preprocess, WORK_SIZE

DATA = Path("data")
OUT = Path("out/tests/stage1")
OUT.mkdir(parents=True, exist_ok=True)

passed, failed = 0, 0


def check(condition, message):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {message}")


# Test 1: output format on every image
print("Test 1: output size, type and value range")
for path in sorted(DATA.glob("*.jpg")):
    img = cv2.imread(str(path))
    t0 = time.time()
    gray, scale = preprocess(img)
    dt = time.time() - t0

    check(gray.ndim == 2, f"{path.name}: output is not single-channel")
    check(gray.dtype == np.float32, f"{path.name}: output is {gray.dtype}, expected float32")
    check(max(gray.shape) == min(WORK_SIZE, max(img.shape[:2])),
          f"{path.name}: longest side is {max(gray.shape)}, expected {WORK_SIZE}")
    check(abs(img.shape[0] * scale - gray.shape[0]) <= 1,
          f"{path.name}: scale {scale:.3f} does not match the output size")
    check(0 <= gray.min() and gray.max() <= 255, f"{path.name}: values outside 0-255")
    check(dt < 2.0, f"{path.name}: took {dt:.2f} s")

    # before/after image for visual inspection
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (gray.shape[1], gray.shape[0]))
    cv2.imwrite(str(OUT / f"{path.stem}_before_after.jpg"), np.hstack([small, gray.astype(np.uint8)]))
    print(f"  {path.name}: {img.shape[1]}x{img.shape[0]} -> {gray.shape[1]}x{gray.shape[0]}, "
          f"scale={scale:.3f}, {dt*1000:.0f} ms")

# Test 2: the same photo at 1000 px and 4000 px must give (almost) the same result.
# This is the "parameters relative to image size" requirement.
print("\nTest 2: size independence (same photo at 1000 px and 4000 px)")
img = cv2.imread(str(DATA / "s332501003_01.jpg"))
for target in (1000, 4000):
    s = target / max(img.shape[:2])
    resized = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    g, _ = preprocess(resized)
    if target == 1000:
        g1000 = g
    else:
        g4000 = g
diff = np.abs(g1000 - g4000).mean()
print(f"  mean absolute difference: {diff:.2f} grey levels (out of 255)")
check(diff < 5.0, f"outputs differ by {diff:.2f} grey levels")

# Test 3: illumination normalisation - the paper should become uniformly bright.
# On the shadow photo, compare brightness spread of the paper before and after.
print("\nTest 3: shadow removal on s332501003_15.jpg")
img = cv2.imread(str(DATA / "s332501003_15.jpg"))
gray, scale = preprocess(img)
before = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (gray.shape[1], gray.shape[0])).astype(np.float32)
# "paper" pixels: the bright 60% of the output, i.e. not ink
paper = gray > np.percentile(gray, 40)
spread_before = before[paper].std()
spread_after = gray[paper].std()
print(f"  paper brightness spread: before {spread_before:.1f}, after {spread_after:.1f}")
check(spread_after < spread_before / 2, "shadow was not reduced by at least half")

print(f"\n{passed} checks passed, {failed} failed")
print(f"Visual results: {OUT}/")
sys.exit(1 if failed else 0)
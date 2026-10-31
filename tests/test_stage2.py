"""Stage 2 test: Harris response, candidate points and corner selection.

Run from the project root:   python3 tests/test_stage2.py
Parts that are not written yet are skipped. Writes debug images to out/tests/stage2/.

Corner error is measured against the ground-truth corners in data/*.json, as a
percentage of the true puzzle side length (the grader's metric, Section 7).
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stage1_preprocess import preprocess
from stage2_corners import harris_response, find_candidates, select_corners
from sudoku_scan import save_harris_debug

DATA = Path("data")
OUT = Path("out/tests/stage2")
OUT.mkdir(parents=True, exist_ok=True)


def corner_error_percent(pred, true):
    """Mean distance between predicted and true corners, as % of the true side length."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    side = np.mean([np.linalg.norm(true[i] - true[(i + 1) % 4]) for i in range(4)])
    return 100 * np.mean(np.linalg.norm(pred - true, axis=1)) / side


rows = []
for path in sorted(DATA.glob("*.jpg")):
    img = cv2.imread(str(path))
    gt = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
    t0 = time.time()
    gray, scale = preprocess(img)

    # Part a: Harris response
    R = harris_response(gray)
    assert R.shape == gray.shape, f"{path.name}: R has shape {R.shape}, expected {gray.shape}"
    assert R.max() > 0, f"{path.name}: no positive response"
    row = {"name": path.stem, "R_max": R.max(), "n_cand": "-", "err": "-", "note": ""}

    # Part b: candidates (student)
    candidates, corners = None, None
    try:
        candidates = np.asarray(find_candidates(R))
        row["n_cand"] = len(candidates)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            row["note"] = f"candidates shape {candidates.shape}, expected (N, 2)"
    except NotImplementedError:
        row["note"] = "find_candidates not written"

    # Part c: corner selection (student)
    if candidates is not None:
        try:
            corners = select_corners(candidates, R, gray)
            if corners is None:
                row["note"] = "returned None (not found)"
            else:
                corners = np.asarray(corners, float)
                if gt.get("corners"):
                    row["err"] = corner_error_percent(corners / scale, gt["corners"])
        except NotImplementedError:
            row["note"] = "select_corners not written"
        except Exception as e:
            row["note"] = f"CRASH {type(e).__name__}: {e}"

    row["ms"] = (time.time() - t0) * 1000
    save_harris_debug(R, candidates, corners, OUT / f"{path.stem}_harris.png")
    rows.append(row)

print(f"{'image':<20}{'R max':>10}{'cands':>8}{'err %':>8}{'ms':>7}  note")
for r in rows:
    err = f"{r['err']:.2f}" if isinstance(r["err"], float) else r["err"]
    print(f"{r['name']:<20}{r['R_max']:>10.2e}{str(r['n_cand']):>8}{err:>8}{r['ms']:>7.0f}  {r['note']}")

errs = [r["err"] for r in rows if isinstance(r["err"], float)]
if errs:
    print(f"\nmean corner error: {np.mean(errs):.2f} %   "
          f"images under 2 %: {sum(e < 2 for e in errs)}/{len(errs)}")
print(f"Debug images: {OUT}/  (green = candidates, red = selected corners)")
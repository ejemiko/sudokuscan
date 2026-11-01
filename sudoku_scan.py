"""SudokuScan: finds a sudoku in a photo, rectifies it and reads it as a 9x9 grid.

Usage:
    python sudoku_scan.py --input photo.jpg --outdir out/
    python sudoku_scan.py --batch images_dir/ --outdir results/
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from stage1_preprocess import preprocess
from stage2_corners import harris_response, find_candidates, select_corners
from stage3_rectify import rectify
from stage5_recognize import recognize, draw_debug
from stage6_validate import validate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EMPTY_GRID = [[0] * 9 for _ in range(9)]


def save_harris_debug(R, candidates, corners, path):
    """Harris response map with all candidates (green) and the 4 selected corners (red)."""
    # Response values span a huge range; log scaling makes weak corners visible too.
    R_pos = np.log1p(np.maximum(R, 0))
    vis = cv2.normalize(R_pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if candidates is not None:
        for x, y in candidates:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    if corners is not None:
        for (x, y), label in zip(corners, ["TL", "TR", "BR", "BL"]):
            cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), 2)
            cv2.putText(vis, label, (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(str(path), vis)


def write_grid_txt(grid, path):
    with open(path, "w") as f:
        for row in grid:
            f.write("".join(str(d) for d in row) + "\n")


def process_image(path, outdir):
    """Runs the full pipeline on one image. Never crashes: on failure writes verdict=not_found."""
    start = time.time()
    name = path.stem
    result = {"image": path.name, "verdict": "not_found", "grid": EMPTY_GRID}
    R, candidates, corners_small = None, None, None

    try:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"could not read {path}")

        # Stage 1: grayscale, denoise, illumination; works on a downscaled copy
        gray, scale = preprocess(img)

        # Stage 2: Harris response -> candidate points -> 4 outer corners
        R = harris_response(gray)
        candidates = find_candidates(R)
        corners_small = select_corners(candidates, R, gray)
        if corners_small is None:
            raise ValueError("puzzle corners not found")
        # corners were found on the small image; convert back to original pixels
        corners = np.asarray(corners_small, dtype=np.float64) / scale
        result["corners"] = corners.tolist()

        # Stage 3: A3, A4, H4 rectifications
        rect = rectify(img, corners)
        for key in ("A3", "A4", "H4"):
            cv2.imwrite(str(outdir / f"{name}_rect_{key}.png"), rect["images"][key])
            result[key] = rect["matrices"][key].tolist()
        result["a3_corner_err_px"] = rect["a3_corner_err_px"]
        result["a4_rms_px"] = rect["a4_rms_px"]

        # Stage 5: cells + OCR on the H4 rectification
        grid, _ = recognize(rect["images"]["H4"])
        result["grid"] = grid

        # Stage 6: sudoku rules check; the debug image marks conflicting cells in red
        verdict, conflicts = validate(grid)
        result["verdict"] = verdict
        result["conflicts"] = conflicts
        debug_img = draw_debug(rect["images"]["H4"], grid, conflicts)
        cv2.imwrite(str(outdir / f"{name}_debug.png"), debug_img)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    finally:
        if R is not None:
            save_harris_debug(R, candidates, corners_small, outdir / f"{name}_harris.png")
        write_grid_txt(result["grid"], outdir / f"{name}_grid.txt")
        result["runtime_s"] = round(time.time() - start, 3)
        with open(outdir / f"{name}_result.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="SudokuScan")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="a single image")
    group.add_argument("--batch", type=Path, help="a directory of images")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.input:
        paths = [args.input]
    else:
        paths = sorted(p for p in args.batch.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    for path in paths:
        r = process_image(path, args.outdir)
        status = r["verdict"] + (f"  ({r['error']})" if "error" in r else "")
        print(f"{path.name}: {status}  [{r['runtime_s']} s]")


if __name__ == "__main__":
    main()
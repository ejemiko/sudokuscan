"""Evaluation metrics of Section 7, computed from the program's output files.

Usage:
    python sudoku_scan.py --batch data/ --outdir results/
    python evaluate.py --results results/ --data data/

For every annotation <name>.json in --data, reads <name>_result.json from --results.
Metrics (overall and per condition C1-C5; an image counts in every condition it has):
  corner error  mean distance between predicted and true corners, % of the true side length
  cell accuracy fraction of the 81 cells correct, averaged over images
  empty F1      F1 of the filled-versus-empty decision ("empty" = positive class), all cells pooled
  exact match   fraction of images with all 81 cells correct
  robustness    fraction of images with a result file and runtime <= 20 s
"""
import argparse
import json
from pathlib import Path

import numpy as np

TIME_LIMIT = 20.0
CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]


def corner_error_pct(pred, true):
    """Mean Euclidean corner distance as a percentage of the true puzzle side length."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    side = np.mean([np.linalg.norm(true[i] - true[(i + 1) % 4]) for i in range(4)])
    return np.mean(np.linalg.norm(pred - true, axis=1)) / side * 100


def evaluate_image(ann, res):
    """Per-image numbers; res is None when the result file is missing (= crash)."""
    truth = np.array(ann["grid"])
    if res is None:
        return {"robust": False, "corner": None, "truth": truth, "pred": np.zeros((9, 9), int)}
    pred = np.array(res.get("grid", np.zeros((9, 9), int)))
    corner = corner_error_pct(res["corners"], ann["corners"]) if "corners" in res else None
    robust = res.get("runtime_s", 0) <= TIME_LIMIT
    return {"robust": robust, "corner": corner, "truth": truth, "pred": pred}


def summarise(items):
    """Section 7 metrics for a list of per-image results."""
    t = np.concatenate([it["truth"].ravel() for it in items])
    p = np.concatenate([it["pred"].ravel() for it in items])
    tp = np.sum((t == 0) & (p == 0))
    prec = tp / max(1, np.sum(p == 0))
    rec = tp / max(1, np.sum(t == 0))
    corners = [it["corner"] for it in items if it["corner"] is not None]
    return {
        "n": len(items),
        "corner": np.mean(corners) if corners else float("nan"),
        "not_found": sum(it["corner"] is None for it in items),
        "cell_acc": np.mean([np.mean(it["truth"] == it["pred"]) for it in items]) * 100,
        "f1": 2 * prec * rec / max(1e-9, prec + rec),
        "exact": np.mean([np.all(it["truth"] == it["pred"]) for it in items]) * 100,
        "robust": np.mean([it["robust"] for it in items]) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="SudokuScan evaluation (Section 7)")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    items, per_image = [], []
    for ann_path in sorted(args.data.glob("*.json")):
        with open(ann_path) as f:
            ann = json.load(f)
        res_path = args.results / f"{ann_path.stem}_result.json"
        res = None
        if res_path.exists():
            with open(res_path) as f:
                res = json.load(f)
        it = evaluate_image(ann, res)
        it["name"], it["conditions"] = ann_path.stem, ann.get("conditions", [])
        items.append(it)

    print(f"{'image':<16} {'cond':<8} {'corner %':>8} {'cells %':>8} {'exact':>5}")
    for it in items:
        c = "not found" if it["corner"] is None else f"{it['corner']:.2f}"
        acc = np.mean(it["truth"] == it["pred"]) * 100
        print(f"{it['name']:<16} {'+'.join(it['conditions']):<8} {c:>8} {acc:>8.1f} "
              f"{'yes' if np.all(it['truth'] == it['pred']) else 'no':>5}")

    print(f"\n{'group':<8} {'n':>3} {'corner %':>9} {'cell acc %':>11} {'empty F1':>9} "
          f"{'exact %':>8} {'robust %':>9}")
    groups = [("all", items)] + [(c, [it for it in items if c in it["conditions"]]) for c in CONDITIONS]
    for label, group in groups:
        if not group:
            continue
        s = summarise(group)
        nf = f" ({s['not_found']} not found)" if s["not_found"] else ""
        print(f"{label:<8} {s['n']:>3} {s['corner']:>9.2f} {s['cell_acc']:>11.1f} {s['f1']:>9.3f} "
              f"{s['exact']:>8.1f} {s['robust']:>9.1f}{nf}")


if __name__ == "__main__":
    main()
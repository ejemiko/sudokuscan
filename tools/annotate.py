"""Ground-truth corner annotation tool.

Usage:
    python tools/annotate.py data/            # annotate images without a .json yet
    python tools/annotate.py data/ --redo     # re-annotate every image

Click the 4 outer corners of the puzzle grid in order TL, TR, BR, BL.
Keys:  u = undo last click   s = save and next   n = skip   q = quit
Coordinates are saved in original-image pixels. Existing "conditions" and
"grid" fields in the .json are kept.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

LABELS = ["TL", "TR", "BR", "BL"]
MAX_DISPLAY_HEIGHT = 900


def annotate(path):
    img = cv2.imread(str(path))
    scale = min(1.0, MAX_DISPLAY_HEIGHT / img.shape[0])
    small = cv2.resize(img, None, fx=scale, fy=scale)
    clicks = []  # in display coordinates

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    win = "annotate"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = small.copy()
        for (x, y), label in zip(clicks, LABELS):
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(vis, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(clicks) == 4:
            pts = clicks + [clicks[0]]
            for a, b in zip(pts, pts[1:]):
                cv2.line(vis, a, b, (0, 200, 0), 1)
        nxt = LABELS[len(clicks)] if len(clicks) < 4 else "done - press s"
        cv2.putText(vis, f"{path.name}  next: {nxt}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow(win, vis)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and clicks:
            clicks.pop()
        elif key == ord("s") and len(clicks) == 4:
            return [[x / scale, y / scale] for x, y in clicks]
        elif key == ord("n"):
            return None
        elif key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in images:
        json_path = path.with_suffix(".json")
        data = json.loads(json_path.read_text()) if json_path.exists() else {}
        if "corners" in data and not args.redo:
            continue
        corners = annotate(path)
        if corners is None:
            continue
        data = {"image": path.name,
                "conditions": data.get("conditions", []),
                "corners": [[round(x, 1), round(y, 1)] for x, y in corners],
                "grid": data.get("grid", [])}
        json_path.write_text(json.dumps(data, indent=2))
        print(f"saved {json_path.name}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

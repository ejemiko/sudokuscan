"""Writes the ground-truth grid and conditions into each image's .json.

Run once:  python tools/fill_metadata.py data/
Existing "corners" in the .json files are kept.
"""
import json
import sys
from pathlib import Path

# 0 = empty cell
GRIDS = {
    "A_krazydad": [
        [2, 0, 5, 0, 0, 7, 0, 0, 6],
        [4, 0, 0, 9, 6, 0, 0, 2, 0],
        [0, 0, 0, 0, 8, 0, 0, 4, 5],
        [9, 8, 0, 0, 7, 4, 0, 0, 0],
        [5, 7, 0, 8, 0, 2, 0, 6, 9],
        [0, 0, 0, 6, 3, 0, 0, 5, 7],
        [7, 5, 0, 0, 2, 0, 0, 0, 0],
        [0, 6, 0, 0, 5, 1, 0, 0, 2],
        [3, 0, 0, 4, 0, 0, 5, 0, 8],
    ],
    "B_mavo": [
        [0, 7, 0, 4, 3, 5, 8, 6, 0],
        [0, 0, 0, 0, 2, 6, 1, 7, 9],
        [0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 8, 0, 6, 0, 0],
        [0, 0, 0, 6, 0, 0, 0, 9, 4],
        [0, 2, 1, 7, 0, 9, 0, 3, 8],
        [0, 1, 7, 0, 5, 4, 0, 0, 0],
        [9, 4, 0, 3, 0, 0, 2, 0, 7],
        [0, 0, 0, 2, 0, 0, 9, 4, 0],
    ],
    "C_7sudoku": [
        [0, 0, 0, 0, 2, 1, 8, 0, 9],
        [0, 8, 0, 3, 5, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 0, 0, 5, 0],
        [0, 4, 2, 0, 0, 3, 0, 0, 0],
        [0, 0, 6, 0, 9, 0, 7, 0, 0],
        [0, 0, 0, 5, 0, 0, 2, 6, 0],
        [0, 7, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 3, 7, 0, 4, 0],
        [2, 0, 9, 4, 1, 0, 0, 0, 0],
    ],
}

# image number -> (puzzle, conditions)
IMAGES = {
    1: ("B_mavo", ["C1"]),
    2: ("C_7sudoku", ["C1"]),
    3: ("A_krazydad", ["C1"]),
    4: ("A_krazydad", ["C2"]),
    5: ("C_7sudoku", ["C2"]),
    6: ("B_mavo", ["C2"]),
    7: ("A_krazydad", ["C3"]),
    8: ("C_7sudoku", ["C3"]),
    9: ("B_mavo", ["C2", "C5"]),
    10: ("A_krazydad", ["C5"]),
    11: ("C_7sudoku", ["C4"]),
    12: ("B_mavo", ["C1"]),
    13: ("B_mavo", ["C5"]),
    14: ("C_7sudoku", ["C2", "C5"]),
    15: ("A_krazydad", ["C4"]),
}


def main(folder):
    for num, (puzzle, conditions) in IMAGES.items():
        img_path = next(Path(folder).glob(f"*_{num:02d}.jpg"))
        json_path = img_path.with_suffix(".json")
        data = json.loads(json_path.read_text()) if json_path.exists() else {}
        data = {"image": img_path.name,
                "conditions": conditions,
                "corners": data.get("corners", []),
                "grid": GRIDS[puzzle]}
        json_path.write_text(json.dumps(data, indent=2))
        print(f"{json_path.name}: {puzzle} {conditions}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")

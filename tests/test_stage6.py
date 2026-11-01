"""Stage 6 test: the validity check.

1. hand-made cases: a valid grid, and a repeat in a row, a column and a box
2. every annotated grid in data/ must be valid (real puzzles have no repeats)

Run from the project root:  python tests/test_stage6.py
"""
import copy
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage6_validate import validate

passed = failed = 0


def check(label, ok):
    global passed, failed
    passed += ok
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


EMPTY = [[0] * 9 for _ in range(9)]

g = copy.deepcopy(EMPTY)
g[0][0], g[4][4], g[8][8] = 5, 5, 5           # same digit, different row, column and box
check("valid grid", validate(g) == ("valid", []))

g = copy.deepcopy(EMPTY)
g[2][0], g[2][7] = 3, 3
check("repeat in a row", validate(g) == ("invalid", [[2, 0], [2, 7]]))

g = copy.deepcopy(EMPTY)
g[1][5], g[6][5] = 9, 9
check("repeat in a column", validate(g) == ("invalid", [[1, 5], [6, 5]]))

g = copy.deepcopy(EMPTY)
g[3][3], g[5][4] = 1, 1
check("repeat in a 3x3 box", validate(g) == ("invalid", [[3, 3], [5, 4]]))

check("empty grid is valid", validate(EMPTY) == ("valid", []))

for path in sorted(glob.glob("data/*.json")):
    with open(path) as f:
        grid = json.load(f)["grid"]
    check(f"annotation {os.path.basename(path)} is valid", validate(grid)[0] == "valid")

print(f"\n{passed}/{passed + failed} passed")
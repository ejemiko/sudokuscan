"""Stage 6: validity check of the recognised givens against the sudoku rules."""


def validate(grid):
    """Returns (verdict, conflicts).

    verdict:   "valid" if no digit repeats in any row, column or 3x3 box, else "invalid"
    conflicts: sorted list of [row, col] of every cell that takes part in a repeat
    Empty cells (0) are ignored.
    """
    groups = []
    for i in range(9):
        groups.append([(i, c) for c in range(9)])                    # row i
        groups.append([(r, i) for r in range(9)])                    # column i
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            groups.append([(br + r, bc + c) for r in range(3) for c in range(3)])   # 3x3 box

    conflicts = set()
    for cells in groups:
        seen = {}                                                    # digit -> cells holding it
        for r, c in cells:
            d = grid[r][c]
            if d != 0:
                seen.setdefault(d, []).append((r, c))
        for where in seen.values():
            if len(where) > 1:
                conflicts.update(where)

    conflicts = sorted([r, c] for r, c in conflicts)
    return ("invalid" if conflicts else "valid"), conflicts
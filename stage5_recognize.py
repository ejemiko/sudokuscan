"""Stage 5: cell extraction and digit recognition."""
import cv2
import numpy as np
import pytesseract

GRID = 9
MARGIN = 4          # px cut from every side of a cell (experiments/stage5_margin*.py)
EDGE_DIST = 6       # width of the outer band of a cell where grid-line remnants live (px)
NOISE_AREA = 15     # ink components smaller than this (px) are noise specks
EMPTY_FRAC = 0.02   # boş hücreler ~0, dolu en az %3,7 -> aradaki boşluğun ortası; seçen: experiments/stage5_empty.py
                    # share of the cell; chosen with experiments/stage5_empty.py
OCR_HEIGHT = 40     # the digit is scaled to this height before OCR (px)
PAD = 20            # white border added around the digit (px)
OCR_CONFIG = "--psm 8 -c tessedit_char_whitelist=123456789"


def binarise(rect_img):
    """Ink = 255, paper = 0. Adaptive threshold copes with shadows (photo 15)."""
    gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY) if rect_img.ndim == 3 else rect_img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def split_cells(img):
    """Equal 9x9 division; returns 81 cells row by row, MARGIN px cut from every side."""
    h, w = img.shape[:2]
    cells = []
    for r in range(GRID):
        for c in range(GRID):
            y0, y1 = round(r * h / GRID), round((r + 1) * h / GRID)
            x0, x1 = round(c * w / GRID), round((c + 1) * w / GRID)
            cells.append(img[y0 + MARGIN:y1 - MARGIN, x0 + MARGIN:x1 - MARGIN])
    return cells


def clean_cell(cell):
    """Removes noise specks and grid-line remnants from a binary cell.

    A component is a grid-line remnant if it touches the cell edge and more than
    half of its pixels lie in the outer EDGE_DIST band. A digit that only touches
    the edge (shifted digit, photo 06) keeps most of its pixels inside and survives.
    """
    h, w = cell.shape
    band = np.ones((h, w), dtype=bool)
    band[EDGE_DIST:h - EDGE_DIST, EDGE_DIST:w - EDGE_DIST] = False

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cell, connectivity=8)
    clean = np.zeros_like(cell)
    for i in range(1, n):                        # label 0 is the background
        x, y, cw, ch, area = stats[i]
        if area < NOISE_AREA:
            continue
        comp = labels == i
        touches = x == 0 or y == 0 or x + cw == w or y + ch == h
        if touches and band[comp].mean() > 0.5:
            continue
        clean[comp] = 255
    return clean


def ink_fraction(clean):
    """Share of the cell covered by the largest ink component (0 if there is none)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if n <= 1:
        return 0.0
    return stats[1:, cv2.CC_STAT_AREA].max() / clean.size


def prepare(clean):
    """Crops the digit to its bounding box, scales it to OCR_HEIGHT and pads it with
    white on every side (equal padding = centred). Black digit on white paper."""
    ys, xs = np.nonzero(clean)
    digit = clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = OCR_HEIGHT / digit.shape[0]
    new_w = max(1, round(digit.shape[1] * scale))
    digit = cv2.resize(digit, (new_w, OCR_HEIGHT), interpolation=cv2.INTER_LINEAR)
    _, digit = cv2.threshold(digit, 127, 255, cv2.THRESH_BINARY)
    digit = 255 - digit
    return cv2.copyMakeBorder(digit, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)


def read_digit(img, config=OCR_CONFIG):
    """Runs Tesseract on one prepared cell. Returns 1-9, or 0 if nothing was read."""
    text = pytesseract.image_to_string(img, config=config).strip()
    return int(text[0]) if text[:1].isdigit() and text[0] != "0" else 0


def draw_debug(rect_img, grid, conflicts=()):
    """Rectified image with cell boxes (green), recognised digits (blue) and
    conflicting cells (red, filled in by Stage 6)."""
    vis = rect_img.copy() if rect_img.ndim == 3 else cv2.cvtColor(rect_img, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    ch, cw = h / GRID, w / GRID
    for r in range(GRID):
        for c in range(GRID):
            x0, y0 = int(c * cw), int(r * ch)
            x1, y1 = int((c + 1) * cw) - 1, int((r + 1) * ch) - 1
            cv2.rectangle(vis, (x0 + MARGIN, y0 + MARGIN), (x1 - MARGIN, y1 - MARGIN), (0, 200, 0), 1)
            if grid[r][c]:
                cv2.putText(vis, str(grid[r][c]), (x0 + MARGIN + 2, y0 + MARGIN + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for r, c in conflicts:
        cv2.rectangle(vis, (int(c * cw) + 1, int(r * ch) + 1),
                      (int((c + 1) * cw) - 2, int((r + 1) * ch) - 2), (0, 0, 255), 3)
    return vis


def recognize(rect_img):
    """Returns (grid, debug_img): a 9x9 list of digits (0 = empty) and a debug image."""
    bw = binarise(rect_img)
    grid = [[0] * GRID for _ in range(GRID)]
    for k, cell in enumerate(split_cells(bw)):
        clean = clean_cell(cell)
        if ink_fraction(clean) < EMPTY_FRAC:     # empty/filled is decided BEFORE OCR
            continue
        grid[k // GRID][k % GRID] = read_digit(prepare(clean))
    return grid, draw_debug(rect_img, grid)
# SudokuScan

COMP 482 Computer Vision, Project 1 · Ecem Yaşar (332501003) · MEF University

SudokuScan finds a printed sudoku in a phone photo, detects its four outer corners with the
Harris detector, rectifies the grid with three geometric models (A3, A4, H4), reads the 81 cells
with Tesseract and checks the result against the sudoku rules.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| 1 | `stage1_preprocess.py` | Downscale (longest side 1000 px), grayscale, Gaussian denoising, illumination correction (division by an estimated background) |
| 2 | `stage2_corners.py` | Harris response map, candidate points (relative threshold + 7×7 non-maximum suppression), puzzle-region search, selection of the four outer corners |
| 3 | `stage3_rectify.py` | A3: affine from three corners (TL, TR, BL); A4: least-squares affine from four corners; H4: homography from four corners. Target 450×450 px. Reports the A3 fourth-corner error and the A4 RMS residual |
| 4 | `stage4_residual.py`, `stage4_grid.py`, `stage4_ocr.py` | Affine versus homography comparison (see below) |
| 5 | `stage5_recognize.py` | 81 cells with a 4 px inner margin, grid-line remnant removal, empty/filled decision before OCR, cell preparation, Tesseract (`--psm 8`, whitelist 1–9) |
| 6 | `stage6_validate.py` | Repeated digits in rows, columns and 3×3 boxes → `valid` / `invalid`, conflicting cells marked in the debug image |

The four final corners always come from `cv2.cornerHarris`. Contours are only used to find the
region that contains the puzzle; their vertices are never used as corner coordinates.

## Installation

Python 3.10 or newer (developed with Python 3.14 on macOS).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**OCR engine.** `pytesseract` is only a wrapper; the Tesseract program itself must be installed
(developed with Tesseract 5.5):

| System | Command |
|---|---|
| macOS (Homebrew) | `brew install tesseract` |
| Ubuntu / Debian | `sudo apt install tesseract-ocr` |
| Windows | installer from https://github.com/UB-Mannheim/tesseract/wiki, then add it to `PATH` |

Check with `tesseract --version`.

## Usage

```bash
python sudoku_scan.py --input photo.jpg --outdir out/          # one image
python sudoku_scan.py --batch images_dir/ --outdir results/    # grader mode
```

For every input image `<name>` the output directory receives:

| File | Content |
|---|---|
| `<name>_grid.txt` | 9 lines of 9 digits, 0 for empty cells (from the H4 rectification) |
| `<name>_result.json` | corners (TL, TR, BR, BL in original pixels), A3, A4, H4 matrices, `a3_corner_err_px`, `a4_rms_px`, grid, verdict, conflicts, runtime |
| `<name>_harris.png` | Harris response map with all candidates (green) and the four selected corners (red) |
| `<name>_rect_A3.png`, `_rect_A4.png`, `_rect_H4.png` | the three 450×450 rectifications |
| `<name>_debug.png` | H4 rectification with cell boxes, recognised digits and conflicting cells (red) |

The program never crashes on a bad image: if no puzzle is found it writes a result file with
`"verdict": "not_found"`.

## Evaluation

```bash
python sudoku_scan.py --batch data/ --outdir results/
python evaluate.py --results results/ --data data/
```

`evaluate.py` computes the Section 7 metrics (corner error, cell accuracy, empty-cell F1, grid
exact match, robustness), overall and per condition C1–C5.

Results on the personal dataset (15 photos):

| Corner error | Cell accuracy | Empty-cell F1 | Exact match | Robustness | Time per photo |
|---|---|---|---|---|---|
| 0.93 % | 97.4 % | 0.983 | 86.7 % | 100 % | 1.3–2.8 s |

## Stage 4: affine versus homography

Run after the batch command above (the scripts read `results/`):

```bash
python stage4_residual.py   # A3 fourth-corner error and A4 RMS vs obliqueness -> out/stage4/residual_vs_obliqueness.png
python stage4_grid.py       # deviation of the grid lines from the ideal positions (projection profiles)
python stage4_ocr.py        # OCR cell accuracy on the A3, A4 and H4 rectifications
```

Obliqueness is the larger of the two opposite-side length ratios (long/short) of the detected
quadrilateral; it is 1 for a frontal view.

## Tests and experiments

```bash
python tests/test_stage1.py   # preprocessing
python tests/test_stage2.py   # corner error against the annotations
python tests/test_stage3.py   # A3 / A4 / H4 and their residuals
python tests/test_stage5.py   # cell accuracy, empty-cell F1, with/without preparation, psm modes, confusions
python tests/test_stage6.py   # validity check
```

Experiments used to choose parameters (`experiments/`):

| Script | Purpose |
|---|---|
| `harris_test.py` | first look at the Harris threshold and blockSize |
| `param_study.py` | Harris parameter study (blockSize, ksize, k) on all photos |
| `stage5_margin.py`, `stage5_margin_ocr.py` | choice of the cell margin |
| `stage5_empty.py` | choice of the empty-cell threshold |

Debug images are written to `out/`.

## Repository layout

```
sudoku_scan.py            command-line interface, runs all stages
stage1_preprocess.py ... stage6_validate.py
stage4_residual.py, stage4_grid.py, stage4_ocr.py
evaluate.py               Section 7 metrics
tests/                    one test script per stage
experiments/              parameter studies
tools/annotate.py         corner annotation tool
tools/fill_metadata.py    writes grids and conditions into the annotation files
data/                     personal dataset: photos + annotations (Appendix A format)
```

## Use of AI tools

Claude (Anthropic) was used as described in `AI_LOG.md` and in Appendix C of the report.
The corner-selection logic (`find_candidates`, `find_puzzle_region`, `select_corners` in
`stage2_corners.py`) and the Stage 4 comparison analysis (`stage4_residual.py`,
`stage4_grid.py`, `stage4_ocr.py`) were written by me without AI-generated code.
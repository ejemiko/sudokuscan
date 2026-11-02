# AI use log

Tool: **Claude** (Anthropic), used through claude.ai as a tutor and coding assistant.
The assignment (Section 11) does not allow AI to write the corner-selection logic (Stage 2)
or the comparison analysis (Stage 4). For those two parts Claude only explained concepts,
asked questions and pointed at errors; the code and the conclusions are mine.
Every parameter that is justified in the report was chosen by me from experiment results.

## By stage

### Data and tools
- **Asked:** a tool to click the four corners of each photo and a script that writes the
  grids and conditions into the annotation files (Appendix A format).
- **AI wrote:** `tools/annotate.py`, `tools/fill_metadata.py`.
- **Me:** took the 15 photos, clicked every corner, typed the grids. I found that the first
  version of `fill_metadata.py` wrote the wrong puzzle/conditions for 6 files (file numbers
  follow capture time, not the order I sent the photos in); it was corrected.

### Stage 1 – preprocessing
- **Asked:** downscaling, grayscale, denoising and shadow correction, with explanations.
- **AI wrote:** `stage1_preprocess.py`, `tests/test_stage1.py`.
- **Me:** checked the results on the shadow photo (15) and the wooden-table photo (11),
  noted the side effects (shadow-edge traces, wood grain) for the failure analysis.

### Stage 2 – Harris corners and corner selection
- **AI wrote:** `harris_response` (the `cv2.cornerHarris` call with size-relative blockSize),
  `tests/test_stage2.py`, `experiments/param_study.py` (the loop only).
- **Me (no AI code):** `find_candidates` (relative threshold + 7×7 non-maximum suppression),
  `find_puzzle_region` (Otsu, RETR_CCOMP outer contours, convex hull, 4-vertex test, area and
  ink filters, most Harris candidates inside) and `select_corners` (candidates inside the
  region with a 3 % margin, extreme x+y / x−y). Claude explained Harris, NMS, Otsu, contour
  hierarchies, convexHull and approxPolyDP and asked questions; when my code had bugs it gave
  hints, and I fixed them.
- **Parameters, chosen by me from the data:** THRES_FRAC 0.002 (threshold study),
  MAX_REGION_FRAC 0.48 and MAX_INK_FRAC 0.27 (middle of the gap between puzzle and
  non-puzzle regions in my photos), BLOCK_FRAC 0.005 kept after the blockSize/ksize/k study
  (0.003 gave 15/15 on my set, but its neighbour 0.002 already fails on photo 06, so I kept
  the value whose neighbours are both good).
- **Bug:** a comment edit merged the `MIN_REGION_FRAC` definition into a comment line; Claude
  helped me locate it, I fixed it.

### Stage 3 – rectification
- **Asked:** A3, A4 (least squares) and H4 with the A3 fourth-corner error and A4 RMS.
- **AI wrote:** `stage3_rectify.py`, `tests/test_stage3.py` (checks A4 against `np.linalg.lstsq`).
- **Me:** studied the degrees of freedom and why A3 cannot model perspective.

### Stage 4 – affine versus homography
- **Me (no AI code):** `stage4_residual.py`, `stage4_grid.py`, `stage4_ocr.py` and all
  conclusions.
- **Claude:** discussed options for defining obliqueness (larger ratio, product of ratios,
  top/bottom only); I chose the larger opposite-side ratio. It listed the steps and the
  functions to use (glob, json, np.linalg.norm, matplotlib, projection profiles with
  `sum(axis)`), asked guiding questions (why photo 09 looks oblique, why A3 beats H4 on 09),
  and gave one throw-away terminal command to print each grid line of one image while
  debugging (not part of the submitted code). From that output I decided not to measure the
  line at 450 px, because it lies on the image border.

### Stage 5 – cells and OCR
- **AI wrote:** `stage5_recognize.py`, `tests/test_stage5.py`, `experiments/stage5_margin.py`,
  `experiments/stage5_margin_ocr.py`, `experiments/stage5_empty.py`.
- **Me:** chose the 4 px margin from the margin tables and images (larger margins cut the large
  KrazyDad digits and shifted digits in photo 06); chose EMPTY_FRAC = 0.02 (middle of the gap
  between empty and filled cells; Claude suggested the middle of the gap); chose `--psm 8`
  after the psm comparison (10 / 8 / 13).

### Stage 6 – validity check and CLI
- **AI wrote:** `stage6_validate.py`, `tests/test_stage6.py`, `sudoku_scan.py` and the change
  that draws the debug image after the validity check.

### Evaluation and documentation
- **AI wrote:** `evaluate.py` (Section 7 metrics), first drafts of `README.md`,
  `requirements.txt` and this log.
- **Me:** checked and edited them.

### Report
- **Me:** wrote the report. Claude reviewed language and checked that numbers match the
  tables; the Stage 4 discussion and the conclusion are my own text.

## Rejected or changed suggestions
- Moving the stage files into a `pipeline/` folder (my idea) was postponed: it would break all
  imports a few days before the deadline and the assignment does not require it.
- blockSize 0.003 was rejected in favour of 0.005 (see Stage 2).
- The line at 450 px was dropped from the grid-deviation measure instead of a "weak peak"
  rule, because that rule would hide exactly the A3 misalignment we want to measure.
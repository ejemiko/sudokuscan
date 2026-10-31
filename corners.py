"""Stage 2: Harris corner detection and automatic corner selection."""


def harris_response(gray):
    """Returns the Harris response map R (same size as gray)."""
    raise NotImplementedError("Stage 2a not written yet")


def find_candidates(R):
    """Turns the response map into discrete candidate points.

    Threshold R relative to its maximum, then apply non-maximum suppression
    so that each physical corner gives exactly one point.
    Returns an (N, 2) array of (x, y) coordinates.
    """
    # TODO (student): threshold + non-maximum suppression
    raise NotImplementedError("find_candidates: student writes this")


def select_corners(candidates, R, gray):
    """Chooses the four outer corners of the puzzle from the candidates.

    Must ignore candidates from background clutter, the page edge and other text.
    Returns a (4, 2) array ordered TL, TR, BR, BL, or None if no puzzle is found.
    """
    # TODO (student): corner-selection rule
    raise NotImplementedError("select_corners: student writes this")

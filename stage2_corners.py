"""Stage 2: Harris corner detection and automatic corner selection."""
import cv2
import numpy as np

# Harris parameters. The working image always has a longest side of ~1000 px
# (see stage1_preprocess.WORK_SIZE); blockSize is still defined as a fraction
# of the image size so it stays correct if the working size changes.
BLOCK_FRAC = 0.005   # neighbourhood window: 0.5% of the longest side (5 px at 1000 px)
KSIZE = 3            # Sobel aperture for the image derivatives
K = 0.04             # Harris free parameter (typical range 0.04-0.06)
THRES_FRAC = 0.01  # aday eşiği, R nin max inin %1 i


def harris_response(gray):
    """Returns the Harris response map R (float32, same size as gray).

    For every pixel, R = det(M) - k * trace(M)^2, where M is the second-moment
    matrix of the image gradients in a blockSize x blockSize window.
    R >> 0: corner, R < 0: edge, R ~ 0: flat region.
    """
    size = max(gray.shape)
    block_size = max(2, int(round(BLOCK_FRAC * size)))
    return cv2.cornerHarris(np.float32(gray), blockSize=block_size, ksize=KSIZE, k=K)


def find_candidates(R):
    """Turns the response map into discrete candidate points.

    Threshold R relative to its maximum, then apply non-maximum suppression
    so that each physical corner gives exactly one point.
    Returns an (N, 2) array of (x, y) coordinates.
    """
    threshold=THRES_FRAC*R.max()
    strong = R > threshold
    
    R_max_local=cv2.dilate(R,np.ones((7,7),np.uint8))
    is_peak = R == R_max_local
    keep = strong & is_peak #keep hem eşiği geçen hem de 7x7 komşuluğunda en yüksek yanıtı veren pikselleri tutar
    
    candidates = np.argwhere(keep)[:,::-1]
    return candidates
    

def select_corners(candidates, R, gray):
    """Chooses the four outer corners of the puzzle from the candidates.

    Must ignore candidates from background clutter, the page edge and other text.
    Returns a (4, 2) array ordered TL, TR, BR, BL, or None if no puzzle is found.
    """
    # TODO (student): corner-selection rule
    raise NotImplementedError("select_corners: student writes this")
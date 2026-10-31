"""Stage 3: rectification with affine (A3, A4) and projective (H4) models.

The four detected corners (TL, TR, BR, BL) are mapped onto a 450x450 square
(50 px per cell) with three different models:

  A3  affine from 3 points (TL, TR, BL)        6 unknowns, 3 points x 2 equations = 6 -> exact
  A4  affine from all 4 points, least squares  6 unknowns, 8 equations -> best fit, residual left
  H4  homography from all 4 points             8 unknowns, 8 equations -> exact
"""
import cv2
import numpy as np

SIZE = 450  # target square: 9 cells x 50 px

# Where each corner should land in the rectified image (TL, TR, BR, BL)
TARGET = np.array([[0, 0], [SIZE, 0], [SIZE, SIZE], [0, SIZE]], dtype=np.float32)


def apply_affine(A, points):
    """Applies a 2x3 affine matrix to an (N, 2) array of points."""
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((len(points), 1))
    return np.hstack([points, ones]) @ A.T          # [x y 1] * A^T -> [x' y']


def apply_homography(H, points):
    """Applies a 3x3 homography to an (N, 2) array of points (with the division by w)."""
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((len(points), 1))
    p = np.hstack([points, ones]) @ H.T              # [x' y' w]
    return p[:, :2] / p[:, 2:3]                      # divide by w: this is the perspective part


def rectify(img, corners):
    """Returns a dict with the three rectified images, the three matrices and the residuals.

    corners: (4, 2) array TL, TR, BR, BL in original-image pixels.
    """
    src = np.asarray(corners, dtype=np.float32)

    # A3: exact affine from three points. BR (index 2) is not used.
    three = [0, 1, 3]                                # TL, TR, BL
    A3 = cv2.getAffineTransform(src[three], TARGET[three])

    # A4: affine fitted to all four points by least squares.
    # estimateAffine2D normally uses RANSAC to reject outliers; with a huge
    # reprojection threshold all four points count as inliers, and its final
    # refinement then minimises the squared error over all four points.
    A4, _ = cv2.estimateAffine2D(src, TARGET, ransacReprojThreshold=1e6)

    # H4: homography from all four points (exact, 8 equations for 8 unknowns)
    H4 = cv2.getPerspectiveTransform(src, TARGET)

    images = {
        "A3": cv2.warpAffine(img, A3, (SIZE, SIZE)),
        "A4": cv2.warpAffine(img, A4, (SIZE, SIZE)),
        "H4": cv2.warpPerspective(img, H4, (SIZE, SIZE)),
    }

    # A3 fourth-corner error: where does the unused BR corner land, and how far from (450, 450)?
    br_mapped = apply_affine(A3, src[[2]])[0]
    a3_corner_err = float(np.linalg.norm(br_mapped - TARGET[2]))

    # A4 RMS residual: root-mean-square distance between mapped corners and their targets
    a4_mapped = apply_affine(A4, src)
    a4_rms = float(np.sqrt(np.mean(np.sum((a4_mapped - TARGET) ** 2, axis=1))))

    return {
        "images": images,
        "matrices": {"A3": A3, "A4": A4, "H4": H4},
        "a3_corner_err_px": round(a3_corner_err, 3),
        "a4_rms_px": round(a4_rms, 3),
        "a3_br_mapped": br_mapped.tolist(),
    }
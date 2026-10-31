"""Stage 1: preprocessing (grayscale, noise reduction, illumination normalisation).

All kernel sizes are fractions of the working image size, so the same code
behaves the same on a 1000 px and a 4000 px photograph.
"""
import cv2
import numpy as np

WORK_SIZE = 1000          # longest side of the working image, in pixels
BLUR_FRAC = 0.003         # noise blur kernel, as a fraction of the longest side
BACKGROUND_FRAC = 0.05    # background-estimate kernel, as a fraction of the longest side


def odd(n):
    """Kernel sizes must be odd and at least 3."""
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


def preprocess(img):
    """Returns (gray, scale).

    gray:  float32 image, illumination-normalised, values roughly 0-255
    scale: downscale factor, so that small = original * scale
    """
    # 1. Downscale large photos so the longest side is WORK_SIZE (never upscale)
    h, w = img.shape[:2]
    scale = min(1.0, WORK_SIZE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    size = max(img.shape[:2])

    # 2. Grayscale: Harris only needs brightness changes, not colour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Noise reduction: small Gaussian blur removes sensor noise and paper texture
    k = odd(BLUR_FRAC * size)
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    # 4. Illumination normalisation: estimate the background (paper + shadows) by
    #    removing the dark ink with a morphological closing, then divide by it.
    #    A shadow darkens ink and paper by the same factor, so the ratio cancels it.
    kb = odd(BACKGROUND_FRAC * size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (kb, kb), 0)
    norm = gray.astype(np.float32) / (background.astype(np.float32) + 1.0)
    norm = np.clip(norm * 255.0, 0, 255)

    return norm.astype(np.float32), scale
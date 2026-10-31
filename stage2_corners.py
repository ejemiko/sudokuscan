"""Stage 2: Harris corner detection and automatic corner selection."""
import cv2
import numpy as np

# Harris parameters. The working image always has a longest side of ~1000 px
# (see stage1_preprocess.WORK_SIZE); blockSize is still defined as a fraction
# of the image size so it stays correct if the working size changes.
BLOCK_FRAC = 0.005   # neighbourhood window: 0.5% of the longest side (5 px at 1000 px)
KSIZE = 3            # Sobel aperture for the image derivatives
K = 0.04             # Harris free parameter (typical range 0.04-0.06)
THRES_FRAC = 0.002  # aday eşiği: R'ın maksimumunun %0.2'si (eğik fotoğraflarda uzak köşeler zayıf, 0.01'de kayboluyordu)MIN_REGION_FRAC = 0.01 ## bölge en az görüntünün %1'i olmalı (harfleri, rakamları eler)
APPROX_FRAC = 0.02  # Çokgen sadeleştirme toleransı: hull çevresinin %2'si
MAX_REGION_FRAC = 0.48   # sudoku görüntünün en fazla %48'i olabilir (masa/kağıt gibi büyük şekilleri eler)
MAX_INK_FRAC = 0.27      # dörtgenin içi en fazla %27 mürekkep olabilir (defter, telefon, fotoğrafı eler)
MARGIN_FRAC = 0.03  # Bölge dışına tolerans: yaklaşık grid kenarının %3'ü


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


def find_puzzle_region(gray, candidates):
    """Finds the quadrilateral region of the puzzle grid."""
    g8 = gray.astype(np.uint8)
    otsu_t,bw = cv2.threshold(g8,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,hierarchy = cv2.findContours(bw,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    H,W=gray.shape
    best_quad = None
    best_count = 0
    
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] != -1:
            continue
        
        if cv2.contourArea(contour) < MIN_REGION_FRAC * H * W:
            continue
        
        hull = cv2.convexHull(contour)
        epsilon = APPROX_FRAC * cv2.arcLength(hull, True)
        quad = cv2.approxPolyDP(hull, epsilon, True)

        if len(quad) != 4:
            continue
        
        if cv2.contourArea(quad) > MAX_REGION_FRAC * H * W:
            continue
        
        mask = np.zeros_like(bw)
        cv2.fillPoly(mask, [quad], 255)
        ink = (bw[mask > 0] > 0).mean()
        
        if ink > MAX_INK_FRAC:
            continue
        
        count = 0
        for x, y in candidates:
            result = cv2.pointPolygonTest(quad, (float(x), float(y)), False)
            if result >= 0:
                count += 1
                
        if count > best_count:
            best_count = count
            best_quad = quad
    return best_quad
                

def select_corners(candidates, R, gray):
    """Chooses the four outer corners of the puzzle."""
    quad = find_puzzle_region(gray, candidates)
    
    if quad is None:
        return None
    
    side = np.sqrt(cv2.contourArea(quad))
    margin = MARGIN_FRAC * side

    inside = []
    for x, y in candidates:
        distance = cv2.pointPolygonTest(
            quad, (float(x), float(y)), True
        )
        if distance >= -margin: #içerideki, sınırdaki ve dışarıda tolerans mesafesi içindeki adayları tutar
            inside.append((x, y))

    inside = np.array(inside)
    
    if len(inside) < 4:
        return None

    s = inside[:, 0] + inside[:, 1]
    d = inside[:, 0] - inside[:, 1]

    tl = inside[np.argmin(s)]
    tr = inside[np.argmax(d)]
    br = inside[np.argmax(s)]
    bl = inside[np.argmin(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)
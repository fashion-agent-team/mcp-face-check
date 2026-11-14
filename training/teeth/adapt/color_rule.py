"""Color rule for food on teeth, crop only (no landmarks, like the server's mouthBox crop).
1) teeth = bright, low-chroma pixels in the middle of the crop
2) candidates = pixels right next to the teeth that are red/orange, green, yellow-brown or dark
3) keep compact blobs mostly surrounded by teeth (a bit stuck on/between teeth, not the lip or gum strip)
score = largest kept blob area / teeth area x 1000."""
import cv2, numpy as np

def color_score(bgr, ring_min=0.45):
    n = 256
    img = cv2.resize(bgr, (n, n), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    C = np.hypot(A, B); H = np.degrees(np.arctan2(B, A))
    roi = np.zeros((n, n), bool); m = int(n * 0.12); roi[m:n - m, m:n - m] = True
    teeth = roi & (L >= min(np.percentile(L[roi], 65), 235)) & (C < 28) & (A < 18)
    teeth = cv2.morphologyEx(teeth.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k, lbl, st, _ = cv2.connectedComponentsWithStats(teeth)
    keep = np.zeros((n, n), np.uint8)
    for i in range(1, k):
        if st[i, cv2.CC_STAT_AREA] >= 40: keep[lbl == i] = 1
    tarea = int(keep.sum())
    if tarea < 250: return 0.0, None
    tb = keep.astype(bool); tL = float(np.median(L[tb]))
    near = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool) & ~tb & roi
    kinds = {"red": (A > 14) & (C > 30) & (H > 38) & (H < 75), "green": (A < -5) & (C > 8),
             "brown": (B > 20) & (L < tL - 25), "dark": L < tL - 70}
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    best, kind = 0.0, None
    for name, cm in kinds.items():
        msk = (near & cm).astype(np.uint8)
        kk, lb, st2, _ = cv2.connectedComponentsWithStats(msk)
        for i in range(1, kk):
            a = st2[i, cv2.CC_STAT_AREA]
            if a < 5: continue
            w, h = st2[i, cv2.CC_STAT_WIDTH], st2[i, cv2.CC_STAT_HEIGHT]
            if name == "dark" and h / max(w, 1) > 2.0: continue  # thin gap between teeth
            blob = (lb == i).astype(np.uint8)
            ring = cv2.dilate(blob, ker).astype(bool) & ~blob.astype(bool)
            tf = tb[ring].mean()
            if tf < ring_min: continue
            sc = 1000 * a * tf / tarea
            if sc > best: best, kind = float(sc), name
    return best, kind

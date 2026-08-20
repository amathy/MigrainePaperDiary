"""Rectification: get from a photographed page to the canonical template frame.

Three stages, each strictly more local than the last, and each one only kept
if it measurably improves agreement with the clean template:

``markers``
    The four printed ArUco markers give an initial homography.  Because the
    marker layout is known exactly, missed markers can be recovered from the
    detector's rejected candidates using an ArUco *board*.

``anchors``
    Printed landmarks - checkbox borders, day numbers, column headers, footer
    text - are located by *masked* normalised cross-correlation in a
    coarse-to-fine sweep, and a corrected homography is fitted through them
    with RANSAC.  The masks exclude every checkbox interior, so what is
    compared is only ink the printer put down.

``field``
    A low-order polynomial displacement field, fitted robustly through the
    anchor inliers, mops up what a homography structurally cannot: page curl
    and lens distortion, which are not projective.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import template as T

MAX_FIELD_SHIFT = 70.0      # a displacement field larger than this is nonsense


# ------------------------------------------------------------------ markers

def detector(refine_win: int = 7) -> "cv2.aruco.ArucoDetector":
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = refine_win
    p.cornerRefinementMaxIterations = 60
    p.cornerRefinementMinAccuracy = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 63
    p.adaptiveThreshWinSizeStep = 8
    p.minMarkerPerimeterRate = 0.012
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minCornerDistanceRate = 0.03
    p.errorCorrectionRate = 0.6
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(T.ARUCO_DICT), p)


@lru_cache(maxsize=16)
def _board(month: int) -> "cv2.aruco.Board":
    """The four corner markers as a planar ArUco board, in canonical units."""
    model = T.month_model(month)
    ids, objs = [], []
    for key, corners in sorted(model["markers"].items(), key=lambda kv: int(kv[0])):
        ids.append(int(key))
        pts = np.asarray(corners, np.float32)
        objs.append(np.column_stack([pts, np.zeros(4, np.float32)]))
    return cv2.aruco.Board(np.asarray(objs, np.float32),
                           cv2.aruco.getPredefinedDictionary(T.ARUCO_DICT),
                           np.asarray(ids, np.int32))


def _resize(gray: np.ndarray, s: float) -> np.ndarray:
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=interp)


def _variants(gray: np.ndarray):
    """Progressively more aggressive views of the image for marker detection.

    Marker decoding is sensitive to how many pixels a marker spans, so the
    image is offered at several working sizes as well as with local contrast
    equalisation and unsharp masking for dim or soft photos.
    """
    yield gray, 1.0
    h, w = gray.shape
    long_side = max(h, w)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    targets = [2000, 1500, 2600, 1100]
    if long_side < 1400:          # small frames: decode from an upscaled copy
        targets = [long_side * 2, int(long_side * 1.5), 2000] + targets
    for target in targets:
        s = target / long_side
        if abs(1.0 - s) < 0.12:
            continue
        yield _resize(gray, s), s

    yield clahe.apply(gray), 1.0
    yield cv2.addWeighted(gray, 1.9, cv2.GaussianBlur(gray, (0, 0), 3.0), -0.9, 0), 1.0
    if long_side > 1700:
        s = 1700 / long_side
        yield clahe.apply(_resize(gray, s)), s
    else:
        s = min(2.5, 2200 / long_side)
        up = _resize(gray, s)
        yield clahe.apply(up), s
        yield cv2.addWeighted(up, 1.9, cv2.GaussianBlur(up, (0, 0), 3.0), -0.9, 0), s


def detect_markers(gray: np.ndarray) -> Dict[int, np.ndarray]:
    """All diary markers found anywhere in the image, in original pixels.

    Each variant of the image is searched; once a plausible month emerges the
    known board layout is used to pull any missed markers back out of the
    detector's rejected candidates.
    """
    det = detector()
    found: Dict[int, np.ndarray] = {}
    for variant, scale in _variants(gray):
        corners, ids, rejected = det.detectMarkers(variant)
        if ids is not None and len(ids):
            month = _month_vote([int(i) for i in ids.ravel()])
            if month is not None and rejected:
                try:
                    corners, ids, _, _ = det.refineDetectedMarkers(
                        variant, _board(month), list(corners), ids, list(rejected))
                except cv2.error:
                    pass
        if ids is None:
            continue
        for i, c in zip(np.asarray(ids).ravel(), corners):
            found.setdefault(int(i), np.asarray(c, np.float64).reshape(4, 2) / scale)
        if len(found) >= 4:
            break
    return found


def _month_vote(ids: List[int]) -> Optional[int]:
    counts: Dict[int, int] = {}
    for i in ids:
        counts[i // 4 + 1] = counts.get(i // 4 + 1, 0) + 1
    return max(counts, key=lambda m: counts[m]) if counts else None


def month_hypotheses(ids) -> List[Tuple[int, List[int]]]:
    groups: Dict[int, List[int]] = {}
    for i in sorted(ids):
        groups.setdefault(i // 4 + 1, []).append(i)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


def fit_homography(src: np.ndarray, dst: np.ndarray,
                   thresh: float) -> Tuple[Optional[np.ndarray], int]:
    """RANSAC to throw out genuine outliers, then least squares on what is left.

    Plain RANSAC is the wrong tool on its own here.  When the page is curled,
    *no* homography fits every point, so a tight RANSAC threshold declares
    most of the good correspondences outliers and locks onto whichever
    minimal subset it can fit exactly - which skews the whole page.  A
    generous threshold rejects only true mis-detections, and the subsequent
    least-squares refit spreads the unavoidable residual evenly instead of
    dumping it all on one side.  The leftover non-projective part is then the
    displacement field's job.
    """
    src, dst = np.float32(src), np.float32(dst)
    if len(src) < 4:
        return None, 0
    if len(src) == 4:
        try:
            return cv2.getPerspectiveTransform(src, dst), 4
        except cv2.error:
            return None, 0

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh,
                                 maxIters=8000, confidence=0.9995)
    if H is None:
        return None, 0
    keep = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), bool)
    if keep.sum() >= 4:
        refit, _ = cv2.findHomography(src[keep], dst[keep], 0)
        if refit is not None:
            H = refit
    return H, int(keep.sum())


def marker_homography(markers: Dict[int, np.ndarray], month: int,
                      ids: List[int]) -> Tuple[Optional[np.ndarray], int]:
    model = T.month_model(month)
    src, dst = [], []
    for i in ids:
        src.extend(markers[i].tolist())
        dst.extend(model["markers"][str(i)])
    # Generous threshold: all four markers really are printed on the page, so
    # anything within a marker's own width is model error, not a wrong match.
    return fit_homography(np.float32(src), np.float32(dst), 30.0)


# ------------------------------------------------------------------ anchors

# (patch half-size, grid spacing, search radius, minimum score) - coarse first.
LEVELS = ((96, 150, 72, 0.28), (56, 104, 36, 0.30), (34, 78, 16, 0.34))
FIELD_LEVELS = ((56, 104, 36, 0.30), (34, 78, 16, 0.34))
PEAK_RATIO = 1.10
BOX_MASK_SHRINK = 0.16   # box interior masked out, shrunk to keep its border


@lru_cache(maxsize=64)
def anchors(month: int, half: int, cell: int):
    """Masked landmark patches cut from the clean template.

    Two properties matter.  *Distinctiveness*: the diary is a regular grid, so
    a small patch matches almost as well one row off as in the right place -
    large patches spanning several rows do not.  *Immunity to the user*: a
    large patch inevitably covers checkboxes, so each patch carries a mask
    that excludes every box interior, and matching is done through that mask.
    What is compared is therefore only ink the printer put down.

    Patches are high-passed, which makes the (necessarily un-centred) masked
    correlation behave like a proper zero-mean one under uneven lighting.
    """
    clean = T.clean_page(month).astype(np.float32)
    h, w = clean.shape
    model = T.month_model(month)

    # Mask out the box interiors - the only part of the page the *user* writes
    # on - but shrink the masked square enough to leave the printed border
    # itself usable.  Those borders matter: they are the only dense landmarks
    # inside the grid, which is exactly where sampling accuracy has to be
    # highest.  A stroke can still stray onto the surviving rim, so this trades
    # a little contamination for a lot of coverage; the robust fits absorb it.
    usable = np.ones((h, w), np.float32)
    for rects in model["boxes"].values():
        for x0, y0, x1, y1 in rects:
            ix, iy = (x1 - x0) * BOX_MASK_SHRINK, (y1 - y0) * BOX_MASK_SHRINK
            cv2.rectangle(usable, (int(x0 + ix), int(y0 + iy)),
                          (int(x1 - ix), int(y1 - iy)), 0.0, -1)
    for corners in model["markers"].values():
        c = np.asarray(corners, np.float32)
        cv2.rectangle(usable, tuple((c.min(0) - 20).astype(int)),
                      tuple((c.max(0) + 20).astype(int)), 0.0, -1)

    hp = _highpass(clean)
    k = 2 * half + 1
    energy = cv2.boxFilter(hp * hp * usable, -1, (k, k), normalize=True)
    cover = cv2.boxFilter(usable, -1, (k, k), normalize=True)
    energy[cover < 0.30] = 0.0
    energy[:half, :] = energy[-half:, :] = 0.0
    energy[:, :half] = energy[:, -half:] = 0.0

    centres, patches, masks = [], [], []
    for cy in range(half, h - half, cell):
        for cx in range(half, w - half, cell):
            sub = energy[cy:cy + cell, cx:cx + cell]
            if sub.size == 0 or sub.max() <= 0.0:
                continue
            yy, xx = np.unravel_index(int(np.argmax(sub)), sub.shape)
            py, px = cy + yy, cx + xx
            sl = (slice(py - half, py + half + 1), slice(px - half, px + half + 1))
            patch, mask = hp[sl], usable[sl]
            if patch.shape != (k, k):
                continue
            if float((patch * patch * mask).sum()) < 40.0 * mask.sum():
                continue
            centres.append((px, py))
            patches.append(np.ascontiguousarray(patch))
            masks.append(np.ascontiguousarray(mask))
    return np.asarray(centres, np.float32), patches, masks


def match_anchors(warped_hp: np.ndarray, month: int, half: int, cell: int,
                  search: int, min_score: float) -> Tuple[np.ndarray, np.ndarray]:
    """Locate each landmark in the high-passed rectified page."""
    centres, patches, masks = anchors(month, half, cell)
    if len(centres) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    H, W = warped_hp.shape
    src, dst = [], []
    for (px, py), patch, mask in zip(centres.astype(int), patches, masks):
        y0c, x0c = max(0, py - half - search), max(0, px - half - search)
        window = warped_hp[y0c:min(H, py + half + search + 1),
                           x0c:min(W, px + half + search + 1)]
        if window.shape[0] <= patch.shape[0] or window.shape[1] <= patch.shape[1]:
            continue
        try:
            res = cv2.matchTemplate(window, patch, cv2.TM_CCORR_NORMED, mask=mask)
        except cv2.error:
            continue
        res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < min_score or not _distinctive(res, loc, score):
            continue
        fx, fy = _subpixel_peak(res, loc)
        src.append([x0c + fx + half, y0c + fy + half])
        dst.append([float(px), float(py)])
    return np.asarray(src, np.float32).reshape(-1, 2), np.asarray(dst, np.float32).reshape(-1, 2)


def _distinctive(res: np.ndarray, loc, score: float, guard: int = 10) -> bool:
    """True when the correlation peak clearly beats any rival peak.

    The diary's rows are evenly pitched, so a landmark can correlate nearly as
    well one row out of register.  Ambiguous matches are discarded rather than
    guessed - that is what stops a whole page snapping to the wrong row.
    """
    x, y = loc
    masked = res.copy()
    masked[max(0, y - guard):y + guard + 1, max(0, x - guard):x + guard + 1] = -1.0
    rival = float(masked.max())
    if rival <= 0.0:
        return True
    return score >= PEAK_RATIO * rival


def _subpixel_peak(res: np.ndarray, loc) -> Tuple[float, float]:
    x, y = loc
    fx, fy = float(x), float(y)
    if 0 < x < res.shape[1] - 1:
        a, b, c = float(res[y, x - 1]), float(res[y, x]), float(res[y, x + 1])
        d = a - 2 * b + c
        if abs(d) > 1e-9:
            fx += float(np.clip(0.5 * (a - c) / d, -1, 1))
    if 0 < y < res.shape[0] - 1:
        a, b, c = float(res[y - 1, x]), float(res[y, x]), float(res[y + 1, x])
        d = a - 2 * b + c
        if abs(d) > 1e-9:
            fy += float(np.clip(0.5 * (a - c) / d, -1, 1))
    return fx, fy


# ------------------------------------------------------- polynomial residual

def _n_terms(deg: int) -> int:
    return (deg + 1) * (deg + 2) // 2


def _poly_terms(pts: np.ndarray, deg: int, size) -> np.ndarray:
    w, h = size
    x = pts[:, 0] / w * 2.0 - 1.0
    y = pts[:, 1] / h * 2.0 - 1.0
    out = np.empty((len(pts), _n_terms(deg)), dtype=x.dtype)
    out[:, 0] = 1.0
    k = 1
    for total in range(1, deg + 1):
        for i in range(total + 1):
            out[:, k] = (x ** (total - i)) * (y ** i)
            k += 1
    return out


def fit_field(src: np.ndarray, dst: np.ndarray, size, deg: int) -> Optional[np.ndarray]:
    """Robust least squares for ``dst -> src`` (canonical -> warped lookup)."""
    A = _poly_terms(dst, deg, size)
    if len(src) < 3 * A.shape[1]:
        return None
    coef = None
    weights = np.ones(len(src))
    for _ in range(4):
        Aw = A * weights[:, None]
        coef, *_ = np.linalg.lstsq(Aw, src * weights[:, None], rcond=None)
        resid = np.linalg.norm(A @ coef - src, axis=1)
        scale = max(1.0, 1.4826 * float(np.median(resid)))
        weights = 1.0 / (1.0 + (resid / (2.5 * scale)) ** 2)
        if weights.sum() < 3 * A.shape[1]:
            return None
    return coef


FIELD_ROW_BLOCK = 128


def apply_field(warped: np.ndarray, coef: np.ndarray, size, deg: int) -> Optional[np.ndarray]:
    """Resample ``warped`` through the fitted displacement field.

    Evaluated in horizontal bands rather than over the whole page at once.
    Building the design matrix for every pixel in one go needs a
    (width*height x terms) array - a couple of hundred megabytes for an A6 page
    at 300 dpi - which is a real ceiling on a small instance.  Banding costs
    nothing and bounds the working set to one band.
    """
    w, h = size
    mx = np.empty((h, w), np.float32)
    my = np.empty((h, w), np.float32)
    xs = np.arange(w, dtype=np.float32)

    for y0 in range(0, h, FIELD_ROW_BLOCK):
        y1 = min(y0 + FIELD_ROW_BLOCK, h)
        rows = y1 - y0
        band = np.empty((rows * w, 2), np.float32)
        band[:, 0] = np.tile(xs, rows)
        band[:, 1] = np.repeat(np.arange(y0, y1, dtype=np.float32), w)
        mapped = _poly_terms(band, deg, size) @ coef
        if float(np.abs(mapped - band).max()) > MAX_FIELD_SHIFT:
            return None
        mx[y0:y1] = mapped[:, 0].reshape(rows, w)
        my[y0:y1] = mapped[:, 1].reshape(rows, w)

    return cv2.remap(warped, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------- template score

SCORE_SIGMAS = (0.0, 1.5, 3.0, 5.0)


@lru_cache(maxsize=16)
def _score_reference(month: int):
    """Masked, high-passed template at several sharpness levels.

    A photo taken slightly out of focus has genuinely lost its fine detail, so
    correlating it against a razor-sharp reference under-reports the match.
    Scoring against a small family of progressively softened references and
    keeping the best makes the measure a statement about *layout* agreement
    rather than about the camera's focus.
    """
    clean = T.clean_page(month).astype(np.float32)
    model = T.month_model(month)
    mask = np.ones(clean.shape, np.uint8)
    for rects in model["boxes"].values():
        for x0, y0, x1, y1 in rects:
            cv2.rectangle(mask, (int(x0) - 5, int(y0) - 5), (int(x1) + 5, int(y1) + 5), 0, -1)
    for corners in model["markers"].values():
        c = np.asarray(corners, np.float32)
        cv2.rectangle(mask, tuple((c.min(0) - 12).astype(int)),
                      tuple((c.max(0) + 12).astype(int)), 0, -1)
    keep = mask > 0

    refs = []
    for sigma in SCORE_SIGMAS:
        soft = clean if sigma <= 0 else cv2.GaussianBlur(clean, (0, 0), sigma)
        v = _highpass(soft)[keep]
        v = v - v.mean()
        n = float(np.sqrt(float((v * v).sum())))
        refs.append(v / n if n > 1e-6 else v)
    return refs, keep


def _highpass(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), 1.6)
    return f - cv2.GaussianBlur(f, (0, 0), 9.0)


def template_score(warped: np.ndarray, month: int,
                   hp: Optional[np.ndarray] = None) -> float:
    """Illumination- and focus-tolerant correlation with the clean template.

    ``hp`` lets a caller that already has the high-passed page hand it in.
    High-passing is two Gaussian blurs over the whole page, and rectification
    scores a dozen candidates, so recomputing it each time is the single
    largest cost in the pipeline on a CPU-limited machine.
    """
    refs, keep = _score_reference(month)
    b = (_highpass(warped) if hp is None else hp)[keep]
    b = b - b.mean()
    n = float(np.sqrt(float((b * b).sum())))
    if n <= 1e-6:
        return 0.0
    b /= n
    return max(float(ref @ b) for ref in refs)


# ------------------------------------------------------------------ pipeline

@lru_cache(maxsize=4)
def _hann(shape) -> np.ndarray:
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def global_shift(warped: np.ndarray, month: int, limit: float = 110.0):
    """Dominant translation between the rectified page and the template.

    Phase correlation looks at the whole page at once, so unlike a local patch
    search it cannot be fooled by the diary's repeating row pitch.  It gives
    the anchor stage a starting point close enough that a small, unambiguous
    search window suffices.
    """
    clean = T.clean_page(month)
    if warped.shape != clean.shape:
        return None
    win = _hann(clean.shape)
    a = _highpass(clean.astype(np.float32)) * win
    b = _highpass(warped.astype(np.float32)) * win
    try:
        (dx, dy), response = cv2.phaseCorrelate(a, b)
    except cv2.error:
        return None
    if not np.isfinite([dx, dy]).all() or float(np.hypot(dx, dy)) > limit:
        return None
    return float(dx), float(dy), float(response)


class Rectified:
    def __init__(self, image, score, stats):
        self.image = image
        self.score = score
        self.stats = stats


def rectify(gray: np.ndarray, markers: Dict[int, np.ndarray], month: int,
            ids: List[int]) -> Optional[Rectified]:
    """Warp ``gray`` into the canonical frame for ``month``, or return None.

    Every stage is a *proposal*: it is adopted only when it raises the
    correlation with the clean template, so a stage that misfires costs time
    and nothing else.
    """
    H, marker_inliers = marker_homography(markers, month, ids)
    if H is None:
        return None

    size = T.canonical_size()
    W, Hh = size

    def warp(mat):
        return cv2.warpPerspective(gray, mat, (W, Hh), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)

    best = warp(H)
    best_hp = _highpass(best)
    best_score = template_score(best, month, best_hp)
    stats = {"markers": float(len(ids)), "marker_inliers": float(marker_inliers),
             "anchor_rate": 0.0, "anchor_inliers": 0.0, "stage": 0.0}

    def consider(image):
        """High-pass once, score once; returns (image, hp, score)."""
        hp = _highpass(image)
        return image, hp, template_score(image, month, hp)

    # Stage 1: whole-page translation. Phase correlation looks at the entire
    # page at once, so unlike a patch search it cannot land one row out.
    shift = global_shift(best, month)
    if shift is not None:
        dx, dy, _ = shift
        for sign in (-1.0, 1.0):
            Ht = np.array([[1, 0, sign * dx], [0, 1, sign * dy], [0, 0, 1]], np.float64) @ H
            cand, cand_hp, score = consider(warp(Ht))
            if score > best_score + 1e-4:
                H, best, best_hp, best_score = Ht, cand, cand_hp, score
                stats["stage"] = 0.5

    # Stage 2: homography through printed landmarks, coarse patches first.
    for half, cell, search, min_score in LEVELS:
        n_total = max(1, len(anchors(month, half, cell)[0]))
        for _ in range(2):
            src, dst = match_anchors(best_hp, month, half, cell, search, min_score)
            if len(src) < 6:
                break
            Hr, inliers = fit_homography(src, dst, 8.0)
            if Hr is None or inliers < 6:
                break
            cand_H = Hr @ H
            cand, cand_hp, score = consider(warp(cand_H))
            if score <= best_score + 1e-4:
                break
            H, best, best_hp, best_score = cand_H, cand, cand_hp, score
            stats.update(anchor_rate=max(stats["anchor_rate"], inliers / n_total),
                         anchor_inliers=float(inliers), stage=1.0)

    # Stage 3: the non-projective remainder - page curl and lens distortion,
    # which no homography can express.
    for half, cell, search, min_score in FIELD_LEVELS:
        src, dst = match_anchors(best_hp, month, half, cell, search, min_score)
        for deg in (3, 2):
            if len(src) < 3 * _n_terms(deg):
                continue
            coef = fit_field(src, dst, size, deg)
            if coef is None:
                continue
            remapped = apply_field(best, coef, size, deg)
            if remapped is None:
                continue
            cand, cand_hp, score = consider(remapped)
            if score > best_score + 1e-4:
                best, best_hp, best_score = cand, cand_hp, score
                stats["stage"] = 2.0
            break

    half, cell = 34, 78
    src, dst = match_anchors(best_hp, month, half, cell, 14, 0.34)
    n_total = max(1, len(anchors(month, half, cell)[0]))
    stats["anchor_rate"] = max(stats["anchor_rate"], len(src) / n_total)
    stats["residual_px"] = (float(np.median(np.linalg.norm(src - dst, axis=1)))
                            if len(src) else 99.0)
    return Rectified(best, best_score, stats)

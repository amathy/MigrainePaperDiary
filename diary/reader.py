"""Read a photographed migraine-diary month page into structured data.

The pipeline is deliberately classical and verifiable end to end - nothing is
inferred from priors about what a diary "usually" says:

1. **Find the page.**  Detect the ArUco markers printed in the page corners.
   Their ids name the month; their corners register the page.
2. **Rectify.**  See :mod:`diary.geometry` - marker homography, then a
   RANSAC-fitted correction through print-only landmarks, then a robust
   polynomial displacement field for page curl.
3. **Verify.**  A month hypothesis is only accepted if the rectified page
   actually correlates with that month's clean template.  Every hypothesis
   the markers support is tried, and if none verifies the image is rejected
   rather than guessed at.
4. **Read.**  Flat-field the paper, then look inside each checkbox (inset, so
   the printed border and any neighbouring overshoot are excluded) for a
   connected run of ink at a threshold scaled to this page's own contrast.
"""

from __future__ import annotations

import calendar
import datetime as _dt
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import geometry as G
from . import template as T

# --------------------------------------------------------------- parameters

BOX_INSET = 0.20              # fraction of the box trimmed off each edge
INK_FRACTION = 0.26           # mark threshold, as a fraction of printed-ink depth
INK_LEVEL_RANGE = (0.12, 0.22)  # clamp for the adaptive threshold
INK_BLUR = 0.030              # smoothing radius, as a fraction of the box size
MIN_MARK_AREA = 0.004         # smallest connected mark, as a fraction of the box
MIN_TEMPLATE_SCORE = 0.35     # below this the page is not a diary month page
MIN_ANCHOR_RATE = 0.30


@dataclass
class CellReading:
    day: int
    values: Tuple[bool, bool, bool]
    coverage: Tuple[float, float, float]


@dataclass
class Reading:
    ok: bool
    month: Optional[int] = None
    year: Optional[int] = None
    days: List[CellReading] = field(default_factory=list)
    reason: str = ""
    diagnostics: Dict[str, float] = field(default_factory=dict)
    warped: Optional[np.ndarray] = None


# ----------------------------------------------------------------- loading

def _register_heif() -> None:
    """Teach Pillow to open HEIC/HEIF, if the optional plugin is installed.

    iPhones photograph in HEIC by default.  Safari usually transcodes to JPEG
    on upload, but not always, and a diary page that cannot even be decoded is
    a poor way to greet someone holding their phone over it.
    """
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:
        pass


MAX_INPUT_LONG_SIDE = 3000


def _bound_size(img: np.ndarray) -> np.ndarray:
    """Cap the working resolution.

    A modern phone shoots 12 MP or more, and every intermediate copy of such a
    frame is tens of megabytes - enough to matter on a small instance.  The
    page is rectified to a fixed 300 dpi canonical frame anyway, and marker
    decoding is comfortable well below this cap, so nothing is gained by
    carrying the full sensor resolution through the pipeline.
    """
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_INPUT_LONG_SIDE:
        return img
    scale = MAX_INPUT_LONG_SIDE / long_side
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def load_image(path: str) -> np.ndarray:
    """Read an image, honouring the EXIF orientation phones write."""
    try:
        from PIL import Image, ImageOps

        _register_heif()
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return _bound_size(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        return _bound_size(img)


# --------------------------------------------------------------- box reading

def _flat_field(gray: np.ndarray) -> np.ndarray:
    """Divide out the paper illumination; result is ~1.0 on blank paper."""
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 12.0)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(gray.astype(np.float32) / np.maximum(bg, 20.0), 0.0, 2.0)


@lru_cache(maxsize=16)
def _printed_mask(month: int) -> np.ndarray:
    """Where the *printer* put ink: borders, day numbers, headers, footer."""
    clean = T.clean_page(month)
    mask = clean < 205
    model = T.month_model(month)
    block = np.zeros(clean.shape, np.uint8)
    for rects in model["boxes"].values():
        for x0, y0, x1, y1 in rects:
            ix, iy = (x1 - x0) * 0.10, (y1 - y0) * 0.10
            cv2.rectangle(block, (int(x0 + ix), int(y0 + iy)),
                          (int(x1 - ix), int(y1 - iy)), 1, -1)
    for corners in model["markers"].values():
        c = np.asarray(corners, np.float32)
        cv2.rectangle(block, tuple((c.min(0) - 8).astype(int)),
                      tuple((c.max(0) + 8).astype(int)), 1, -1)
    return mask & (block == 0)


def ink_level(norm: np.ndarray, month: int) -> float:
    """Threshold for "this is a pen mark", scaled to the page's own contrast.

    A flat, washed-out photo renders the printed grey of the diary itself only
    a fraction below paper white, and a pencil mark on such a page is fainter
    still.  Measuring how dark this page's *known printed* ink came out gives a
    contrast reference that is specific to the photo, so the same threshold
    works for a crisp scan and for a dim phone snap.
    """
    ink = np.clip(1.0 - norm, 0.0, 1.0)
    sample = ink[_printed_mask(month)]
    if sample.size < 100:
        return INK_LEVEL_RANGE[0]
    printed = float(np.quantile(sample, 0.85))
    return float(np.clip(INK_FRACTION * printed, *INK_LEVEL_RANGE))


def _measure(norm: np.ndarray, rect, level: float,
             inset: float = BOX_INSET) -> Tuple[bool, float]:
    """Decide whether one checkbox carries a mark, and how much ink it holds.

    Only the *interior* of the box is looked at, inset far enough that neither
    the printed border nor a neighbour's overshooting stroke can reach it.
    The decision rests on the largest **connected** run of ink rather than on
    total coverage: a pen stroke - even a thin, faint, pencil one - is a
    continuous line tens of pixels long, whereas dust, sensor noise and JPEG
    ringing are isolated specks.  Asking for connectivity buys sensitivity to
    faint marks without buying sensitivity to grit.
    """
    x0, y0, x1, y1 = rect
    dx, dy = (x1 - x0) * inset, (y1 - y0) * inset
    a, b = max(int(round(x0 + dx)), 0), max(int(round(y0 + dy)), 0)
    c = min(int(round(x1 - dx)), norm.shape[1])
    d = min(int(round(y1 - dy)), norm.shape[0])
    if c - a < 6 or d - b < 6:
        return False, 0.0

    ink = np.clip(1.0 - norm[b:d, a:c], 0.0, 1.0)
    sigma = INK_BLUR * max(x1 - x0, y1 - y0)
    if sigma > 0.3:
        ink = cv2.GaussianBlur(ink, (0, 0), sigma)
    binary = (ink > level).astype(np.uint8)
    if not binary.any():
        return False, 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    return largest >= MIN_MARK_AREA * binary.size, float(binary.mean())


# ------------------------------------------------------------------ pipeline

def read(path_or_image, want_debug: bool = False,
         year: Optional[int] = None) -> Reading:
    img = load_image(path_or_image) if isinstance(path_or_image, str) else path_or_image
    if img is None:
        return Reading(False, reason="the file could not be decoded as an image")
    img = _bound_size(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    markers = G.detect_markers(gray)
    if not markers:
        return Reading(False, reason="no diary registration markers were found in the image")

    best: Optional[Tuple] = None
    for month, ids in G.month_hypotheses(markers.keys()):
        rect = G.rectify(gray, markers, month, ids)
        if rect is None:
            continue
        if best is None or rect.score > best[0].score:
            best = (rect, month)
        if rect.score >= MIN_TEMPLATE_SCORE:
            break

    if best is None:
        return Reading(False, reason="the page geometry could not be fitted to the diary template")

    rect, month = best
    diagnostics = dict(rect.stats, template_score=float(rect.score), month=float(month))

    if rect.score < MIN_TEMPLATE_SCORE:
        return Reading(False, month=month, diagnostics=diagnostics,
                       warped=rect.image if want_debug else None,
                       reason="this does not look like a migraine diary month page "
                              f"(template match {rect.score:.2f})")
    if rect.stats.get("anchor_rate", 0.0) < MIN_ANCHOR_RATE:
        return Reading(False, month=month, diagnostics=diagnostics,
                       warped=rect.image if want_debug else None,
                       reason="too little of the diary layout is visible "
                              f"({rect.stats.get('anchor_rate', 0):.0%} of landmarks found)")

    model = T.month_model(month)
    norm = _flat_field(rect.image)
    level = ink_level(norm, month)
    diagnostics["ink_level"] = level
    # The February page prints 29 rows; how many of them are real days depends
    # on the year, so the year has to be settled before the rows are emitted.
    year = year or _dt.date.today().year
    ndays = calendar.monthrange(year, month)[1]

    days = []
    for day in range(1, ndays + 1):
        rects = model["boxes"].get(str(day))
        if rects is None:
            continue
        measured = [_measure(norm, r, level) for r in rects]
        days.append(CellReading(day, tuple(m[0] for m in measured),
                                tuple(m[1] for m in measured)))

    return Reading(True, month=month, year=year, days=days, diagnostics=diagnostics,
                   warped=rect.image if want_debug else None)


def to_rows(reading: Reading, year: Optional[int] = None) -> List[List[str]]:
    year = year or reading.year or _dt.date.today().year
    return [[_dt.date(year, reading.month, c.day).isoformat()] +
            ["yes" if v else "no" for v in c.values] for c in reading.days]

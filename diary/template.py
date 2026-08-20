"""Build and load the canonical diary template model.

The template model is derived directly from the source PDF, so the geometry
used by the reader is exactly the geometry that was printed.  Everything is
expressed in a *canonical page frame*: the month page rendered at
``CANONICAL_DPI``, origin top-left, x right, y down.

Key facts about the diary (all verified from the PDF):

* Month pages are PDF pages 4, 6, 8, ... 26 (0-based).
* Each month page carries four ArUco markers (``DICT_4X4_50``) in the page
  corners.  Their ids encode both the month and which corner they are::

      month  = id // 4 + 1          (1..12)
      corner = id %  4              (0=TL, 1=TR, 2=BL, 3=BR)

* Checkboxes sit on a fixed grid: a left column group holding days 1..16 and
  a right column group holding days 17..31, each row being ``M``, ``H``,
  ``Rx`` left to right.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import cv2
import numpy as np

CANONICAL_DPI = 300
_PT_PER_INCH = 72.0

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "template_model.json")
RENDER_DIR = os.path.join(HERE, "template_pages")

ARUCO_DICT = cv2.aruco.DICT_4X4_50
FIRST_MONTH_PAGE = 4  # 0-based index of the January page
PAGE_STRIDE = 2  # notes page + month page

COLUMNS = ("migraine", "headache", "medication")


def _aruco_detector() -> "cv2.aruco.ArucoDetector":
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement matters: a half-pixel bias on a corner
    # marker turns into several pixels of drift at the far side of the page.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 60
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(dictionary, params)


def month_page_index(month: int) -> int:
    return FIRST_MONTH_PAGE + PAGE_STRIDE * (month - 1)


def build_model(pdf_path: str, write: bool = True) -> dict:
    """Derive the template model (and clean page renders) from the PDF."""
    import pymupdf

    doc = pymupdf.open(pdf_path)
    detector = _aruco_detector()
    scale = CANONICAL_DPI / _PT_PER_INCH

    os.makedirs(RENDER_DIR, exist_ok=True)
    months: Dict[str, dict] = {}
    size = None

    for month in range(1, 13):
        page = doc[month_page_index(month)]
        pix = page.get_pixmap(dpi=CANONICAL_DPI)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
        size = (pix.width, pix.height)

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) != 4:
            raise RuntimeError(f"expected 4 markers on month {month}, got {ids}")
        markers = {int(i): np.asarray(c[0], float).tolist() for i, c in zip(ids.ravel(), corners)}
        for marker_id in markers:
            if marker_id // 4 + 1 != month:
                raise RuntimeError(f"marker {marker_id} does not belong to month {month}")

        boxes = _extract_boxes(page, scale)
        months[str(month)] = {
            "pdf_page": month_page_index(month),
            "markers": {str(k): v for k, v in sorted(markers.items())},
            "boxes": {str(d): boxes[d] for d in sorted(boxes)},
        }
        cv2.imwrite(os.path.join(RENDER_DIR, f"month_{month:02d}.png"), gray)

    model = {
        "dpi": CANONICAL_DPI,
        "canonical_size": list(size),
        "aruco_dict": "DICT_4X4_50",
        "columns": list(COLUMNS),
        "months": months,
    }
    if write:
        with open(MODEL_PATH, "w") as fh:
            json.dump(model, fh, indent=1)
    return model


def _extract_boxes(page, scale: float) -> Dict[int, List[List[float]]]:
    """Pull the checkbox rectangles out of the page's vector drawings."""
    rects = [
        d["rect"]
        for d in page.get_drawings()
        if d["type"] == "s" and 12.0 < d["rect"].width < 15.0 and 12.0 < d["rect"].height < 15.0
    ]
    left = sorted([r for r in rects if r.x0 < 150], key=lambda r: (r.y0, r.x0))
    right = sorted([r for r in rects if r.x0 >= 150], key=lambda r: (r.y0, r.x0))
    if len(left) % 3 or len(right) % 3:
        raise RuntimeError("checkbox count is not a multiple of three")

    boxes: Dict[int, List[List[float]]] = {}
    for group, first_day in ((left, 1), (right, 17)):
        for row in range(len(group) // 3):
            trio = sorted(group[row * 3:row * 3 + 3], key=lambda r: r.x0)
            boxes[first_day + row] = [
                [r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale] for r in trio
            ]
    return boxes


_MODEL_CACHE: dict | None = None


def load_model() -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        with open(MODEL_PATH) as fh:
            _MODEL_CACHE = json.load(fh)
    return _MODEL_CACHE


def month_model(month: int) -> dict:
    return load_model()["months"][str(month)]


def clean_page(month: int) -> np.ndarray:
    img = cv2.imread(os.path.join(RENDER_DIR, f"month_{month:02d}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"missing clean render for month {month}")
    return img


def canonical_size() -> tuple:
    w, h = load_model()["canonical_size"]
    return int(w), int(h)


def days_in_month(month: int, year: int) -> int:
    import calendar

    return calendar.monthrange(year, month)[1]


if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "Template/Headache diary with ArUco markersA6_nice.pdf"
    m = build_model(pdf)
    print(f"wrote {MODEL_PATH}: {len(m['months'])} months, canonical {m['canonical_size']}")

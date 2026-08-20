#!/usr/bin/env python3
"""Build a set of images that ReadMigraineDiary is *supposed* to refuse.

The interesting cases are near-misses, not obvious ones: other pages of the
same diary, a month page with its registration markers cropped away, a page
photographed so badly that the reading would be a guess.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diary import synth  # noqa: E402

PDF = "Template/Headache diary with ArUco markersA6_nice.pdf"


def _page(pdf, index, dpi=300):
    import pymupdf

    doc = pymupdf.open(pdf)
    pix = doc[index].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n >= 3 else cv2.COLOR_GRAY2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF)
    ap.add_argument("--out", default="Training/negatives")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(4242)
    n = 0

    def save(name, img, quality=85):
        nonlocal n
        cv2.imwrite(os.path.join(args.out, name + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        n += 1

    # Other pages of the same diary - these carry no month markers.
    for label, index in [("cover", 0), ("belongs_to", 1), ("how_to_use", 2),
                         ("notes_january", 3), ("notes_blank", 27)]:
        page = _page(args.pdf, index)
        save(f"other_page_{label}", synth.photograph(page, rng, "normal"))

    # A month page with the marker corners cropped off.
    for month in (1, 7):
        truth = synth.sample_truth(month, 2026, rng)
        page = synth.render_filled_page(month, truth, rng)
        photo = synth.photograph(page, np.random.default_rng(11 + month), "easy")
        h, w = photo.shape[:2]
        save(f"cropped_month{month:02d}", photo[int(h * 0.30):int(h * 0.72),
                                                int(w * 0.22):int(w * 0.80)])

    # A month page photographed far too badly to read honestly.
    truth = synth.sample_truth(3, 2026, rng)
    page = synth.render_filled_page(3, truth, rng)
    photo = synth.photograph(page, np.random.default_rng(5), "hard")
    tiny = cv2.resize(photo, None, fx=0.16, fy=0.16, interpolation=cv2.INTER_AREA)
    save("unreadably_small_month03", cv2.GaussianBlur(tiny, (0, 0), 1.4), 25)

    # Things that are not a diary at all.
    save("blank_paper", np.full((1400, 1000, 3), 238, np.uint8))
    noise = rng.integers(0, 255, (1200, 1600, 3), dtype=np.uint8)
    save("random_noise", noise)
    text = np.full((1600, 1200, 3), 245, np.uint8)
    for i in range(28):
        cv2.putText(text, "the quick brown fox jumps over the lazy dog",
                    (70, 90 + i * 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    save("printed_text_page", synth.photograph(text, rng, "easy"))
    grid = np.full((1500, 1100, 3), 250, np.uint8)
    for r in range(20):
        for c in range(4):
            cv2.rectangle(grid, (150 + c * 180, 120 + r * 66),
                          (210 + c * 180, 180 + r * 66), (90, 90, 90), 3)
    save("unrelated_checkbox_form", synth.photograph(grid, rng, "easy"))

    print(f"wrote {n} negative images to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

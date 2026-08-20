#!/usr/bin/env python3
"""Generate the synthetic training/validation set.

For each of the 12 month pages in the template PDF, render ``--per-page``
filled-in variants and photograph each one under different conditions.
Images land in ``Training/images`` and the matching ground truth in
``Training/ground_truth`` (same stem, ``.csv``), in exactly the format
ReadMigraineDiary emits.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diary import synth  # noqa: E402

DIFFICULTY_MIX = ["easy"] * 8 + ["normal"] * 16 + ["hard"] * 6


def write_truth(path: str, month: int, year: int, truth) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "migraine", "headache", "medication"])
        for day in sorted(truth):
            w.writerow([_dt.date(year, month, day).isoformat()] +
                       ["yes" if v else "no" for v in truth[day]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-page", type=int, default=30)
    ap.add_argument("--out", default="Training")
    ap.add_argument("--year", type=int, default=_dt.date.today().year)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    gt_dir = os.path.join(args.out, "ground_truth")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    jobs = [(month, k, args.seed, args.year, img_dir, gt_dir)
            for month in range(1, 13) for k in range(args.per_page)]

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for stem in ex.map(_make_one, jobs, chunksize=4):
            done += 1
            if done % 30 == 0:
                print(f"  {done}/{len(jobs)} images", flush=True)

    print(f"wrote {done} images to {img_dir} and ground truth to {gt_dir}")
    return 0


def _make_one(job):
    month, k, seed, year, img_dir, gt_dir = job
    rng = np.random.default_rng(seed + month * 1000 + k)
    difficulty = DIFFICULTY_MIX[k % len(DIFFICULTY_MIX)]
    truth = synth.sample_truth(month, year, rng)
    page = synth.render_filled_page(month, truth, rng)
    photo = synth.photograph(page, rng, difficulty)

    stem = f"month{month:02d}_{k:02d}_{difficulty}"
    quality = int(rng.integers(*{"easy": (82, 97), "normal": (55, 95),
                                 "hard": (28, 75)}[difficulty]))
    cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), photo,
                [cv2.IMWRITE_JPEG_QUALITY, quality])
    write_truth(os.path.join(gt_dir, stem + ".csv"), month, year, truth)
    return stem


if __name__ == "__main__":
    raise SystemExit(main())

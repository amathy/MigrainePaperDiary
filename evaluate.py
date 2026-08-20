#!/usr/bin/env python3
"""Score ReadMigraineDiary against the generated ground truth."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if rows and rows[0][0].strip().lower() == "date":
        rows = rows[1:]
    return {r[0]: tuple(v.strip().lower() for v in r[1:4]) for r in rows if r}


def score_one(args):
    img, gt_path = args
    from diary import reader

    truth = read_csv(gt_path)
    try:
        result = reader.read(img)
    except Exception as exc:
        return img, dict(rejected=True, reason=f"error: {exc}", cells=3 * len(truth),
                         wrong=3 * len(truth), rows=len(truth), rows_wrong=len(truth))

    if not result.ok:
        return img, dict(rejected=True, reason=result.reason, cells=3 * len(truth),
                         wrong=3 * len(truth), rows=len(truth), rows_wrong=len(truth))

    import datetime as _dt
    year = int(next(iter(truth)).split("-")[0]) if truth else _dt.date.today().year
    pred = {_dt.date(year, result.month, c.day).isoformat():
            tuple("yes" if v else "no" for v in c.values) for c in result.days}

    wrong = 0
    rows_wrong = 0
    detail = []
    for date, exp in truth.items():
        got = pred.get(date)
        if got is None:
            wrong += 3
            rows_wrong += 1
            continue
        bad = sum(1 for a, b in zip(exp, got) if a != b)
        if bad:
            rows_wrong += 1
            detail.append((date, exp, got, [c.coverage for c in result.days
                                            if c.day == int(date[-2:])][0]))
        wrong += bad
    extra = set(pred) - set(truth)
    return img, dict(rejected=False, cells=3 * len(truth), wrong=wrong, rows=len(truth),
                     rows_wrong=rows_wrong, extra=len(extra), detail=detail,
                     diag=result.diagnostics)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="Training")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--show", type=int, default=12, help="how many failures to list")
    args = ap.parse_args()

    images = sorted(glob.glob(os.path.join(args.dir, "images", "*.jpg")))
    if args.limit:
        images = images[::max(1, len(images) // args.limit)][:args.limit]
    jobs = [(im, os.path.join(args.dir, "ground_truth",
                              os.path.splitext(os.path.basename(im))[0] + ".csv"))
            for im in images]
    jobs = [j for j in jobs if os.path.exists(j[1])]

    cells = wrong = rows = rows_wrong = rejected = 0
    failures = []
    per_image = []
    reasons = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for img, r in ex.map(score_one, jobs, chunksize=2):
            cells += r["cells"]
            wrong += r["wrong"]
            rows += r["rows"]
            rows_wrong += r["rows_wrong"]
            per_image.append(((r["cells"] - r["wrong"]) / max(1, r["cells"]), img))
            if r["rejected"]:
                rejected += 1
                reasons[r["reason"][:70]] += 1
                failures.append((r["wrong"], img, "REJECTED: " + r["reason"], []))
            elif r["wrong"]:
                failures.append((r["wrong"], img, f"{r['wrong']} wrong cells", r["detail"]))

    print(f"\nimages          : {len(jobs)}")
    print(f"rejected        : {rejected}")
    print(f"cell accuracy   : {(cells - wrong) / max(1, cells):.4%}  ({cells - wrong}/{cells})")
    print(f"row accuracy    : {(rows - rows_wrong) / max(1, rows):.4%}  ({rows - rows_wrong}/{rows})")
    perfect = len(jobs) - len(failures)
    print(f"perfect images  : {perfect}/{len(jobs)}")
    per_image.sort()
    above = sum(1 for a, _ in per_image if a > 0.95)
    print(f"images >95% acc : {above}/{len(per_image)}")
    if per_image:
        worst = per_image[0]
        print(f"worst image     : {worst[0]:.2%}  ({os.path.basename(worst[1])})")
        accepted = [a for a, _ in per_image if a > 0.0]
        if accepted:
            print(f"worst accepted  : {min(accepted):.2%}")
    if reasons:
        print("\nrejection reasons:")
        for k, v in reasons.most_common():
            print(f"  {v:4d}  {k}")
    if failures:
        print(f"\nworst {min(args.show, len(failures))} images:")
        for w, img, msg, detail in sorted(failures, reverse=True)[:args.show]:
            print(f"  {os.path.basename(img)}: {msg}")
            for d in detail[:4]:
                print(f"      {d[0]} expected {d[1]} got {d[2]} coverage "
                      f"{tuple(round(x, 3) for x in d[3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

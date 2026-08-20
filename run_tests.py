#!/usr/bin/env python3
"""Self-contained checks for ReadMigraineDiary.

Covers the template model, the reading pipeline end to end, the rejection
behaviour that keeps the tool from inventing data, and the CLI contract.

    python run_tests.py
"""

from __future__ import annotations

import csv
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from diary import reader, synth, template as T  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail and not condition else ""))


# --------------------------------------------------------------- the template

def test_template():
    print("template model")
    model = T.load_model()
    check("twelve month pages", len(model["months"]) == 12)
    check("canonical page size", tuple(model["canonical_size"]) == (1242, 1750))

    ok_ids = ok_boxes = True
    for month in range(1, 13):
        m = T.month_model(month)
        ids = sorted(int(k) for k in m["markers"])
        if ids != [4 * (month - 1) + i for i in range(4)]:
            ok_ids = False
        days = sorted(int(k) for k in m["boxes"])
        expected = 29 if month == 2 else (30 if month in (4, 6, 9, 11) else 31)
        if days != list(range(1, expected + 1)) or any(len(v) != 3 for v in m["boxes"].values()):
            ok_boxes = False
    check("marker ids encode the month", ok_ids)
    check("every day has three checkboxes", ok_boxes)

    boxes = T.month_model(1)["boxes"]
    w = boxes["1"][0][2] - boxes["1"][0][0]
    check("checkbox is ~56px in the canonical frame", 54 < w < 58, f"{w:.1f}")


# ------------------------------------------------------------------- reading

def _round_trip(month, seed, difficulty):
    rng = np.random.default_rng(seed)
    truth = synth.sample_truth(month, 2026, rng)
    page = synth.render_filled_page(month, truth, rng)
    photo = synth.photograph(page, rng, difficulty)
    result = reader.read(photo)
    return truth, result


def test_reading():
    print("reading a synthesised page")
    for month, seed, difficulty in [(1, 101, "easy"), (2, 202, "normal"),
                                    (7, 303, "normal"), (12, 404, "hard")]:
        truth, result = _round_trip(month, seed, difficulty)
        if not result.ok:
            check(f"month {month:02d} ({difficulty}) is read", False, result.reason)
            continue
        check(f"month {month:02d} ({difficulty}) is read", True)
        check(f"month {month:02d} identified from the markers", result.month == month)
        wrong = sum(1 for c in result.days
                    for a, b in zip(truth[c.day], c.values) if a != b)
        total = 3 * len(result.days)
        check(f"month {month:02d} cells agree with ground truth",
              wrong <= 0.05 * total, f"{wrong}/{total} wrong")


def test_orientation():
    print("orientation independence")
    import cv2

    rng = np.random.default_rng(77)
    truth = synth.sample_truth(5, 2026, rng)
    page = synth.render_filled_page(5, truth, rng)
    photo = synth.photograph(page, np.random.default_rng(78), "easy")
    base = reader.read(photo)
    check("upright page is read", base.ok, base.reason)
    for name, rot in [("180 degrees", cv2.ROTATE_180),
                      ("90 degrees", cv2.ROTATE_90_CLOCKWISE)]:
        r = reader.read(cv2.rotate(photo, rot))
        same = r.ok and base.ok and [c.values for c in r.days] == [c.values for c in base.days]
        check(f"rotated {name} gives the same answer", same,
              r.reason if not r.ok else "values differ")


def test_calendar():
    print("calendar handling")
    rng = np.random.default_rng(9)
    truth = synth.sample_truth(2, 2026, rng)
    page = synth.render_filled_page(2, truth, rng)
    r = reader.read(synth.photograph(page, np.random.default_rng(10), "easy"))
    check("February is read", r.ok, r.reason)
    if r.ok:
        import datetime

        n = 29 if datetime.date.today().year % 4 == 0 else 28
        check(f"February emits {n} rows for the current year", len(r.days) == n,
              f"{len(r.days)} rows")
        leap = reader.read(synth.photograph(page, np.random.default_rng(10), "easy"),
                           year=2024)
        rows = reader.to_rows(leap)
        check("leap year February emits 29 rows", leap.ok and len(rows) == 29,
              f"{len(rows)} rows")
        check("leap year rows are dated 2024",
              bool(rows) and rows[0][0].startswith("2024-02-01"))


# ----------------------------------------------------------------- rejection

def test_rejection():
    print("rejecting things that are not a diary month page")
    negatives = sorted(glob.glob(os.path.join(ROOT, "Training", "negatives", "*.jpg")))
    if not negatives:
        check("negative set exists (run make_negatives.py)", False)
        return
    for path in negatives:
        r = reader.read(path)
        check(f"rejects {os.path.basename(path)}", not r.ok,
              f"accepted as month {r.month}")


# ---------------------------------------------------------------------- CLI

def test_cli():
    print("command line interface")
    images = sorted(glob.glob(os.path.join(ROOT, "Training", "images", "*.jpg")))
    if not images:
        check("training images exist (run generate_training_data.py)", False)
        return
    src = images[0]
    stem = os.path.splitext(os.path.basename(src))[0]
    with tempfile.TemporaryDirectory() as tmp:
        copy = os.path.join(tmp, stem + ".jpg")
        with open(src, "rb") as a, open(copy, "wb") as b:
            b.write(a.read())

        proc = subprocess.run([sys.executable, os.path.join(ROOT, "ReadMigraineDiary"),
                               copy], capture_output=True, text=True)
        check("exits 0 on a valid page", proc.returncode == 0, proc.stderr.strip())
        check("prints a summary line", "month" in proc.stdout and not proc.stderr.strip(),
              proc.stderr.strip() or proc.stdout.strip())

        proc = subprocess.run([sys.executable, os.path.join(ROOT, "ReadMigraineDiary"),
                               copy, "--quiet"], capture_output=True, text=True)
        check("--quiet prints nothing", proc.returncode == 0 and not proc.stdout.strip(),
              proc.stdout.strip())

        out = os.path.join(tmp, stem + ".csv")
        check("writes CSV beside the image with the same stem", os.path.exists(out))
        if os.path.exists(out):
            rows = list(csv.reader(open(out)))
            check("CSV header is date,migraine,headache,medication",
                  rows[0] == ["date", "migraine", "headache", "medication"], str(rows[0]))
            body = rows[1:]
            gt = os.path.join(ROOT, "Training", "ground_truth", stem + ".csv")
            expected = len(list(csv.reader(open(gt)))) - 1
            check("one row per day of the month", len(body) == expected,
                  f"{len(body)} vs {expected}")
            check("four columns per row", all(len(r) == 4 for r in body))
            check("yes/no values only",
                  all(v in ("yes", "no") for r in body for v in r[1:]))
            check("dates are ordered ISO dates",
                  [r[0] for r in body] == sorted(r[0] for r in body))

        bad = sorted(glob.glob(os.path.join(ROOT, "Training", "negatives", "*.jpg")))
        if bad:
            target = os.path.join(tmp, "reject.jpg")
            with open(bad[0], "rb") as a, open(target, "wb") as b:
                b.write(a.read())
            proc = subprocess.run([sys.executable, os.path.join(ROOT, "ReadMigraineDiary"),
                                   target], capture_output=True, text=True)
            check("exits 1 on a rejected image", proc.returncode == 1)
            check("writes no CSV for a rejected image",
                  not os.path.exists(os.path.join(tmp, "reject.csv")))

        proc = subprocess.run([sys.executable, os.path.join(ROOT, "ReadMigraineDiary"),
                               os.path.join(tmp, "nope.jpg")], capture_output=True, text=True)
        check("exits 2 on a missing file", proc.returncode == 2)


# -------------------------------------------------------------------- webapp

def test_webapp():
    print("web app")
    try:
        import flask
        del flask
    except ImportError:
        check("Flask is installed (pip install -r requirements.txt)", False)
        return

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DIARY_UPLOAD_DIR"] = tmp
        from webapp import app as webapp_module

        client = webapp_module.create_app().test_client()

        page = client.get("/")
        check("home page renders", page.status_code == 200)
        html = page.get_data(as_text=True)
        check("home page offers a camera capture input", 'capture="environment"' in html)
        check("home page offers a file input", 'type="file"' in html)
        check("health check responds", client.get("/healthz").status_code == 200)

        images = sorted(glob.glob(os.path.join(ROOT, "Training", "images", "*_easy.jpg")))
        if not images:
            check("training images exist (run generate_training_data.py)", False)
            return
        source = images[0]
        stem = os.path.splitext(os.path.basename(source))[0]

        with open(source, "rb") as fh:
            posted = client.post("/upload", data={"page": (fh, "page.jpg")},
                                 content_type="multipart/form-data")
        check("a valid page redirects to a result", posted.status_code == 302,
              str(posted.status_code))
        if posted.status_code != 302:
            return

        location = posted.headers["Location"]
        result = client.get(location)
        check("result page renders", result.status_code == 200)
        body = result.get_data(as_text=True)
        check("result page names the month", "2026" in body, "no month heading")
        check("result page links the CSV download", "diary.csv" in body)

        served = client.get(location.rstrip("/") + "/diary.csv")
        check("CSV downloads as an attachment",
              served.status_code == 200
              and "attachment" in served.headers.get("Content-Disposition", ""))
        got = served.get_data(as_text=True).strip().splitlines()
        expected = open(os.path.join(ROOT, "Training", "ground_truth",
                                     stem + ".csv")).read().strip().splitlines()
        check("downloaded CSV matches the ground truth", got == expected,
              f"{sum(1 for a, b in zip(got, expected) if a != b)} rows differ")

        negatives = sorted(glob.glob(os.path.join(ROOT, "Training", "negatives", "*.jpg")))
        if negatives:
            with open(negatives[0], "rb") as fh:
                bad = client.post("/upload", data={"page": (fh, "page.jpg")},
                                  content_type="multipart/form-data")
            check("a non-diary image is refused", bad.status_code == 422,
                  str(bad.status_code))
            check("the refusal says the image does not fit the template",
                  "doesn't fit the diary template" in bad.get_data(as_text=True))

        with open(source, "rb") as fh:
            wrong = client.post("/upload", data={"page": (fh, "notes.txt")},
                                content_type="multipart/form-data")
        check("a non-image file type is refused", wrong.status_code == 400)
        check("posting no file is refused",
              client.post("/upload", data={}, content_type="multipart/form-data")
              .status_code == 400)

        for token in ("../../etc/passwd", "nope", "f" * 32):
            if client.get(f"/result/{token}").status_code != 404:
                check(f"unknown token {token!r} is a 404", False)
                break
        else:
            check("unknown and malformed result tokens 404", True)

        stale = os.path.join(tmp, "a" * 32)
        os.makedirs(stale, exist_ok=True)
        os.utime(stale, (0, 0))
        webapp_module.prune_uploads(tmp, ttl=3600)
        check("stale uploads are pruned", not os.path.exists(stale))
        keeper = os.path.join(tmp, "b" * 32)
        os.makedirs(keeper, exist_ok=True)
        webapp_module.prune_uploads(tmp, ttl=3600)
        check("fresh uploads are kept", os.path.exists(keeper))

    os.environ.pop("DIARY_UPLOAD_DIR", None)


def main() -> int:
    for fn in (test_template, test_reading, test_orientation, test_calendar,
               test_rejection, test_cli, test_webapp):
        fn()
        print()
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for name in FAIL:
        print(f"  failed: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

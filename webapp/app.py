"""A small Flask front end for ReadMigraineDiary.

The flow is deliberately linear: photograph or upload a diary month page, the
image is stored, ``ReadMigraineDiary`` is run over it, and the resulting CSV is
shown as a table with a download link.  If the reader refuses the image, the
refusal is shown as-is - the point of the reader's rejection path is that a
page it cannot register honestly produces no data, and the web app must not
paper over that.

The reader is invoked as the actual command-line program rather than imported.
That keeps the web app using exactly the tool that was measured, and isolates
the worker from anything OpenCV might do to a malformed image.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
import time
import uuid

from flask import (Flask, abort, redirect, render_template, request,
                   send_file, url_for)
from werkzeug.utils import secure_filename

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READER = os.path.join(ROOT, "ReadMigraineDiary")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp",
                      ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
READ_TIMEOUT_SECONDS = 180
TOKEN_RE = re.compile(r"\A[0-9a-f]{32}\Z")

COLUMN_LABELS = (("migraine", "M", "Migraine"),
                 ("headache", "H", "Headache"),
                 ("medication", "Rx", "Medication"))


def _upload_root() -> str:
    return os.environ.get("DIARY_UPLOAD_DIR", os.path.join(ROOT, "uploads"))


def _upload_ttl_seconds() -> float:
    try:
        return float(os.environ.get("DIARY_UPLOAD_TTL_HOURS", "6")) * 3600.0
    except ValueError:
        return 6 * 3600.0


def prune_uploads(root: str, ttl: float) -> int:
    """Delete stored pages older than the retention window.

    These are photographs of somebody's health record.  They are kept only
    long enough to show the result and let it be downloaded.
    """
    if ttl <= 0 or not os.path.isdir(root):
        return 0
    cutoff = time.time() - ttl
    removed = 0
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if not TOKEN_RE.match(name) or not os.path.isdir(folder):
            continue
        try:
            if os.path.getmtime(folder) < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["UPLOAD_ROOT"] = _upload_root()
    app.config["UPLOAD_TTL"] = _upload_ttl_seconds()
    os.makedirs(app.config["UPLOAD_ROOT"], exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}, 200

    @app.post("/upload")
    def upload():
        upload = request.files.get("page")
        if upload is None or not upload.filename:
            return render_template("index.html",
                                   error="Choose a photo of a diary page first."), 400

        name = secure_filename(upload.filename) or "page.jpg"
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return render_template(
                "index.html",
                error=f"{ext or 'That file type'} is not an image this app can open. "
                      "Use a JPEG, PNG or HEIC photo."), 400

        prune_uploads(app.config["UPLOAD_ROOT"], app.config["UPLOAD_TTL"])

        token = uuid.uuid4().hex
        folder = os.path.join(app.config["UPLOAD_ROOT"], token)
        os.makedirs(folder, exist_ok=True)
        image_path = os.path.join(folder, "page" + ext)
        upload.save(image_path)

        outcome = run_reader(image_path)
        if not outcome["ok"]:
            return render_template("rejected.html", reason=outcome["reason"],
                                   image_url=url_for("page_image", token=token)), 422
        return redirect(url_for("result", token=token))

    @app.get("/result/<token>")
    def result(token):
        folder = _folder(app, token)
        csv_path = os.path.join(folder, "page.csv")
        if not os.path.exists(csv_path):
            abort(404)
        rows = read_rows(csv_path)
        return render_template("result.html", token=token, rows=rows,
                               summary=summarise(rows),
                               month_label=month_label(rows),
                               columns=COLUMN_LABELS,
                               image_url=url_for("page_image", token=token))

    @app.get("/result/<token>/diary.csv")
    def download(token):
        csv_path = os.path.join(_folder(app, token), "page.csv")
        if not os.path.exists(csv_path):
            abort(404)
        rows = read_rows(csv_path)
        stem = (month_label(rows) or "migraine-diary").lower().replace(" ", "-")
        return send_file(csv_path, mimetype="text/csv", as_attachment=True,
                         download_name=f"{stem}.csv")

    @app.get("/result/<token>/page")
    def page_image(token):
        folder = _folder(app, token)
        for name in sorted(os.listdir(folder)):
            if name.startswith("page") and not name.endswith(".csv"):
                return send_file(os.path.join(folder, name))
        abort(404)

    @app.errorhandler(413)
    def too_large(_):
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        return render_template("index.html",
                               error=f"That image is larger than {limit} MB. "
                                     "Try again at a lower resolution."), 413

    @app.errorhandler(404)
    def not_found(_):
        return render_template("rejected.html", not_found=True,
                               reason="That result has expired or never existed. "
                                      "Uploads are not kept indefinitely."), 404

    return app


def _folder(app: Flask, token: str) -> str:
    """Resolve an upload token to its directory, refusing anything crafted."""
    if not TOKEN_RE.match(token or ""):
        abort(404)
    folder = os.path.join(app.config["UPLOAD_ROOT"], token)
    if not os.path.isdir(folder):
        abort(404)
    return folder


# ------------------------------------------------------------------- reading

def run_reader(image_path: str) -> dict:
    """Run the ReadMigraineDiary CLI over one image."""
    try:
        proc = subprocess.run([sys.executable, READER, image_path, "--quiet"],
                              capture_output=True, text=True,
                              timeout=READ_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "Reading the image took too long and was stopped."}
    except OSError as exc:
        return {"ok": False, "reason": f"The reader could not be started ({exc})."}

    if proc.returncode == 0:
        return {"ok": True, "reason": ""}
    return {"ok": False, "reason": _clean_reason(proc.stderr)}


def _clean_reason(stderr: str) -> str:
    """Turn the CLI's stderr line into something worth showing a person."""
    line = (stderr or "").strip().splitlines()[-1:] or [""]
    text = line[0]
    text = re.sub(r"^ReadMigraineDiary:\s*", "", text)
    text = re.sub(r"^rejected [^:]*:\s*", "", text)
    return text or "The image could not be read."


def read_rows(csv_path: str) -> list:
    """Load the emitted CSV into rows ready for the table."""
    with open(csv_path, newline="") as fh:
        raw = list(csv.reader(fh))
    if raw and raw[0][:1] == ["date"]:
        raw = raw[1:]

    rows = []
    for record in raw:
        if len(record) < 4:
            continue
        date = record[0]
        try:
            day = _dt.date.fromisoformat(date)
            label = day.strftime("%a")
            number = day.day
        except ValueError:
            label, number = "", date
        # Named "marks", not "values": Jinja resolves ``row.values`` to the
        # dict's built-in .values method, which silently renders as empty.
        marks = [record[i].strip().lower() == "yes" for i in (1, 2, 3)]
        rows.append({
            "date": date,
            "weekday": label,
            "day": number,
            "weekend": label in ("Sat", "Sun"),
            "marks": marks,
            "marked": any(marks),
        })
    return rows


def summarise(rows: list) -> dict:
    counts = [sum(1 for r in rows if r["marks"][i]) for i in range(3)]
    return {
        "days": len(rows),
        "counts": counts,
        "affected": sum(1 for r in rows if r["marked"]),
    }


def month_label(rows: list) -> str:
    if not rows:
        return ""
    try:
        return _dt.date.fromisoformat(rows[0]["date"]).strftime("%B %Y")
    except ValueError:
        return ""


app = create_app()

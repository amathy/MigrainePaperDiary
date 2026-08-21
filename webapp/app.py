"""A small Flask front end for ReadMigraineDiary.

The flow is deliberately linear: photograph or upload a diary month page,
``ReadMigraineDiary`` is run over it, and the resulting CSV is shown as a table
with a download.  If the reader refuses the image, the refusal is shown as-is -
the point of the reader's rejection path is that a page it cannot register
honestly produces no data, and the web app must not paper over that.

**Nothing is retained.**  These are photographs of somebody's health record.
The upload is written to a temporary directory only because the reader takes a
file path, and that directory is removed in a ``finally`` block before the
response is returned - on the rejection path too, and on the error path too.
The table is rendered from memory and the CSV download is produced in the
browser from data embedded in the page, so no result outlives the request and
there is nothing on disk to leak, prune, or expire.

The reader is invoked as the actual command-line program rather than imported.
That keeps the web app using exactly the tool that was measured, and isolates
the worker from anything OpenCV might do to a malformed image.
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from flask import Flask, Response, render_template, request
from werkzeug.utils import secure_filename

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READER = os.path.join(ROOT, "ReadMigraineDiary")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp",
                      ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
READ_TIMEOUT_SECONDS = 180
THUMBNAIL_WIDTH = 900          # the page shown back for checking, not stored
MAX_CSV_ECHO_BYTES = 64 * 1024

COLUMN_LABELS = (("migraine", "M", "Migraine"),
                 ("headache", "H", "Headache"),
                 ("medication", "Rx", "Medication"))


def _upload_root() -> str:
    return os.environ.get("DIARY_UPLOAD_DIR", os.path.join(ROOT, "uploads"))


def _sweep_seconds() -> float:
    """How long an *orphaned* working directory may linger before a sweep."""
    try:
        return max(60.0, float(os.environ.get("DIARY_SWEEP_MINUTES", "15")) * 60.0)
    except ValueError:
        return 900.0


# Directories this app has ever created: the current per-request working
# directory, and the token directories an earlier version kept results in.
_OURS = re.compile(r"\A(read-[A-Za-z0-9_]+|[0-9a-f]{32})\Z")


def sweep_orphans(root: str, older_than: float) -> int:
    """Remove working directories a crashed request could have left behind.

    The normal path deletes its own directory in a ``finally`` block, so this
    only ever finds debris from a process that was killed mid-read.  It is a
    safety net, not the retention mechanism - there is no retention.

    It also clears the token directories an earlier version of this app used to
    keep results in, so upgrading actually disposes of what that version
    stored rather than leaving it on disk forever.  Only directories matching
    a shape this app creates are touched.
    """
    if not os.path.isdir(root):
        return 0
    cutoff = time.time() - older_than
    removed = 0
    for name in os.listdir(root):
        folder = os.path.join(root, name)
        if not _OURS.match(name) or not os.path.isdir(folder):
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
    app.config["SWEEP_SECONDS"] = _sweep_seconds()
    os.makedirs(app.config["UPLOAD_ROOT"], exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}, 200

    @app.post("/upload")
    def upload():
        posted = request.files.get("page")
        if posted is None or not posted.filename:
            return render_template("index.html",
                                   error="Choose a photo of a diary page first."), 400

        name = secure_filename(posted.filename) or "page.jpg"
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return render_template(
                "index.html",
                error=f"{ext or 'That file type'} is not an image this app can open. "
                      "Use a JPEG, PNG or HEIC photo."), 400

        sweep_orphans(app.config["UPLOAD_ROOT"], app.config["SWEEP_SECONDS"])

        # A temporary directory, removed unconditionally below. The reader
        # needs a path on disk; nothing else here does.
        folder = tempfile.mkdtemp(prefix="read-", dir=app.config["UPLOAD_ROOT"])
        try:
            image_path = os.path.join(folder, "page" + ext)
            posted.save(image_path)
            thumbnail = thumbnail_data_uri(image_path)
            outcome = run_reader(image_path)

            if not outcome["ok"]:
                return render_template("rejected.html", reason=outcome["reason"],
                                       image_data=thumbnail), 422

            csv_path = os.path.splitext(image_path)[0] + ".csv"
            with open(csv_path, newline="") as fh:
                csv_text = fh.read()
        finally:
            shutil.rmtree(folder, ignore_errors=True)

        rows = parse_rows(csv_text)
        label = month_label(rows)
        return render_template("result.html", rows=rows, summary=summarise(rows),
                               month_label=label, columns=COLUMN_LABELS,
                               csv_text=csv_text, image_data=thumbnail,
                               filename=csv_filename(label))

    @app.post("/download")
    def download():
        """Echo posted CSV text back as an attachment (no-JavaScript fallback).

        The browser normally builds the file itself from data already in the
        page; this exists so the download still works with scripting off. The
        text is passed straight through and never touches disk.
        """
        text = request.form.get("csv", "")[:MAX_CSV_ECHO_BYTES]
        name = csv_filename(request.form.get("label", ""))
        return Response(text, mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.errorhandler(413)
    def too_large(_):
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        return render_template("index.html",
                               error=f"That image is larger than {limit} MB. "
                                     "Try again at a lower resolution."), 413

    @app.errorhandler(404)
    def not_found(_):
        return render_template("rejected.html", not_found=True,
                               reason="Nothing is kept after a page is read, so "
                                      "results cannot be revisited by URL. "
                                      "Upload the photo again to read it."), 404

    return app


def csv_filename(label: str) -> str:
    stem = (label or "migraine-diary").lower().replace(" ", "-")
    stem = re.sub(r"[^a-z0-9._-]", "", stem) or "migraine-diary"
    return f"{stem}.csv"


def thumbnail_data_uri(path: str) -> str:
    """A small JPEG of the uploaded page, inlined in the response.

    Shown so the reading can be checked against the photo it came from. It is
    embedded in the HTML rather than served from a URL precisely because there
    is no stored file to serve.
    """
    try:
        from PIL import Image, ImageOps

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except Exception:
            pass

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.width > THUMBNAIL_WIDTH:
                height = round(im.height * THUMBNAIL_WIDTH / im.width)
                im = im.resize((THUMBNAIL_WIDTH, height), Image.LANCZOS)
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=72, optimize=True)
    except Exception:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


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


def parse_rows(csv_text: str) -> list:
    """Turn the emitted CSV text into rows ready for the table."""
    raw = list(csv.reader(io.StringIO(csv_text)))
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

# ReadMigraineDiary

Turns a photo or scan of a filled-in migraine-diary month page into a CSV of
daily migraine / headache / medication entries — as a command-line tool and as
a web app you can point a phone camera at.

```bash
./ReadMigraineDiary path/to/january.jpg
# writes path/to/january.csv
```

```csv
date,migraine,headache,medication
2026-01-01,no,no,yes
2026-01-02,no,no,no
2026-01-03,no,yes,yes
```

If the image is not a diary month page — or is too degraded to read honestly —
nothing is written, a reason is printed to stderr, and the exit status is 1.

## Accuracy

Measured on the 360-image synthetic set in `Training/` (12 month pages × 30
photographs each, spanning perspective, page curl, uneven light and shadow,
defocus and motion blur, sensor noise, JPEG artefacts, resolutions from 820 px
to 2600 px, and 90°/180° rotations):

| metric | result |
| --- | --- |
| cell accuracy (date × column) | **99.60 %** (32 719 / 32 850) |
| whole-day rows exactly right | 99.39 % |
| images read with zero errors | 340 / 360 |
| images at or above 95 % accuracy | 359 / 360 |
| worst accepted image | 95.56 % |
| images rejected | 1 / 360 |
| non-diary images correctly rejected | 12 / 12 |

`run_tests.py` covers the template model, reading, rotation invariance,
leap-year February, all 12 rejection cases, the CLI contract and the web app
end to end — including that nothing is left on disk after a read: **71 checks,
0 failures**.

The single rejected page is a small, heavily motion-blurred frame whose ArUco
markers decode to the *wrong* ids; the template check catches the resulting
mis-registration and the page is refused rather than mis-transcribed.

Reproduce with:

```bash
python generate_training_data.py && python evaluate.py
```

## Web app

**Live at https://migraine-diary-reader.onrender.com**

A small Flask front end wraps the same tool: photograph or upload a month page,
and the reading comes back as a table with a CSV download.

```bash
pip install -r requirements.txt
python wsgi.py                     # http://localhost:5000
```

- **Home** — a drop zone with *Take a photo* (opens the camera on a phone via
  `capture="environment"`) and *Choose a file*, with a preview before sending.
  Below it, the blank diary itself is offered at `/diary-template.pdf` with
  printing instructions (the pages are already A6, so print at 100% rather than
  "fit to page", or four-up on A4 and cut), and a disclaimer: not a medical
  device, readings are automated and can be wrong, check them against the
  paper.
- **Upload** — the image is written to a temporary directory and
  `ReadMigraineDiary` is run over it as a subprocess, then the directory is
  deleted. Running the actual CLI keeps the web app on exactly the code path
  that was measured, and isolates the worker from anything OpenCV might do to a
  malformed image.
- **Result** — the CSV rendered as a table, with per-column totals, weekday
  labels, a collapsible view of the photo that was read, and a download button.
  Rendered directly into the upload response; see *Nothing is retained* below.
- **Rejected** — if the reader refuses the image, the refusal is shown along
  with the reader's own reason and advice on retaking the photo. Nothing is
  transcribed and no CSV is produced. The whole point of the reader's rejection
  path is that a page it cannot register yields no data, and the web app must
  not paper over that.

### Nothing is retained

These are photographs of somebody's health record, so no copy of the photo or
of the reading outlives the request that produced it:

- The upload is written to a temporary directory **only** because the reader
  takes a file path. That directory is removed in a `finally` block before the
  response is returned — on the rejection path and the error path too.
- The result table is rendered straight into the response. There is no result
  URL, so a reading cannot be revisited, shared by link, or found by guessing.
- The **CSV download is built in the browser** from data embedded in the page,
  so the file never has to exist on the server. With scripting off, a fallback
  form posts the text back and it is echoed straight through, still untouched
  by disk.
- The photo shown back under "the page that was read" is a downscaled JPEG
  inlined as a `data:` URI — again, because there is no stored file to serve.
- `sweep_orphans` exists purely as a safety net for a request killed mid-read,
  and it also clears the token directories an earlier version of this app kept
  results in, so upgrading disposes of them. It only touches directories whose
  names match a shape this app creates.

Two things this does *not* claim. Render's access log records request lines, so
it will show that a `POST /upload` happened, with a timestamp — but no filename
and no diary content. And the result page cannot be reloaded or bookmarked;
refreshing re-submits the form, which is the honest consequence of keeping
nothing.

| variable | default | purpose |
| --- | --- | --- |
| `DIARY_UPLOAD_DIR` | `./uploads` | parent for the per-request temporary directory |
| `DIARY_SWEEP_MINUTES` | `15` | age at which crash debris is swept |
| `PORT` | `5000` | dev-server port (`python wsgi.py`) |

### Deploying to Render

`render.yaml` is a Blueprint: push the repo to GitHub, then in Render pick
**New → Blueprint** and point it at the repo. It provisions one Python web
service running

```
gunicorn wsgi:app --workers 1 --threads 4 --timeout 240 --bind 0.0.0.0:$PORT
```

with `/healthz` as the health check and uploads under `/tmp`. Three details
that matter for a server deployment:

- `requirements.txt` uses **`opencv-contrib-python-headless`** — the regular
  wheel needs `libGL`, which server images do not have.
- Reading happens in a subprocess, so the web worker stays small; one worker
  with a few threads is enough and keeps memory within a small instance.
- The blueprint defaults to the **free** plan, which spins down when idle
  (expect a slow first request after a pause). Switch `plan:` to `starter` for
  an always-on instance.

The committed `diary/template_model.json` and `diary/template_pages/` mean the
deploy needs neither the source PDF nor PyMuPDF.

**Sizing.** A free instance gives 512 MB and 0.15 CPU, and the reader is built
for accuracy rather than speed, so one page takes roughly **35 seconds** there
(about 2 seconds on a laptop). The free plan also spins down when idle, adding
around 50 seconds to the first request after a pause. Both go away on `starter`.
Verified on the live free instance: four pages read, every CSV byte-identical
to its ground truth, and non-diary images refused in about 6 seconds.

Getting it to fit 512 MB took three changes worth knowing about, since a
straightforward implementation does not fit: the polynomial displacement field
is evaluated in horizontal bands instead of building a design matrix over every
pixel at once (a few hundred MB for one page); the page high-pass is computed
once per rectification candidate rather than on every scoring call; and input
images are capped at 3000 px on the long side, since a 12 MP phone photo
otherwise carries tens of MB through every intermediate copy for no benefit.

## How it works

The diary was designed to be machine-readable: each month page carries four
ArUco markers (`DICT_4X4_50`) whose ids encode both the month and the corner
(`month = id // 4 + 1`, `corner = id % 4`). Everything downstream is classical
computer vision — no learned model, nothing inferred from what a diary
"usually" says.

**1 · Template model** (`diary/template.py`). The canonical geometry is derived
straight from the source PDF: the checkbox rectangles come from the PDF's own
vector drawing operators, and the marker corners from detecting the markers in
a clean 300 dpi render. The reader therefore measures exactly where the printer
printed, not where a hand-transcribed guess says it should be.

**2 · Rectification** (`diary/geometry.py`), in three increasingly local
stages. Each stage is a *proposal*, adopted only if it raises the correlation
with the clean template, so a stage that misfires costs time and nothing else.

- *Markers.* The detected marker corners give an initial homography. Detection
  runs over several rescaled and contrast-equalised views, and any marker that
  the detector missed is recovered from its rejected candidates using the known
  four-marker board layout.
- *Landmarks.* Printed features — box borders, day numbers, headers, footer —
  are located by masked normalised cross-correlation, coarse patches first,
  and a corrected homography is fitted through them.
- *Displacement field.* A robustly fitted low-order polynomial mops up what a
  homography structurally cannot: page curl and lens distortion are not
  projective.

**3 · Verification.** Every month the markers support is tried, and one is
accepted only if the rectified page actually correlates with that month's clean
template. This is what makes a mis-decoded marker id a *rejection* instead of a
plausible-looking wrong month.

**4 · Reading** (`diary/reader.py`). The page is flat-fielded, then each
checkbox interior — inset far enough that neither the printed border nor a
neighbour's overshooting stroke can reach it — is thresholded and measured.

### Robustness choices worth naming

- **RANSAC, then least squares on the inliers.** Plain RANSAC is actively wrong
  on the four marker corners: when the page is curled *no* homography fits all
  of them, so a tight threshold declares good points outliers and locks onto
  whichever minimal subset it can fit exactly — skewing the whole page. A
  generous threshold rejects only true mis-detections, and the refit spreads
  the unavoidable residual evenly.
- **Ratio-tested landmarks.** The diary is a regular grid, so a patch can match
  almost as well one row out of register. A match is used only when its peak
  clearly beats the best rival peak in the search window; ambiguous landmarks
  are dropped, not guessed.
- **Masked landmark patches.** Large patches are distinctive but unavoidably
  cover checkboxes, so each patch carries a mask excluding every box interior.
  What is compared is only ink the printer put down — which lets the printed
  box borders serve as dense landmarks exactly where sampling accuracy matters
  most.
- **Phase correlation for the global shift.** It looks at the whole page at
  once, so unlike a patch search it cannot land one row off.
- **Connectivity, not coverage.** A box reads "yes" from the largest
  *connected* run of ink, not from total darkness. A pen stroke — even a thin,
  faint, pencil one — is a continuous line; dust, sensor noise and JPEG ringing
  are isolated specks.
- **Contrast-relative ink threshold.** How dark this page's *known printed* ink
  came out is a per-photo contrast reference, so one threshold serves both a
  crisp scan and a dim phone snap.

## Rejection

An image is refused when any of these fails, and the reason says which:

- no diary markers found at all;
- the page geometry cannot be fitted to the template;
- too little of the printed layout is locatable after rectification;
- the rectified page does not correlate with that month's clean template.

Verified against `Training/negatives/`: the diary's own cover, instructions and
notes pages (no markers), month pages with the marker corners cropped away, a
month page photographed unreadably small, blank paper, random noise, a page of
printed text, and an unrelated checkbox form. Two of these do produce spurious
marker detections — and are then caught by the template check, which scores
them at 0.00–0.01 against a threshold of 0.35.

## Repository layout

| path | purpose |
| --- | --- |
| `ReadMigraineDiary` | the CLI |
| `webapp/app.py` | Flask routes, upload handling, CSV rendering |
| `webapp/templates/` | home, result and rejection pages |
| `wsgi.py` | WSGI entry point (`gunicorn wsgi:app`) |
| `render.yaml` | Render Blueprint |
| `diary/template.py` | builds/loads the canonical template model from the PDF |
| `diary/geometry.py` | marker detection and rectification |
| `diary/reader.py` | verification and checkbox reading |
| `diary/synth.py` | pen-mark rendering and phone-photo simulation |
| `generate_training_data.py` | builds `Training/images` + `Training/ground_truth` |
| `make_negatives.py` | builds `Training/negatives` |
| `evaluate.py` | scores the reader against the ground truth |
| `run_tests.py` | template, reading, rejection, CLI and web app checks |

## Training data

`generate_training_data.py` renders 30 filled-in variants of each of the 12
month pages and photographs each one. Marks are drawn as ticks, crosses,
scribble fills, slashes, blobs and circles with a wobbly variable-pressure
stroke model, in black, blue and pencil, at varying widths and opacities, with
per-page consistency (one person, mostly one pen and one habit) plus realistic
incidentals — the year written on the header lines, an occasional margin note.
Each page is then curled, perspective-warped onto a textured surface, lit
unevenly, shadowed, blurred, noised, colour-shifted, downsampled and JPEG
compressed. Ground truth is sampled with plausible structure: migraine episodes
cluster over consecutive days and medication follows pain.

Images land in `Training/images/monthMM_NN_difficulty.jpg` with matching ground
truth in `Training/ground_truth/` in the same CSV format the reader emits, so
predictions and truth are directly comparable.

## Setup

```bash
python -m venv .venv && .venv/bin/python -m pip install -r requirements-dev.txt
```

`requirements.txt` is the runtime (web app + CLI); `requirements-dev.txt` adds
PyMuPDF and pyflakes, needed only to rebuild the template model or regenerate
the training set.

The CLI re-executes itself inside `./.venv` if OpenCV is not importable from
the interpreter that launched it, so `./ReadMigraineDiary image.jpg` works
without activating the environment first.

`diary/template_model.json` and `diary/template_pages/` are committed, so
reading does not need the PDF. Rebuild them after a template change with:

```bash
python -m diary.template "Template/Headache diary with ArUco markersA6_nice.pdf"
```

## Known limitations

- The month is taken from the marker ids. If every marker is unreadable the
  page is rejected; it is not guessed from the page content, because month
  pages are typographically near-identical and a wrong month would be a
  confident-looking fabrication.
- Dates use the current year unless `--year` is given. February emits 28 or 29
  rows according to that year, regardless of the 29 rows printed on the page.
- Only month pages are read. The dotted notes pages carry no markers and are
  rejected by design.

"""Synthesis of realistic filled-in, photographed diary pages.

Two stages:

1. :func:`render_filled_page` takes the clean canonical render of a month page
   and draws hand-made pen marks into the checkboxes that the ground truth
   says are "yes".  Marks are ticks, crosses, scribble fills, slashes and
   blobs, drawn with a wobbly variable-pressure stroke model so they look
   hand-drawn rather than vector-perfect.

2. :func:`photograph` turns that flat page into something that looks like a
   phone snapshot: page curl, perspective, a desk background, uneven
   lighting and shadows, defocus/motion blur, sensor noise, white balance
   drift, downsampling and JPEG artefacts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from . import template as T

# ---------------------------------------------------------------- pen strokes

INK_COLOURS = [
    (18, 18, 22),      # black biro
    (36, 34, 30),      # soft black
    (70, 42, 26),      # dark blue (BGR)
    (96, 58, 30),      # lighter blue biro
    (110, 108, 104),   # pencil
    (86, 84, 80),      # dark pencil
]


def _smooth_noise(rng: np.random.Generator, n: int, amp: float) -> np.ndarray:
    """Low-frequency 1-D wobble, used to make straight strokes look human."""
    ctrl = rng.normal(0.0, amp, size=max(3, n // 8 + 2))
    x = np.linspace(0, len(ctrl) - 1, n)
    return np.interp(x, np.arange(len(ctrl)), ctrl)


def _humanise(pts: np.ndarray, rng: np.random.Generator, wobble: float) -> np.ndarray:
    """Resample a polyline densely and add correlated jitter."""
    pts = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-6:
        return pts
    n = max(12, int(total / 1.5))
    t = np.concatenate([[0.0], np.cumsum(seg)]) / total
    u = np.linspace(0, 1, n)
    out = np.stack([np.interp(u, t, pts[:, 0]), np.interp(u, t, pts[:, 1])], axis=1)
    out[:, 0] += _smooth_noise(rng, n, wobble)
    out[:, 1] += _smooth_noise(rng, n, wobble)
    return out


def _draw_stroke(canvas: np.ndarray, pts: np.ndarray, colour, width: float,
                 rng: np.random.Generator, opacity: float = 1.0) -> None:
    """Draw one pen stroke with varying width/pressure onto a BGR canvas."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return
    n = len(pts)
    pressure = 1.0 + _smooth_noise(rng, n, 0.22)
    pressure = np.clip(pressure, 0.55, 1.5)
    # Pen lifts slightly at the very start/end of a stroke.
    taper = np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / max(2.0, n * 0.08), 0.45, 1.0)

    # Work in a local buffer around the stroke's bounding box; a page-sized
    # buffer per segment is what makes naive versions of this crawl.
    pad = int(math.ceil(width * 2 + 4))
    H, W = canvas.shape[:2]
    x0 = max(0, int(np.floor(pts[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - pad)
    x1 = min(W, int(np.ceil(pts[:, 0].max())) + pad)
    y1 = min(H, int(np.ceil(pts[:, 1].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    local = pts - np.array([x0, y0], np.float64)
    layer = np.zeros((y1 - y0, x1 - x0), np.float32)
    scratch = np.empty_like(layer)
    for i in range(n - 1):
        w = max(1, int(round(width * pressure[i] * taper[i])))
        a = float(np.clip(opacity * (0.75 + 0.35 * pressure[i]), 0.15, 1.0))
        p0 = tuple(np.round(local[i]).astype(int))
        p1 = tuple(np.round(local[i + 1]).astype(int))
        scratch[:] = 0.0
        cv2.line(scratch, p0, p1, 1.0, w, cv2.LINE_AA)
        np.maximum(layer, scratch * a, out=layer)

    # Ballpoint ink is not perfectly uniform - modulate coverage a little.
    grain = cv2.GaussianBlur(rng.random(layer.shape, dtype=np.float32), (0, 0), 1.6)
    layer *= 0.78 + 0.32 * (grain - grain.mean() + 0.5)
    layer = np.clip(layer, 0.0, 1.0)[:, :, None]
    col = np.asarray(colour, np.float32).reshape(1, 1, 3)
    view = canvas[y0:y1, x0:x1]
    view[:] = view * (1.0 - layer) + col * layer


# --------------------------------------------------------------- mark shapes

def _tick(rng: np.random.Generator, w: float, h: float) -> np.ndarray:
    """Check mark: short down-stroke then a long up-stroke, often overshooting."""
    x0 = rng.uniform(0.05, 0.28) * w
    y0 = rng.uniform(0.42, 0.60) * h
    xm = rng.uniform(0.32, 0.48) * w
    ym = rng.uniform(0.72, 0.95) * h
    x1 = rng.uniform(0.80, 1.22) * w
    y1 = rng.uniform(-0.30, 0.22) * h
    return np.array([[x0, y0], [xm, ym], [x1, y1]])


def _cross(rng: np.random.Generator, w: float, h: float) -> List[np.ndarray]:
    pad = rng.uniform(-0.12, 0.18)
    a = np.array([[pad * w, pad * h], [(1 - pad) * w, (1 - pad) * h]])
    b = np.array([[(1 - pad) * w, pad * h], [pad * w, (1 - pad) * h]])
    return [a, b]


def _slash(rng: np.random.Generator, w: float, h: float) -> List[np.ndarray]:
    pad = rng.uniform(-0.05, 0.15)
    if rng.random() < 0.5:
        return [np.array([[pad * w, (1 - pad) * h], [(1 - pad) * w, pad * h]])]
    return [np.array([[pad * w, pad * h], [(1 - pad) * w, (1 - pad) * h]])]


def _hatch_fill(rng: np.random.Generator, w: float, h: float) -> List[np.ndarray]:
    """Back-and-forth scribble that fills the square, as the diary asks for."""
    strokes = []
    step = rng.uniform(0.075, 0.17) * h
    pad = rng.uniform(0.02, 0.14)
    diag = rng.random() < 0.35
    y = pad * h
    flip = False
    while y < (1 - pad) * h:
        x0, x1 = pad * w, (1 - pad) * w
        if flip:
            x0, x1 = x1, x0
        dy = step * 0.55 if diag else 0.0
        strokes.append(np.array([[x0, y], [x1, y + dy * (1 if flip else -1)]]))
        y += step
        flip = not flip
    if not strokes:
        strokes.append(np.array([[pad * w, h / 2], [(1 - pad) * w, h / 2]]))
    # join the sweeps into one continuous scribble
    joined = [strokes[0]]
    for s in strokes[1:]:
        joined.append(np.array([joined[-1][-1], s[0]]))
        joined.append(s)
    return [np.concatenate(joined, axis=0)]


def _blob(rng: np.random.Generator, w: float, h: float) -> List[np.ndarray]:
    """Solid-ish filled square drawn as tight overlapping loops."""
    cx, cy = w / 2, h / 2
    strokes = []
    for k in range(rng.integers(3, 6)):
        rx = rng.uniform(0.22, 0.44) * w * (1 - 0.12 * k)
        ry = rng.uniform(0.22, 0.44) * h * (1 - 0.12 * k)
        th = np.linspace(0, 2 * math.pi * rng.uniform(1.0, 1.8), 40)
        strokes.append(np.stack([cx + rx * np.cos(th), cy + ry * np.sin(th)], axis=1))
    return strokes


def _circle_mark(rng: np.random.Generator, w: float, h: float) -> List[np.ndarray]:
    cx, cy = w * rng.uniform(0.45, 0.55), h * rng.uniform(0.45, 0.55)
    rx, ry = w * rng.uniform(0.34, 0.52), h * rng.uniform(0.34, 0.52)
    th = np.linspace(rng.uniform(0, 6), rng.uniform(0, 6) + 2 * math.pi * rng.uniform(1.0, 1.3), 60)
    return [np.stack([cx + rx * np.cos(th), cy + ry * np.sin(th)], axis=1)]


MARK_STYLES = ("fill", "tick", "cross", "slash", "blob", "circle")
STYLE_WEIGHTS = np.array([0.34, 0.26, 0.16, 0.08, 0.10, 0.06])


def _mark_strokes(style: str, rng: np.random.Generator, w: float, h: float) -> List[np.ndarray]:
    if style == "tick":
        return [_tick(rng, w, h)]
    if style == "cross":
        return _cross(rng, w, h)
    if style == "slash":
        return _slash(rng, w, h)
    if style == "fill":
        return _hatch_fill(rng, w, h)
    if style == "blob":
        return _blob(rng, w, h)
    return _circle_mark(rng, w, h)


# --------------------------------------------------------------- page filling

def _draw_digit_strokes(rng: np.random.Generator, x: float, y: float, s: float) -> List[np.ndarray]:
    """A crude hand-written digit-ish squiggle for the year lines."""
    n = rng.integers(2, 4)
    strokes = []
    for _ in range(n):
        pts = np.stack([
            x + rng.uniform(0, 0.7, 4) * s,
            y + rng.uniform(-1.0, 0.05, 4) * s,
        ], axis=1)
        strokes.append(pts)
    return strokes


def render_filled_page(month: int, truth: Dict[int, Tuple[bool, bool, bool]],
                       rng: np.random.Generator) -> np.ndarray:
    """Return a BGR canonical page with the ground-truth marks drawn on it."""
    clean = T.clean_page(month)
    canvas = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR).astype(np.float32)
    boxes = T.month_model(month)["boxes"]

    # One person, one pen (mostly) - occasionally they switch mid-month.
    base_colour = np.asarray(INK_COLOURS[rng.integers(len(INK_COLOURS))], np.float32)
    base_width = rng.uniform(3.4, 7.0)
    style_bias = rng.dirichlet(STYLE_WEIGHTS * 12.0)
    consistency = rng.uniform(0.55, 0.98)
    personal_style = MARK_STYLES[int(rng.choice(len(MARK_STYLES), p=style_bias))]

    for day, values in sorted(truth.items()):
        rects = boxes[str(day)]
        for col, on in enumerate(values):
            if not on:
                continue
            x0, y0, x1, y1 = rects[col]
            w, h = x1 - x0, y1 - y0
            style = personal_style if rng.random() < consistency else \
                MARK_STYLES[int(rng.choice(len(MARK_STYLES), p=style_bias))]
            strokes = _mark_strokes(style, rng, w, h)
            colour = np.clip(base_colour + rng.normal(0, 9, 3), 0, 200)
            width = float(np.clip(base_width * rng.uniform(0.82, 1.20), 2.2, 9.0))
            opacity = float(np.clip(rng.normal(0.92, 0.10), 0.42, 1.0))
            ox, oy = rng.normal(0, 0.035 * w), rng.normal(0, 0.035 * h)
            for s in strokes:
                pts = _humanise(s, rng, wobble=w * rng.uniform(0.012, 0.045))
                pts[:, 0] += x0 + ox
                pts[:, 1] += y0 + oy
                _draw_stroke(canvas, pts, colour, width, rng, opacity)

    # Realistic incidentals: the year written on the header lines, and the
    # occasional stray note in the margin.
    if rng.random() < 0.75:
        for i in range(2):
            for s in _draw_digit_strokes(rng, 920 + i * 75 + rng.uniform(-5, 5), 124, 40):
                _draw_stroke(canvas, _humanise(s, rng, 1.6), base_colour, base_width, rng, 0.9)
    if rng.random() < 0.15:
        # a short note scrawled in the free right-hand margin
        y = rng.uniform(420, 1300)
        x = rng.uniform(1020, 1060)
        for _ in range(int(rng.integers(1, 4))):
            pts = np.stack([x + np.cumsum(rng.uniform(3, 11, 14)),
                            y + rng.normal(0, 6, 14)], axis=1)
            _draw_stroke(canvas, _humanise(pts, rng, 1.4), base_colour, base_width * 0.8, rng, 0.8)
            y += rng.uniform(26, 40)

    return np.clip(canvas, 0, 255).astype(np.uint8)


# ---------------------------------------------------------- photo simulation

def _paper_texture(shape, rng: np.random.Generator) -> np.ndarray:
    n = cv2.GaussianBlur(rng.normal(0, 1, shape[:2]).astype(np.float32), (0, 0), 1.1)
    n = n / (np.abs(n).max() + 1e-6)
    return 1.0 + n[:, :, None] * rng.uniform(0.005, 0.022)


def _curl(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gentle cylindrical bend, as when a page will not lie flat."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    amp_x = rng.uniform(-0.030, 0.030) * w
    amp_y = rng.uniform(-0.022, 0.022) * h
    phase = rng.uniform(0, math.pi)
    dx = amp_x * np.sin(math.pi * yy / h + phase)
    dy = amp_y * np.sin(math.pi * xx / w + phase * 0.7)
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _background(shape, rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    base = np.array([rng.uniform(35, 175), rng.uniform(35, 170), rng.uniform(35, 170)], np.float32)
    bg = np.ones((h, w, 3), np.float32) * base
    # coarse texture (desk grain / tablecloth / wood)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), rng.uniform(2, 18))
    tex /= (np.abs(tex).max() + 1e-6)
    bg *= 1.0 + tex[:, :, None] * rng.uniform(0.05, 0.28)
    if rng.random() < 0.35:  # faint wood grain lines
        stripes = np.sin(np.linspace(0, rng.uniform(8, 40), w))[None, :] * rng.uniform(4, 14)
        bg += stripes[:, :, None]
    return np.clip(bg, 0, 255)


def _illumination(shape, rng: np.random.Generator) -> np.ndarray:
    """Smooth multiplicative light field: gradient + vignette + soft shadow."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = xx / w, yy / h
    field = 1.0 + rng.uniform(-0.35, 0.35) * (nx - 0.5) + rng.uniform(-0.35, 0.35) * (ny - 0.5)

    # vignette
    r = np.sqrt((nx - 0.5) ** 2 + (ny - 0.5) ** 2) / 0.707
    field *= 1.0 - rng.uniform(0.05, 0.38) * r ** 2

    # a couple of soft blobs: highlight from a lamp, shade from the phone
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(0, 1), rng.uniform(0, 1)
        sigma = rng.uniform(0.18, 0.55)
        strength = rng.uniform(-0.34, 0.24)
        field *= 1.0 + strength * np.exp(-(((nx - cx) ** 2 + (ny - cy) ** 2) / (2 * sigma ** 2)))

    if rng.random() < 0.30:  # harder shadow edge from a hand or the phone itself
        ang = rng.uniform(0, math.pi)
        d = (nx - rng.uniform(0.2, 0.8)) * math.cos(ang) + (ny - rng.uniform(0.2, 0.8)) * math.sin(ang)
        field *= 1.0 - rng.uniform(0.15, 0.42) / (1.0 + np.exp(-d / rng.uniform(0.01, 0.06)))

    return np.clip(field, 0.12, 1.9)[:, :, None]


def _perspective(page_shape, out_shape, rng: np.random.Generator, severity: float):
    h, w = page_shape
    H, W = out_shape
    margin = rng.uniform(0.03, 0.13)
    scale = rng.uniform(0.68, 0.97)
    box_w, box_h = W * scale, H * scale
    cx = W / 2 + rng.normal(0, W * margin * 0.5)
    cy = H / 2 + rng.normal(0, H * margin * 0.5)

    ar = w / h
    if box_w / box_h > ar:
        box_w = box_h * ar
    else:
        box_h = box_w / ar

    dst = np.array([
        [cx - box_w / 2, cy - box_h / 2],
        [cx + box_w / 2, cy - box_h / 2],
        [cx + box_w / 2, cy + box_h / 2],
        [cx - box_w / 2, cy + box_h / 2],
    ], np.float32)

    jitter = severity * min(box_w, box_h)
    dst += rng.normal(0, jitter, dst.shape).astype(np.float32)

    ang = math.radians(rng.normal(0, 7.0))
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]], np.float32)
    centre = dst.mean(0)
    dst = (dst - centre) @ R.T + centre

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def _motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(3, 12))
    ang = rng.uniform(0, math.pi)
    kern = np.zeros((k, k), np.float32)
    cv2.line(kern, (int(k / 2 - math.cos(ang) * k / 2), int(k / 2 - math.sin(ang) * k / 2)),
             (int(k / 2 + math.cos(ang) * k / 2), int(k / 2 + math.sin(ang) * k / 2)), 1.0, 1)
    if kern.sum() == 0:
        return img
    return cv2.filter2D(img, -1, kern / kern.sum())


def photograph(page: np.ndarray, rng: np.random.Generator, difficulty: str = "normal") -> np.ndarray:
    """Simulate a phone photo of ``page`` lying on some surface."""
    sev = {"easy": 0.008, "normal": 0.022, "hard": 0.042}[difficulty]

    page = page.astype(np.float32) * _paper_texture(page.shape, rng)
    if rng.random() < (0.25 if difficulty == "easy" else 0.65):
        page = _curl(page, rng)

    long_side = int(rng.integers(*{"easy": (1500, 2600), "normal": (1100, 2600),
                                   "hard": (820, 1900)}[difficulty]))
    if rng.random() < 0.5:
        out_h, out_w = long_side, int(long_side * rng.uniform(0.68, 0.80))
    else:
        out_h, out_w = int(long_side * rng.uniform(0.68, 0.80)), long_side

    M = _perspective(page.shape[:2], (out_h, out_w), rng, sev)
    bg = _background((out_h, out_w), rng)

    warped = cv2.warpPerspective(page, M, (out_w, out_h), flags=cv2.INTER_AREA,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.ones(page.shape[:2], np.float32), M, (out_w, out_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)

    # drop shadow under the page so the edges are not razor sharp
    shadow = cv2.GaussianBlur(mask, (0, 0), rng.uniform(4, 16))
    off = int(rng.integers(2, 14))
    shadow = np.roll(np.roll(shadow, off, axis=0), off, axis=1)
    bg *= (1.0 - rng.uniform(0.15, 0.5) * shadow)[:, :, None]

    m3 = np.clip(mask, 0, 1)[:, :, None]
    img = warped * m3 + bg * (1.0 - m3)

    img *= _illumination((out_h, out_w), rng)

    # white balance / exposure drift
    img *= np.array([rng.uniform(0.90, 1.10), rng.uniform(0.93, 1.07),
                     rng.uniform(0.90, 1.12)], np.float32)
    img *= rng.uniform(0.82, 1.18)

    if rng.random() < 0.55:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, {"easy": 1.1, "normal": 1.9, "hard": 2.8}[difficulty]))
    if rng.random() < (0.10 if difficulty == "easy" else 0.30):
        img = _motion_blur(img, rng)

    img += rng.normal(0, rng.uniform(1.0, {"easy": 3.0, "normal": 6.5, "hard": 11.0}[difficulty]),
                      img.shape).astype(np.float32)
    img = np.clip(img, 0, 255)

    # camera contrast curve
    g = rng.uniform(0.85, 1.18)
    img = 255.0 * np.power(img / 255.0, g)

    out = np.clip(img, 0, 255).astype(np.uint8)

    rot = rng.random()
    if rot < 0.06:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rot < 0.12:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rot < 0.16:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return out


# ---------------------------------------------------------------- ground truth

def sample_truth(month: int, year: int, rng: np.random.Generator) -> Dict[int, Tuple[bool, bool, bool]]:
    """Draw a plausible month of migraine data (episodes cluster; meds follow pain)."""
    n = T.days_in_month(month, year)
    p_mig = rng.uniform(0.03, 0.30)
    p_head = rng.uniform(0.05, 0.35)
    p_med_mig = rng.uniform(0.55, 0.97)
    p_med_head = rng.uniform(0.15, 0.70)
    truth = {}
    mig_run = head_run = 0
    for day in range(1, n + 1):
        p = p_mig * (3.2 if mig_run else 1.0)
        migraine = rng.random() < min(p, 0.85)
        mig_run = mig_run + 1 if migraine and mig_run < 2 else (1 if migraine else 0)

        headache = False
        if not migraine:
            p = p_head * (2.2 if head_run else 1.0)
            headache = rng.random() < min(p, 0.8)
        head_run = head_run + 1 if headache and head_run < 2 else (1 if headache else 0)

        if migraine:
            med = rng.random() < p_med_mig
        elif headache:
            med = rng.random() < p_med_head
        else:
            med = rng.random() < 0.02
        truth[day] = (migraine, headache, med)
    return truth

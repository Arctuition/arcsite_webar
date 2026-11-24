"""The photo's camera, solved against the house wall a deck goes on.

1. A Gemini text model marks the wall in the photo: its ground line and ends, straight lines along it and up it, the
   side wall at a visible corner, doors (known heights) and any existing deck. Points are [y, x] in 0-1000.
2. Vanishing points give the camera's turn and tilt (and the focal length when two perpendicular directions are seen),
   the ground line gives where the wall stands, door heights (or a 5 ft camera) give the scale.

The answer is a camera in the wall frame: feet, origin on the ground at the wall's left end as seen from the yard,
x along the wall to the right, y up, z out from the wall toward the yard (where the camera stands).

  .venv/bin/python -m tuning.camera angled existing-deck   # analyses, solves and writes out/camera/<photo>.jpg overlays
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
CACHE = HERE / "out" / "walls.json"

DOOR_FT = {"hinged": 6.67, "french": 6.67, "sliding": 6.67, "garage": 7.0}
CAMERA_FT = 5.0  # a phone held at eye level, when no door gives the scale
DEFAULT_HFOV = 63.0
HAND_HELD_FT = 5.2  # weak prior on the camera's height above the ground
HEIGHT_SIGMA_FT = 3.0
HEIGHT_WEIGHT = 5e-4  # vs the line cost: moves a straight-on camera's tilt freely, a well-constrained one hardly  # a phone's main camera, when the photo shows no two perpendicular directions

WALL_PROMPT = """\
You are measuring a photo of a house so a new deck can be drawn against one of its walls in correct perspective. \
Pick the single exterior wall a deck would attach to: a ground-level wall the camera sees well, usually the one with the \
main back or patio doors. Never pick a wall or wall section that holds garage doors or fronts a driveway; when the house \
front has a garage, pick the section beside it with the entry door or the windows. Points are [y, x] normalized to 0-1000. Be precise: these lines are used to compute the camera.
Return JSON:
{"label": a short description of that wall,
 "base_line": [[y, x], [y, x]]: the left and right ends (in the image) of the line where that wall meets the ground. \
Where bushes or furniture hide it, give where the wall would meet the ground behind them. If an existing deck or porch \
covers the bottom of the wall so the ground there cannot be seen, give null,
 "left_end": "corner" if the wall's left end in the image is a visible building corner, "cut" if the wall runs out of \
the picture or is hidden there,
 "right_end": the same for its right end,
 "top_line": [[y, x], [y, x]]: the top edge of that wall where it meets the roof or eave, its left and right ends,
 "horizontal_lines": up to 6 more long straight lines that run along that wall (siding courses, tops and bottoms of \
windows and doors, trim, a fence or deck rim fixed to it), each [[y, x], [y, x]] with the two points far apart,
 "vertical_lines": up to 6 straight vertical building edges on or at that wall (its corners, sides of doors and windows, \
downpipes), each [[y, x], [y, x]] with the two points far apart; prefer long ones such as the wall's corners from the \
ground to the eave,
 "side_wall": null, or, if a building corner of that wall is visible together with the other wall meeting it there: \
{"end": "left" or "right" (which end of the main wall), "lines": up to 4 long straight lines running along that other \
wall (its ground line, top edge, siding, window tops), each [[y, x], [y, x]]},
 "doors": [{"type": "hinged", "french", "sliding" or "garage", "box": [ymin, xmin, ymax, xmax]}] for doors in that wall \
(the box is the door opening itself, frame to frame, threshold to head),
 "existing_deck": null, or, for a deck, porch or patio already built against that wall: {"box": [ymin, xmin, ymax, xmax], \
"surface_line": [[y, x], [y, x]] the left and right ends of the line where that wall meets the deck's floor, \
"height_ft": how high that floor is above the ground (count the stair risers, about 7 inches each)}}"""


# ---------------------------------------------------------------- analysis


def analyze(name: str, jpeg: bytes, refresh: bool = False) -> dict:
    """The model's marks for one photo, cached per photo name under out/ so runs agree."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if refresh or name not in cache:
        cache[name] = mark_wall(jpeg)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache[name]


def mark_wall(jpeg: bytes) -> dict:
    """The model's marks for a photo, uncached (raises app.gemini.GeminiError)."""
    from app import gemini

    return gemini.analyze([WALL_PROMPT, (jpeg, "image/jpeg")], timeout=120)


# ---------------------------------------------------------------- geometry helpers


def _px(pt, w: int, h: int) -> np.ndarray:
    """[y, x] in 0-1000 → pixel [u, v]."""
    return np.array([float(pt[1]) / 1000 * w, float(pt[0]) / 1000 * h])


def _segments(raw, w, h) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for seg in raw or []:
        try:
            a, b = _px(seg[0], w, h), _px(seg[1], w, h)
        except (TypeError, IndexError, ValueError):
            continue
        if np.linalg.norm(a - b) > 0.04 * max(w, h):
            out.append((a, b))
    return out


def _line(a, b) -> np.ndarray:
    l = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
    return l / np.linalg.norm(l[:2])


def vanishing_point(segments) -> np.ndarray | None:
    """Least-squares meeting point of image lines, homogeneous (w ≈ 0 when they are parallel). Each line weighs by length."""
    if len(segments) < 2:
        return None
    rows = []
    for a, b in segments:
        rows.append(_line(a, b) * math.sqrt(np.linalg.norm(a - b)))
    _, _, vt = np.linalg.svd(np.array(rows))
    v = vt[-1]
    return v / np.linalg.norm(v)


def _direction(v: np.ndarray, f: float, c: np.ndarray) -> np.ndarray:
    """Camera-frame direction (x right, y down, z forward) of a homogeneous vanishing point."""
    d = np.array([v[0] - c[0] * v[2], v[1] - c[1] * v[2], f * v[2]])
    return d / np.linalg.norm(d)


def _focal_from(v1, v2, c) -> float | None:
    """Focal length from two vanishing points of perpendicular directions (both finite)."""
    if v1 is None or v2 is None or abs(v1[2]) < 1e-9 or abs(v2[2]) < 1e-9:
        return None
    p1, p2 = v1[:2] / v1[2] - c, v2[:2] / v2[2] - c
    f2 = -float(p1 @ p2)
    return math.sqrt(f2) if f2 > 0 else None



# ---------------------------------------------------------------- pixel lines


def detect_lines(jpeg: bytes) -> np.ndarray:
    """Straight segments in the photo (OpenCV's line segment detector), as an (n, 2, 2) array of pixel end points."""
    import cv2

    gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return np.zeros((0, 2, 2))
    segs = lines.reshape(-1, 2, 2).astype(float)
    keep = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) > 0.02 * max(gray.shape)
    return segs[keep]


def _inside(poly: np.ndarray, pts: np.ndarray, grow: float) -> np.ndarray:
    import cv2

    centre = poly.mean(axis=0)
    poly = (centre + (poly - centre) * grow).astype(np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in pts])


def _angle_err(segs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Angle (degrees, 0-90) between each segment and the line from its middle to the vanishing point."""
    mid = segs.mean(axis=1)
    seg = segs[:, 1] - segs[:, 0]
    to_vp = vp[:2][None, :] - mid * vp[2]
    cross = np.abs(seg[:, 0] * to_vp[:, 1] - seg[:, 1] * to_vp[:, 0])
    dot = np.abs((seg * to_vp).sum(axis=1))
    return np.degrees(np.arctan2(cross, dot))


def snap(marks: dict, w: int, h: int, detected: np.ndarray, params, rounds=((4.0, 1.12), (2.5, 1.08), (1.5, 1.05))):
    """Replace the model's rough lines by the photo's own: detected segments inside the wall (or the side wall) that
    point at the vanishing point the current fit predicts, refit, repeat with a tighter tolerance."""
    base = _segments([marks.get("base_line") or (marks.get("existing_deck") or {}).get("surface_line")], w, h)[0]
    top = _segments([marks.get("top_line")], w, h)
    along_marks = _segments([marks.get("top_line")] + list(marks.get("horizontal_lines") or []), w, h) + [base]
    if top:
        tl, tr = sorted(top[0], key=lambda q: q[0])
    else:
        ys = [q[1] for sgm in along_marks for q in sgm]
        tl, tr = np.array([base[0][0], min(ys)]), np.array([base[1][0], min(ys)])
    bl, br = sorted(base, key=lambda q: q[0])
    wall_poly = np.array([bl, br, tr, tl])
    side = _segments((marks.get("side_wall") or {}).get("lines"), w, h)
    side_poly = None
    if side:
        import cv2

        pts = np.array([q for sgm in side for q in sgm], np.float32)
        side_poly = cv2.convexHull(pts).reshape(-1, 2).astype(float)
        if len(side_poly) < 3:
            side_poly = None
    mids = detected.mean(axis=1) if len(detected) else np.zeros((0, 2))
    in_wall = _inside(wall_poly, mids, 1.0) if len(detected) else np.zeros(0, bool)
    used = {}
    for tol, grow in rounds:
        yaw, pitch, roll, hfov = params
        f = w / 2 / math.tan(math.radians(hfov / 2))
        k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        rot = _rotation(yaw, pitch, roll)
        in_wall = _inside(wall_poly, mids, grow) if len(detected) else in_wall
        in_side = _inside(side_poly, mids, grow) if side_poly is not None and len(detected) else np.zeros(len(detected), bool)
        fams, used = [], {}
        for name, d, region in (("along", (1, 0, 0), in_wall), ("vertical", (0, 1, 0), in_wall | in_side),
                                ("side", (0, 0, 1), in_side)):
            if not len(detected):
                break
            ok = region & (_angle_err(detected, k @ rot @ np.array(d, float)) < tol)
            segs = [(a, b) for a, b in detected[ok]]
            if name == "along":
                segs.append((bl, br))  # the ground line always counts
            if len(segs) >= 2:
                fams.append(_family(segs, d))
                used[name] = segs
        if not fams:
            break
        params, _ = fit_rotation(fams, w, h, init=params)
    return params, used


# ---------------------------------------------------------------- solve


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """World (wall frame: x right, y up, z to the yard) → camera (x right, y down, z forward). yaw > 0 looks toward +x,
    pitch > 0 looks down, roll turns the image clockwise; all in degrees, 0/0/0 faces the wall straight on."""
    ys, ps = math.radians(yaw), math.radians(pitch)
    look = np.array([math.cos(ps) * math.sin(ys), -math.sin(ps), -math.cos(ps) * math.cos(ys)])
    right = np.cross(look, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(look, right)
    r = math.radians(roll)
    x = math.cos(r) * right + math.sin(r) * down
    y = -math.sin(r) * right + math.cos(r) * down
    return np.stack([x, y, look])


def _fit_cost(params, fams, w, h, prior_hfov):
    yaw, pitch, roll, hfov = params
    if not (5 <= hfov <= 120):
        return 1e9
    f = w / 2 / math.tan(math.radians(hfov / 2))
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    rot = _rotation(yaw, pitch, roll)
    total = 0.0
    for d, (mid, seg, length) in fams:
        vp = k @ rot @ d
        to_vp = vp[:2][None, :] - mid * vp[2]
        cross = seg[:, 0] * to_vp[:, 1] - seg[:, 1] * to_vp[:, 0]
        sin = cross / (np.linalg.norm(seg, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-12)
        total += float((length * sin**2).sum() / length.sum())
    return total + 0.002 * ((hfov - prior_hfov) / 20) ** 2 + 0.0005 * (roll / 5) ** 2


def fit_rotation(fams, w, h, prior_hfov=DEFAULT_HFOV, init=None):
    """Turn, tilt, roll and field of view that best explain the marked lines' directions: a coarse grid, then a
    shrinking local search. Lines along the wall, up it and along the side wall each vote for their own direction."""
    best, best_cost = (list(init), _fit_cost(init, fams, w, h, prior_hfov)) if init else (None, 1e18)
    for yaw in (range(-85, 86, 5) if init is None else []):
        for pitch in range(-40, 41, 5):
            for hfov in range(35, 106, 10):
                c = _fit_cost((yaw, pitch, 0.0, hfov), fams, w, h, prior_hfov)
                if c < best_cost:
                    best, best_cost = [float(yaw), float(pitch), 0.0, float(hfov)], c
    step = [2.5, 2.5, 2.0, 5.0]
    while max(step) > 0.05:
        moved = False
        for i in range(4):
            for sgn in (1, -1):
                trial = list(best)
                trial[i] += sgn * step[i]
                c = _fit_cost(trial, fams, w, h, prior_hfov)
                if c < best_cost:
                    best, best_cost, moved = trial, c, True
        if not moved:
            step = [s / 2 for s in step]
    return best, best_cost


def _family(segments, d):
    a = np.array([s[0] for s in segments])
    b = np.array([s[1] for s in segments])
    seg = b - a
    return (np.array(d, float), ((a + b) / 2, seg, np.linalg.norm(seg, axis=1)))


def solve(marks: dict, w: int, h: int, detected: np.ndarray | None = None) -> dict:
    c = np.array([w / 2, h / 2])
    deck_hidden = marks.get("base_line") is None and (marks.get("existing_deck") or {}).get("surface_line")
    level_ft = 0.0  # height of the line the wall's ends are measured on
    if deck_hidden:
        base_raw = marks["existing_deck"]["surface_line"]
        level_ft = float(marks["existing_deck"].get("height_ft") or 0)
    else:
        base_raw = marks["base_line"]
    base = _segments([base_raw], w, h)[0]
    left, right = sorted(base, key=lambda p: p[0])
    along = _segments([marks.get("top_line")] + list(marks.get("horizontal_lines") or []), w, h) + [(left, right)]
    vertical = _segments(marks.get("vertical_lines"), w, h)
    side_lines = _segments((marks.get("side_wall") or {}).get("lines"), w, h)

    fams = [_family(along, (1, 0, 0))]
    if vertical:
        fams.append(_family(vertical, (0, 1, 0)))
    if side_lines:
        fams.append(_family(side_lines, (0, 0, 1)))
    params, cost = fit_rotation(fams, w, h)
    rough = list(params)
    used = {}
    if detected is not None and len(detected):
        params, used = snap(marks, w, h, detected, params)
        cost = _fit_cost(params, [_family(v, d) for v, d in ((used.get("along"), (1, 0, 0)),
                         (used.get("vertical"), (0, 1, 0)), (used.get("side"), (0, 0, 1))) if v], w, h, DEFAULT_HFOV)
    def geometry(params):
        yaw, pitch, roll, hfov = params
        focal = w / 2 / math.tan(math.radians(hfov / 2))
        rot = _rotation(yaw, pitch, roll)  # world → camera

        def ray(p):
            r = rot.T @ np.array([p[0] - c[0], p[1] - c[1], focal])
            return r / np.linalg.norm(r)

        # With the camera 1 unit above the measuring line: where the wall's ends stand (camera at the origin)
        def ground(p):
            r = ray(p)
            if r[1] >= -1e-6:
                raise ValueError("the wall's ground line is above the horizon")
            return r * (-1.0 / r[1])

        gl, gr = ground(left), ground(right)
        origin_z = (gl[2] + gr[2]) / 2  # the wall plane z (world axes are the wall's own after the fit)

        def on_wall(p):
            r = ray(p)
            if abs(r[2]) < 1e-9:
                return None
            t = origin_z / r[2]
            return r * t if t > 0 else None

        doors = []
        for door in marks.get("doors") or []:
            try:
                y0, x0, y1, x1 = [float(v) for v in door["box"]]
            except (KeyError, TypeError, ValueError):
                continue
            xm = (x0 + x1) / 2
            top, bottom = on_wall(_px([y0, xm], w, h)), on_wall(_px([y1, xm], w, h))
            if top is None or bottom is None or top[1] <= bottom[1]:
                continue
            doors.append({"type": door.get("type"), "scale": DOOR_FT.get(str(door.get("type")), 6.67) / (top[1] - bottom[1]),
                          "bottom": bottom})
        if doors:
            scale, scale_from = float(np.median([d["scale"] for d in doors])), f"{len(doors)} door(s)"
        else:
            scale, scale_from = CAMERA_FT - level_ft, "camera at 5 ft"

        origin = np.array([gl[0], -1.0, origin_z])  # left end, on the measuring line
        eye = (-origin) * scale
        eye[1] += level_ft  # from the measuring line down to the ground
        return {
            "size": [w, h],
            "focal_px": focal,
            "hfov_deg": hfov,
            "vfov_deg": math.degrees(2 * math.atan(h / 2 / focal)),
            "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
            "scale_from": scale_from,
            "measured_on": f"existing deck surface at {level_ft:.1f} ft" if deck_hidden else "ground",
            "camera_height_ft": float(eye[1]),
            "eye": eye.tolist(),
            "rotation": rot.tolist(),
            "wall_length_ft": float((gr[0] - gl[0]) * scale),
            "left_end": marks.get("left_end"),
            "right_end": marks.get("right_end"),
            "doors": [{"type": d["type"], "x_ft": float((d["bottom"][0] - gl[0]) * scale),
                       "sill_ft": float((d["bottom"][1] + 1.0) * scale + level_ft)} for d in doors],
        }

    # Lines barely fix the tilt of a camera facing the wall straight on; a weak pull toward a hand-held camera height
    # settles it there, while a tilt the lines do fix (a photo from a balcony) stays.
    fams_used = [_family(v, d) for v, d in ((used.get("along"), (1, 0, 0)), (used.get("vertical"), (0, 1, 0)),
                                            (used.get("side"), (0, 0, 1))) if v] or fams

    def total(q):
        try:
            height = geometry(q)["camera_height_ft"]
        except ValueError:
            return 1e9
        return _fit_cost(q, fams_used, w, h, DEFAULT_HFOV) + HEIGHT_WEIGHT * ((height - HAND_HELD_FT) / HEIGHT_SIGMA_FT) ** 2

    lines_only = list(params)
    best_cost = total(params)
    step = [1.0, 1.0, 0.5, 2.0]
    while max(step) > 0.05:
        moved = False
        for i in range(4):
            for sgn in (1, -1):
                trial = list(params)
                trial[i] += sgn * step[i]
                c2 = total(trial)
                if c2 < best_cost:
                    params, best_cost, moved = trial, c2, True
        if not moved:
            step = [v / 2 for v in step]
    out = geometry(params)
    out.update({
        "fit_cost": _fit_cost(params, fams_used, w, h, DEFAULT_HFOV),
        "lines": {"along": len(along), "vertical": len(vertical), "side": len(side_lines)},
        "rough_fit": [round(v, 1) for v in rough],
        "lines_only_fit": [round(v, 1) for v in lines_only],
        "snapped": {k: len(v) for k, v in used.items()},
        "snapped_lines": {k: [[a.tolist(), b.tolist()] for a, b in v] for k, v in used.items()},
    })
    return out


def project(cam: dict, pts) -> np.ndarray:
    """Wall-frame points (feet) → pixels; NaN for points behind the camera."""
    rot = np.array(cam["rotation"])
    eye = np.array(cam["eye"])
    w, h = cam["size"]
    out = []
    for p in pts:
        q = rot @ (np.asarray(p, float) - eye)
        out.append([np.nan, np.nan] if q[2] <= 1e-6 else [w / 2 + cam["focal_px"] * q[0] / q[2], h / 2 + cam["focal_px"] * q[1] / q[2]])
    return np.array(out)


# ---------------------------------------------------------------- the deck on that wall


def place(deck: dict, cam: dict, wall: dict | None = None) -> dict:
    """Where the deck goes along the photo's wall: the design's distance from the wall's end (wall = the engine's
    exported wall run {a, b} in the outline frame), measured from a visible corner; else centred on the doors or wall."""
    from .describe import Frame

    f = Frame(deck)
    xs = [p[0] for p in f.points]
    width = max(xs) - min(xs)
    photo_len = cam["wall_length_ft"]
    how = "centred on the wall"
    left = (photo_len - width) / 2
    if wall:
        ends = sorted(f._x(p) - f.x0 for p in (wall["a"], wall["b"]))
        from_left, from_right = -ends[0], ends[1] - width  # design gaps between the deck and each wall end
        if cam.get("left_end") == "corner":
            left, how = from_left, "design distance from the left corner"
        elif cam.get("right_end") == "corner":
            left, how = photo_len - width - from_right, "design distance from the right corner"
    left = max(0.0, min(left, max(0.0, photo_len - width)))
    return {"x_ft": left, "width_ft": width, "how": how}


def deck_points(deck: dict, placement: dict):
    """The deck's outline and stairs footprint in the wall frame, at floor height (for overlays)."""
    from .describe import Frame

    f = Frame(deck)
    y = deck["height_in"] / 12
    outline = [(x + placement["x_ft"], y, z) for x, z in f.points]
    stairs = []
    for s in deck.get("stairs", []):
        pts = deck["outline"]
        a, b = pts[s["edge"]], pts[(s["edge"] + 1) % len(pts)]
        n = math.dist(a, b)
        p0 = f.map((a[0] + (b[0] - a[0]) * s["from"] / n, a[1] + (b[1] - a[1]) * s["from"] / n))
        p1 = f.map((a[0] + (b[0] - a[0]) * (s["from"] + s["width"]) / n, a[1] + (b[1] - a[1]) * (s["from"] + s["width"]) / n))
        ex, ez = (p1[0] - p0[0]) / s["width"], (p1[1] - p0[1]) / s["width"]
        ox, oz = ez, -ex  # outward of a counter-clockwise edge in the wall frame (x right, z to the yard)
        cx, cz = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        cx0, cz0 = [(sum(q[i] for q in f.points) / len(f.points)) for i in (0, 1)]
        if (cx - cx0) * ox + (cz - cz0) * oz < 0:
            ox, oz = -ox, -oz
        run = s["risers"] * 0.83
        X = placement["x_ft"]
        stairs.append([(p0[0] + X, y, p0[1]), (p1[0] + X, y, p1[1]),
                       (p1[0] + ox * run + X, 0, p1[1] + oz * run), (p0[0] + ox * run + X, 0, p0[1] + oz * run)])
    return outline, stairs


def engine_camera(deck: dict, cam: dict, placement: dict) -> dict:
    """The solved camera in the engine's deck frame (outline feet, z up) for AR_Web_RenderDeckPhotoDiagram."""
    from .describe import Frame

    f = Frame(deck)
    rot = np.array(cam["rotation"])  # wall → camera
    look = rot.T @ [0, 0, 1]
    up = rot.T @ [0, -1, 0]

    def to_outline(p, point=True):
        x, y, z = p
        if point:
            x, z = x - placement["x_ft"] + f.x0, z + f.z0
        return [x * f.right[0] + z * f.out[0], x * f.right[1] + z * f.out[1], y]

    eye = np.array(cam["eye"])
    return {"projection": "perspective", "eye": to_outline(eye), "target": to_outline(eye + look * 20),
            "up": to_outline(up, point=False), "fovDeg": cam["vfov_deg"]}


# ---------------------------------------------------------------- overlay


def overlay(photo: bytes, marks: dict, cam: dict, deck: dict | None = None, placement: dict | None = None) -> bytes:
    im = Image.open(io.BytesIO(photo)).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, w // 400)
    for seg in [marks["base_line"]] + list(marks.get("horizontal_lines") or []) + [marks.get("top_line")]:
        for a, b in _segments([seg], w, h):
            d.line([tuple(a), tuple(b)], fill=(0, 200, 255), width=lw)
    for a, b in _segments(marks.get("vertical_lines"), w, h):
        d.line([tuple(a), tuple(b)], fill=(255, 200, 0), width=lw)
    for a, b in _segments((marks.get("side_wall") or {}).get("lines"), w, h):
        d.line([tuple(a), tuple(b)], fill=(120, 255, 120), width=lw)
    colours = {"along": (0, 120, 255), "vertical": (255, 140, 0), "side": (0, 170, 0)}
    for name, segs in (cam.get("snapped_lines") or {}).items():
        for a, b in segs:
            d.line([tuple(a), tuple(b)], fill=colours[name], width=lw + 1)
    # A 1 ft grid on the ground in front of the wall, and the wall's ground line, from the solved camera
    L = cam["wall_length_ft"]
    for x in np.arange(0, L + 0.01, 2):
        p = project(cam, [(x, 0, 0), (x, 0, 16)])
        if not np.isnan(p).any():
            d.line([tuple(p[0]), tuple(p[1])], fill=(255, 255, 255), width=1)
    for z in range(0, 17, 2):
        p = project(cam, [(0, 0, z), (L, 0, z)])
        if not np.isnan(p).any():
            d.line([tuple(p[0]), tuple(p[1])], fill=(255, 255, 255), width=1)
    p = project(cam, [(0, 0, 0), (L, 0, 0), (L, 8, 0), (0, 8, 0)])
    if not np.isnan(p).any():
        d.line([tuple(q) for q in p] + [tuple(p[0])], fill=(255, 60, 60), width=lw)
    if deck and placement:
        outline, stairs = deck_points(deck, placement)
        for poly in [outline] + stairs:
            q = project(cam, poly)
            if not np.isnan(q).any():
                d.line([tuple(x) for x in q] + [tuple(q[0])], fill=(255, 0, 255), width=lw * 2)
            ground_poly = project(cam, [(x, 0, z) for x, _, z in poly])
            if not np.isnan(ground_poly).any():
                d.line([tuple(x) for x in ground_poly] + [tuple(ground_poly[0])], fill=(160, 0, 160), width=lw)
    text = (f"hfov {cam['hfov_deg']:.0f}  yaw {cam['yaw_deg']:.0f}  pitch {cam['pitch_deg']:.0f}  roll {cam['roll_deg']:.1f}  "
            f"cam {cam['camera_height_ft']:.1f} ft ({cam['scale_from']})  wall {cam['wall_length_ft']:.1f} ft")
    d.rectangle([0, 0, w, 14 + w // 70], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=88)
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-snap", action="store_true", help="use only the model's lines, not the photo's detected ones")
    ap.add_argument("--wall", default=None, help="JSON file with the engine's exported wall run {a, b}")
    args = ap.parse_args()
    deck = json.loads((HERE / "deck.json").read_text())
    wall = json.loads(Path(args.wall).read_text()) if args.wall else None
    out_dir = HERE / "out" / "camera"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.photos:
        from .inputs import prep_like_canvas

        jpeg = prep_like_canvas((HERE / "photos" / f"{name}.jpg").read_bytes(), 2048, 92)
        w, h = Image.open(io.BytesIO(jpeg)).size
        marks = analyze(name, jpeg, args.refresh)
        cam = solve(marks, w, h, None if args.no_snap else detect_lines(jpeg))
        placement = place(deck, cam, wall)
        cam["placement"] = placement
        cam["engine"] = engine_camera(deck, cam, placement)
        results[name] = cam
        (out_dir / f"{name}.jpg").write_bytes(overlay(jpeg, marks, cam, deck, placement))
        print(name, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in cam.items()
                                if k not in ("rotation", "snapped_lines", "engine")}, default=str))
    path = out_dir / "cameras.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(results)
    path.write_text(json.dumps(merged, indent=1))


if __name__ == "__main__":
    main()

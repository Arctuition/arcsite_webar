"""Show a generated deck in the homeowner's photo with Gemini.

Mirrors the iOS ARDeckAI flow: one "design" call adds the deck to the photo in
the chosen material (the master); every other material is a "restyle" of that
master, so the deck keeps its shape across materials. Unlike iOS, the deck's
structure comes from the model itself:

1. A text model finds where on the house this deck shape could go, by snap kind:
   one wall (rectangle, angled), an outside corner (L) or a part of the house that
   sticks out (U). For each candidate: where it meets the ground in the image, how
   far the camera is off to its left or right, and which one is recommended
   (analyze_photo; cached per photo and kind).
2. The homeowner taps a candidate on the photo (and, on a wall, drags the deck along
   it): a placement. By default it is the recommended candidate.
3. The deck's USDZ is rendered from that viewpoint and from the top, as plain
   diagrams, and the placement is drawn on a copy of the photo as a guide line.
   L decks come in two shapes, wrapping to the right or to the left; only corners
   that turn the same way are offered.
4. Photo, guide and diagrams go to the image model, each labeled, with the deck
   described relative to the building (a template that the photo screen can edit).
5. Each result is checked: first for the guide's magenta left in it and for the photo
   changed away from the deck (pixel_check, no model), then against the design by the text
   model (one set of stairs where the design puts them, the house and framing unchanged,
   nothing else added, the deck at its height). Two results are generated in parallel and
   the first that passes is returned; if neither does, one more round, and failing that,
   the best result is returned flagged as unverified. A single generation matched the
   design about 40-50% of the time on hard photos; the checker agreed with a human on 13/14.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import gemini, render
from .deck import FT, Layout, _inward_normals, make_layout

MAX_PHOTO_DIMENSION = 1536  # same upload size as the iOS implementation
# Always-available references; the main one is rendered per photo from its analyzed viewpoint
# (matched_reference), which tested far better than a fixed front or 45° view on angled photos.
REFERENCE_VIEWS = ("front", "top")

# Same presets as ARDeckAIStyle on iOS
STYLES = {
    "cedar": ("Natural Cedar", "natural cedar wood boards"),
    "composite": ("Gray Composite", "gray composite decking boards"),
    "ipe": ("Ipe Hardwood", "dark ipe hardwood boards"),
    "white": ("White Painted", "white painted wood boards"),
}

HEIGHT_ANCHORS = {2: "about knee height on an adult", 3: "between knee and hip height on an adult", 4: "about waist height on an adult", 6: "about shoulder height on an adult"}

# Prompt templates, editable from the photo screen. [PHOTO], [PLACEMENT GUIDE], [MATCHED VIEW], [TOP VIEW]
# and [FRONT VIEW] mark where the images go (text and images are sent interleaved). {deck} is the
# generated description of this deck, {wall} the chosen wall, {placement} where on it the deck goes,
# {guide} how to read the guide line, {viewpoint} the camera angle, and {style} the material phrase.
IMAGE_MARKERS = {"PHOTO": "photo", "PLACEMENT GUIDE": "guide", "MATCHED VIEW": "matched", "TOP VIEW": "top", "FRONT VIEW": "front"}
MAX_TEMPLATE_CHARS = 8000

DESIGN_TEMPLATE = """\
Edit the photo below by adding a newly built deck to it. Change nothing else in the photo.
PHOTO TO EDIT:
[PHOTO]
PLACEMENT GUIDE: the same photo with a bright magenta line drawn on the ground where the deck meets the building. \
{guide} The line is only a guide: never draw it.
[PLACEMENT GUIDE]
DESIGN DIAGRAM 1: the deck drawn from the same viewpoint as the photo ({viewpoint}). \
A plain diagram, not a scene; the gray block is the building wall the deck is attached to:
[MATCHED VIEW]
DESIGN DIAGRAM 2: the same deck from directly above (top view); the gray block is the building, at the top:
[TOP VIEW]
Build exactly this deck, matching the diagrams' shape and proportions:
{deck}
- Decking: {style}; wooden posts and framing.
Use real-world scale from the photo (doors are about 7 ft tall). Keep the deck's true size, even if it then covers part of the foreground.
Output the PHOTO TO EDIT itself with the deck added: same camera position, angle, framing and aspect ratio, same everything else. \
Do not copy the diagrams' gray block or flat colors, and do not draw the guide line. \
Do not add furniture, plants, lights or anything else. Photorealistic, no text, no watermark."""

RESTYLE_TEMPLATE = """\
This photo shows a deck. Change only the deck's material and color to {style}. \
Keep the deck's exact shape, size, position, height, framing, posts, railings and steps, \
and keep everything else in the photo unchanged. Do not add or remove anything. Photorealistic, no text, no watermark."""

# Checks a design result against the design. {stairs} is the stairs line from the deck description.
# Every flag below was a failure the earlier checker let through in testing (reframed photos,
# invented upper decks and pergolas, the guide line left in the image).
JUDGE_PROMPT = """\
The first image is a photo of a house; the second is the same photo after a deck was added to it. \
The deck was meant to be attached to {wall}, about {height} ft above the ground, with this stairs design: {stairs}
Check the second image and answer in JSON:
{"deck_present": true if a deck was added,
 "stairs": where the deck's stairs are, left/right as seen by someone facing that wall: "left side", "right side", "outer edge", "back", "none" or "several",
 "house_changed": true if the house itself (walls, windows, doors, roof, floors) was changed beyond adding the deck,
 "reframed": true if the second image is framed differently from the first: zoomed in or out, cropped, shifted, or seen from a moved camera,
 "added_structures": true if anything was added besides one deck with its railings and stairs, e.g. a second-story deck or balcony, a pergola, a roof or awning,
 "guide_line": true if a bright magenta line or mark is visible anywhere,
 "height_ok": true if the deck sits at about the designed height against the house, not raised to an upper floor, floating, sunken or tilted,
 "matches": true only if a deck was added, its stairs match the stairs design (or there are none if it says no stairs), and nothing above is wrong,
 "problem": if it does not match, a few words on what is wrong, else ""}"""
# Failures that ruin the photo outright rank below a deck that is merely off-design
# when nothing passes and the least bad result is shown.
JUDGE_FLAGS = {  # field: (value that fails, problem text, severity)
    "deck_present": (False, "no deck was added", 2),
    "house_changed": (True, "the house was changed", 2),
    "reframed": (True, "the photo was reframed", 2),
    "added_structures": (True, "extra structures were added", 2),
    "guide_line": (True, "the guide line is still visible", 2),
    "height_ok": (False, "the deck is at the wrong height", 1),
}
CANDIDATES_PER_ROUND = 2
MAX_ROUNDS = 2

# Where each deck shape snaps to on the house
KIND_FOR_SHAPE = {"rect": "wall", "angled": "wall", "l": "corner", "l_left": "corner", "u": "bump"}
# Which way an L deck's side part runs back from the corner, i.e. which corners it fits
CORNER_SIDE = {"l": "right", "l_left": "left"}

# Photo analysis, one prompt per snap kind. Asking which end of a wall is closer to the camera (something
# visible in the image) gave a stable camera side; asking for a signed angle did not. Points come back as
# [y, x] in 0-1000, Gemini's native convention, which it locates most reliably.
_ANALYZE_TAIL = """\
"closer_end": which end of {wall}, in the image, is closer to the camera: "left", "right", or "neither" if the camera faces it straight on,
"angle_deg": how far the camera is turned away from facing {wall} straight on, 0 to 75 (0 = straight on),
"recommended": true for the single best one for a deck (usually by the main doors), else false"""
ANALYZE_PROMPTS = {
    "wall": """\
You are helping a homeowner place a new deck against their house in this photo. Find up to 3 exterior walls a deck \
could attach to: ground-level walls the camera sees well. Points are [y, x] normalized to 0-1000. Return JSON:
{"elevation_deg": how far the camera looks down on the scene (0 = eye level),
 "candidates": [{"label": a short description that identifies the wall in the photo,
   "base_line": [[y1, x1], [y2, x2]]: the left and right ends (in the image) of the line where that wall meets the ground,
   "length_ft": the wall's approximate length, estimated from doors (about 7 ft tall) and windows,
""" + _ANALYZE_TAIL.replace("{wall}", "that wall") + "}]}",
    "corner": """\
You are helping a homeowner wrap a new L-shaped deck around an outside corner of their house in this photo. Find up to 3 \
outside (convex) building corners at ground level where two exterior walls meet and the camera sees both walls. \
For each, the main wall is the one that faces the camera more; the side wall is the other one. \
Points are [y, x] normalized to 0-1000. Return JSON:
{"elevation_deg": how far the camera looks down on the scene (0 = eye level),
 "candidates": [{"label": a short description that identifies the main wall in the photo,
   "corner": the point where the corner meets the ground,
   "main_end": the other end of the main wall's ground line,
   "side_end": the other end of the side wall's ground line,
   "length_ft": the main wall's approximate length, estimated from doors (about 7 ft tall) and windows,
   "side_length_ft": the side wall's approximate length,
""" + _ANALYZE_TAIL.replace("{wall}", "the main wall") + "}]}",
    "bump": """\
You are helping a homeowner wrap a new U-shaped deck around a part of their house that sticks out (a bump-out, wing or \
projecting room), covering its front wall and both of its side walls. Find up to 3 such parts at ground level that the camera \
sees. Points are [y, x] normalized to 0-1000. Return JSON:
{"elevation_deg": how far the camera looks down on the scene (0 = eye level),
 "candidates": [{"label": a short description that identifies that part's front wall in the photo,
   "front_line": [[y1, x1], [y2, x2]]: the left and right ends (in the image) of the line where its front wall meets the ground,
   "left_back": where its left side wall meets the main house wall at the ground,
   "right_back": where its right side wall meets the main house wall at the ground,
   "length_ft": the front wall's approximate length, estimated from doors (about 7 ft tall) and windows,
   "depth_ft": how far that part sticks out from the main house,
""" + _ANALYZE_TAIL.replace("{wall}", "its front wall") + "}]}",
}
FALLBACK_WALL = "the building wall facing the camera"
GUIDE_COLOR = (255, 0, 200)  # magenta: rare in house photos, so the guide line reads as a marker

# Pixel checks of a design result (pixel_check)
PIXEL_SIZE = 384  # width the magenta share is measured at
PIXEL_GRID = (32, 24)  # cells the photo and result are compared in; coarse, so a slight shift doesn't count
PIXEL_CELL_CHANGED = 0.6  # a cell's normalized luminance difference above which it counts as changed
MAGENTA_MAX_SHARE = 0.0008  # more magenta than the photo had, as a share of pixels
OUTSIDE_MAX_CHANGED = 0.2  # share of cells away from the deck that may change; kept framing measured <= 0.10
DECK_AREA_FALLBACK = (0.1, 0.35, 0.9, 1.0)  # without a guide: the lower middle of the photo


@dataclass(frozen=True)
class View:
    """Where the camera stands relative to the wall the deck attaches to."""

    wall: str
    azimuth: int  # degrees; negative = camera off to the wall's left (as seen facing the wall), like render.VIEW_ANGLES
    elevation: int  # degrees above eye level
    wall_length_ft: float | None
    analyzed: bool  # False when the analysis failed and this is the straight-on fallback

    def phrase(self) -> str:
        """The {viewpoint} text."""
        if self.azimuth == 0:
            where = "the camera faces the wall straight on"
        else:
            where = f"the camera stands about {abs(self.azimuth)} degrees off to the {'left' if self.azimuth < 0 else 'right'} of the wall"
        return f"{where}, about {self.elevation} degrees above eye level"


FALLBACK_VIEW = View(FALLBACK_WALL, 0, 10, None, False)


class PhotoError(Exception):
    pass


def style_list() -> list[dict]:
    return [{"id": key, "title": title} for key, (title, _) in STYLES.items()]


def _depth_phrase(width: float, depth: float) -> str:
    ratio = depth / width
    if abs(ratio - 1) < 0.1:
        return "about as deep as it is wide"
    if ratio > 1:
        return f"deeper than it is wide ({ratio:.1f} times its width)"
    return f"about {ratio:.0%} as deep as it is wide, a real deck, not a narrow ledge"


def _shape_lines(shape: str, d: dict[str, int], width: float, depth: float, wall_length: float | None) -> list[str]:
    """Where the deck sits on the building, with left/right as seen by someone facing {wall}."""
    if shape in CORNER_SIDE:
        side = CORNER_SIDE[shape]
        front, full = (d["A"], d["C"]) if shape == "l" else (d["C"], d["A"])  # short and full side lengths
        return [
            "The building has an outside corner where {wall} meets a side wall, and the deck wraps {placement} (see the top view).",
            f"Along {{wall}} it reaches {front} ft out from the wall and spans {d['B']} ft.",
            f"Past the wall's {side} end it continues back along the building's {side} side wall as a {d['D']} ft wide part, "
            f"{full} ft from its outer edge to its back edge.",
        ]
    if shape == "u":
        return [
            "Part of the building sticks out, {wall} being its front, and the deck wraps {placement}: its front and both sides (see the top view).",
            f"Overall {d['B']} ft wide and {d['A']} ft deep. In front of {{wall}} it is {d['A'] - d['wall']} ft deep; "
            f"beside that part, the left part is {d['E']} ft wide and the right part {d['D']} ft wide.",
        ]
    share = ""
    if wall_length and wall_length > width * 1.15:
        share = f" That is about {width / wall_length:.0%} of the wall's length, so it does not run along the whole wall."
    lines = [
        "It is attached to {wall}, {placement}.",
        f"It runs {round(width)} ft along that wall, about {width / 7:.1f} times the height of a door.{share}",
        f"It sticks out {round(depth)} ft from the wall: {_depth_phrase(width, depth)}.",
    ]
    if shape == "angled":
        lines.append(
            "Its outline is NOT a rectangle: the two outer corners are cut off at 45 degrees, so the outer edge has three segments: "
            f"the sides run {d['A']} ft straight out from the wall, then angle inward ({d['B']} ft diagonals) to a {d['C']} ft outer edge parallel to the wall."
        )
    return lines


def _stairs_line(layout: Layout, label: str | None) -> str:
    if not label:
        return "No stairs."
    i = layout.edge_index(label)
    nx, nz = (-v for v in _inward_normals(layout.points)[i])  # outward, in plan (z away from the house)
    if nz > 0.7:
        where = "centered on the deck's outer edge (parallel to the wall, farthest from it), going straight down away from the wall"
    elif nx < -0.7:
        where = "centered on the deck's left side (left as seen facing {wall}), going down to the left, parallel to the wall"
    elif nx > 0.7:
        where = "centered on the deck's right side (right as seen facing {wall}), going down to the right, parallel to the wall"
    else:
        a, b = layout.points[i], layout.points[(i + 1) % len(layout.points)]
        xs = [p[0] for p in layout.points]
        side = "left" if (a[0] + b[0]) / 2 < (min(xs) + max(xs)) / 2 else "right"
        where = f"centered on the back edge of the {side} part, beside the building, going down toward the back"
    return f"One 4 ft wide set of stairs, {where}. No other stairs."


def deck_description(request: dict, layout: Layout, wall_length: float | None = None) -> str:
    """The {deck} lines. They refer to the building as {wall} and {placement}, filled in from the placement."""
    xs = [p[0] for p in layout.points]
    zs = [p[1] for p in layout.points]
    width, depth = (max(xs) - min(xs)) / FT, (max(zs) - min(zs)) / FT
    height = request["height_ft"]
    stairs = request.get("stair_edge")
    lines = _shape_lines(request["shape"], request["dims"], width, depth, wall_length) + [
        f"Its floor is {height} ft above the ground ({HEIGHT_ANCHORS.get(height, 'measured from the ground')}). "
        "The railing top is 3 ft above the floor.",
        _stairs_line(layout, stairs),
        "White railings with vertical balusters on every open edge"
        + (", with a gap only where the stairs are" if stairs else "")
        + ". No railing against the building.",
    ]
    return "\n".join(f"- {line}" for line in lines)


# How {placement} and {guide} read for each snap kind: (with a guide, without one), guide instructions
PLACEMENT_PHRASES = {
    "wall": ("centered on the magenta guide line", "centered on it"),
    "corner": ("around the corner the magenta guide line follows", "around that corner"),
    "bump": ("around the part of the building the magenta guide line outlines", "around that part"),
}
GUIDE_PHRASES = {
    "wall": "Center the deck on that line, with its wall-side edge about as long as the line.",
    "corner": "The deck's back edges follow that line along the two walls it wraps.",
    "bump": "The deck's back edges follow that line along the three walls it wraps.",
}


def _fill(template: str, deck: str, style: str, view: View, kind: str = "wall", placed: bool = False) -> str:
    # str.replace rather than format(): edited templates may contain other braces. {deck} goes first
    # because it contains {wall} and {placement} itself.
    return (
        template.replace("{deck}", deck)
        .replace("{wall}", view.wall)
        .replace("{placement}", PLACEMENT_PHRASES[kind][0 if placed else 1])
        .replace("{guide}", GUIDE_PHRASES[kind])
        .replace("{viewpoint}", view.phrase())
        .replace("{style}", STYLES[style][1])
    )


def design_parts(
    template: str, deck: str, style: str, view: View, images: dict[str, tuple[bytes, str] | None], kind: str = "wall"
) -> list[gemini.Part]:
    """Split the filled template at its image markers into interleaved text and image parts.
    A marker whose image is None (no guide when the photo had no walls to place on) is dropped,
    along with a "PLACEMENT GUIDE" label line just before it."""
    if "[PHOTO]" not in template:
        raise PhotoError("The design prompt must contain [PHOTO] where the photo goes.")
    pattern = "(" + "|".join(re.escape(f"[{m}]") for m in IMAGE_MARKERS) + ")"
    parts: list[gemini.Part] = []
    for chunk in re.split(pattern, _fill(template, deck, style, view, kind, images.get("guide") is not None)):
        if chunk.startswith("[") and chunk[1:-1] in IMAGE_MARKERS:
            image = images.get(IMAGE_MARKERS[chunk[1:-1]])
            if image is not None:
                parts.append(image)
            elif parts and isinstance(parts[-1], str):
                kept = [line for line in parts[-1].splitlines() if not line.startswith("PLACEMENT GUIDE")]
                parts[-1] = "\n".join(kept)
                if not parts[-1].strip():
                    parts.pop()
        elif chunk.strip():
            parts.append(chunk.strip())
    return parts


def _wall_phrase(text: str) -> str:
    """'Front facade with the doors.' -> 'the front facade with the doors', to read well inside sentences."""
    text = text.strip().rstrip(".")
    if not text:
        return FALLBACK_WALL
    text = text[0].lower() + text[1:]
    return text if text.split(" ", 1)[0] in ("the", "a", "an") else "the " + text


def _point(raw) -> list[float]:
    """Gemini's [y, x] in 0-1000 -> [x, y] in 0-1."""
    y, x = raw
    return [min(max(float(x) / 1000, 0.0), 1.0), min(max(float(y) / 1000, 0.0), 1.0)]


def _far(a: list[float], b: list[float]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 0.02


def _candidate(kind: str, raw: dict) -> dict | None:
    """Normalize one candidate; None if its geometry is unusable."""
    angle = min(max(float(raw.get("angle_deg") or 0), 0), 75)
    side = str(raw.get("closer_end", "neither")).lower()
    # The closer end of the wall is on the camera's side: left end closer = camera off to the wall's left
    azimuth = 0 if angle < 10 or side not in ("left", "right") else (-angle if side == "left" else angle)
    c = {
        "label": _wall_phrase(str(raw.get("label") or "")),
        "azimuth": round(azimuth / 5) * 5,
        "length_ft": float(raw["length_ft"]) if raw.get("length_ft") else None,
        "recommended": bool(raw.get("recommended")),
    }
    if kind == "wall":
        line = sorted([_point(p) for p in raw["base_line"]])
        if not _far(*line):
            return None
        c["line"] = line
    elif kind == "corner":
        corner, main_end, side_end = _point(raw["corner"]), _point(raw["main_end"]), _point(raw["side_end"])
        if not (_far(corner, main_end) and _far(corner, side_end)):
            return None
        c.update(corner=corner, main_end=main_end, side_end=side_end,
                 side_length_ft=float(raw["side_length_ft"]) if raw.get("side_length_ft") else None,
                 # Which way the side wall runs back from the corner in the image: an L deck fits only its own side
                 side="left" if side_end[0] < corner[0] else "right")
    else:
        front = sorted([_point(p) for p in raw["front_line"]])
        if not _far(*front):
            return None
        c.update(front_line=front, left_back=_point(raw["left_back"]), right_back=_point(raw["right_back"]),
                 depth_ft=float(raw["depth_ft"]) if raw.get("depth_ft") else None)
    return c


_analysis_cache: dict[str, dict] = {}


def analyze_photo(jpeg: bytes, kind: str) -> dict:
    """Where a deck of this snap kind could go in the photo, from a text model; cached per photo and kind.

    Returns {"kind", "analyzed", "elevation", "candidates": [...]} with points as [x, y] in 0-1. Each candidate
    has "label", "azimuth", "length_ft", "recommended" and, per kind: "line" (wall); "corner", "main_end",
    "side_end", "side_length_ft", "side" (corner); "front_line", "left_back", "right_back", "depth_ft" (bump).
    "candidates" is empty if the analysis failed or found nothing."""
    if kind not in ANALYZE_PROMPTS:
        raise PhotoError(f"Unknown snap kind {kind}.")
    key = f"{kind}:{hashlib.sha1(jpeg).hexdigest()}"
    if key in _analysis_cache:
        return _analysis_cache[key]
    result = {"kind": kind, "analyzed": False, "elevation": 10, "candidates": []}
    try:
        answer = gemini.analyze([ANALYZE_PROMPTS[kind], (jpeg, "image/jpeg")], timeout=90)
        result["elevation"] = round(min(max(float(answer.get("elevation_deg") or 0), 3), 35))
        for raw in answer.get("candidates") or []:
            try:
                candidate = _candidate(kind, raw)
            except (KeyError, TypeError, ValueError):
                continue
            if candidate:
                result["candidates"].append(candidate)
        candidates = result["candidates"] = result["candidates"][:3]
        if candidates and not any(c["recommended"] for c in candidates):
            candidates[0]["recommended"] = True
        for c in [c for c in candidates if c["recommended"]][1:]:  # exactly one
            c["recommended"] = False
        result["analyzed"] = True
    except (gemini.GeminiError, TypeError, ValueError, AttributeError):
        return result  # not cached: a failed analysis (timeout, bad answer) is retried with the next request
    if len(_analysis_cache) > 64:
        _analysis_cache.clear()
    _analysis_cache[key] = result
    return result


def _along(a: list[float], b: list[float], t: float) -> list[float]:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def _share(feet: float, length: float | None, default: float) -> float:
    return min(max(feet / length, 0.12), 1.0) if length else default


def fitting_candidates(analysis: dict, shape: str) -> list[dict]:
    """The candidates this deck shape can go on: an L deck only fits corners that turn its way."""
    side = CORNER_SIDE.get(shape)
    return [c for c in analysis["candidates"] if side is None or c.get("side") == side]


def default_placement(kind: str, analysis: dict, request: dict, layout: Layout) -> dict | None:
    """The recommended candidate (or the first that fits) as a placement, drawn the way the photo screen
    draws it before any tap."""
    fitting = fitting_candidates(analysis, request["shape"])
    c = next((c for c in fitting if c["recommended"]), fitting[0] if fitting else None)
    if not c:
        return None
    d = request["dims"]
    if kind == "wall":
        xs = [p[0] for p in layout.points]
        half = _share((max(xs) - min(xs)) / FT, c["length_ft"], 0.6) / 2
        path = [_along(*c["line"], 0.5 - half), _along(*c["line"], 0.5 + half)]
    elif kind == "corner":
        # Deck along the main wall for its part next to the house (B - D), back along the side wall
        # (full side minus short side: the house's depth)
        house_depth = abs(d["C"] - d["A"])
        path = [
            _along(c["corner"], c["main_end"], _share(d["B"] - d["D"], c["length_ft"], 0.5)),
            c["corner"],
            _along(c["corner"], c["side_end"], _share(house_depth, c.get("side_length_ft"), 0.6)),
        ]
    else:
        path = [c["left_back"], c["front_line"][0], c["front_line"][1], c["right_back"]]
    return {"kind": kind, "label": c["label"], "azimuth": c["azimuth"], "length_ft": c["length_ft"],
            "path": path, "elevation": analysis["elevation"]}


def parse_placement(data: dict, kind: str) -> dict:
    """A placement sent by the photo screen: the chosen candidate's label/azimuth/length, the deck's outline on
    the photo as a 2-4 point "path" (0-1 coordinates)."""
    try:
        if data.get("kind") != kind:
            raise ValueError("placement is for another deck shape")
        path = [[min(max(float(v), 0.0), 1.0) for v in pt][:2] for pt in data["path"]]
        if not 2 <= len(path) <= 4:
            raise ValueError("path needs 2 to 4 points")
        return {
            "kind": kind,
            "label": _wall_phrase(str(data["label"]))[:200],
            "azimuth": max(-75, min(75, round(float(data["azimuth"]) / 5) * 5)),
            "length_ft": float(data["length_ft"]) if data.get("length_ft") else None,
            "path": path,
            "elevation": round(min(max(float(data.get("elevation", 10)), 3), 35)),
        }
    except (KeyError, TypeError, ValueError) as err:
        raise PhotoError(f"Invalid deck placement: {err}") from err


def view_for(placement: dict | None) -> View:
    if not placement:
        return FALLBACK_VIEW
    return View(placement["label"], int(placement["azimuth"]), int(placement["elevation"]), placement.get("length_ft"), True)


def guide_image(jpeg: bytes, path: list[list[float]]) -> bytes:
    """The photo with the placement drawn on it as a thick magenta line."""
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = image.size
    points = [(x * w, y * h) for x, y in path]
    width = max(8, round(w * 0.012))
    draw = ImageDraw.Draw(image)
    draw.line(points, fill=GUIDE_COLOR, width=width, joint="curve")
    for x, y in (points[0], points[-1]):
        draw.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=GUIDE_COLOR)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _load_request(generated: Path, model_id: str) -> dict:
    meta_path = generated / f"{model_id}.json"
    if not meta_path.exists():
        raise FileNotFoundError(model_id)
    return json.loads(meta_path.read_text())["request"]


def prompt_info(generated: Path, model_id: str) -> dict:
    """Default templates plus what {deck} and {style} expand to, for the prompt editor."""
    request = _load_request(generated, model_id)
    return {
        "design": DESIGN_TEMPLATE,
        "restyle": RESTYLE_TEMPLATE,
        "deck": deck_description(request, make_layout(request["shape"], request["dims"])),
        "styles": {key: phrase for key, (_, phrase) in STYLES.items()},
    }


def ensure_references(generated: Path, model_id: str) -> dict[str, bytes]:
    """Front and top reference PNGs for a generated model, rendered once and cached next to its USDZ."""
    paths = {view: generated / f"{model_id}.{view}.png" for view in REFERENCE_VIEWS}
    if all(p.exists() for p in paths.values()):
        return {view: p.read_bytes() for view, p in paths.items()}
    request = _load_request(generated, model_id)
    layout = make_layout(request["shape"], request["dims"])
    images = render.reference_images(generated / f"{model_id}.usdz", layout, request["height_ft"] * FT, REFERENCE_VIEWS)
    for view, data in images.items():
        _write_atomic(paths[view], data)
    return images


def matched_reference(generated: Path, model_id: str, azimuth: int, elevation: int) -> bytes:
    """The deck rendered from a photo's viewpoint; cached per model and (rounded) angle."""
    path = generated / f"{model_id}.view{azimuth:+d}_{elevation}.png"
    if path.exists():
        return path.read_bytes()
    request = _load_request(generated, model_id)
    layout = make_layout(request["shape"], request["dims"])
    view = (azimuth, elevation)
    data = render.reference_images(generated / f"{model_id}.usdz", layout, request["height_ft"] * FT, [view])[view]
    _write_atomic(path, data)
    return data


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)  # atomic, so a concurrent request never reads half a file


def prepare_photo(data: bytes) -> tuple[bytes, tuple[int, int]]:
    """Upright, RGB, at most MAX_PHOTO_DIMENSION on the long side, as JPEG."""
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as err:
        raise PhotoError("That file isn't a photo we can read.") from err
    scale = min(1.0, MAX_PHOTO_DIMENSION / max(image.size))
    if scale < 1:
        image = image.resize((math.floor(image.width * scale), math.floor(image.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), image.size


def _check(style: str, template: str | None) -> None:
    if style not in STYLES:
        raise PhotoError(f"Unknown style {style}.")
    if template is not None and len(template) > MAX_TEMPLATE_CHARS:
        raise PhotoError("That prompt is too long.")


def _gray_grid(image: Image.Image) -> np.ndarray:
    """Luminance on a coarse grid, normalized so a global brightness or contrast shift doesn't count."""
    cells = np.asarray(image.convert("L").resize(PIXEL_GRID, Image.BOX), dtype=np.float32)
    return (cells - cells.mean()) / (cells.std() + 1e-6)


def _magenta_share(rgb: np.ndarray) -> float:
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return float(((r > 170) & (b > 120) & (r - g > 110) & (b - g > 70)).mean())


def _deck_area(path: list[list[float]] | None) -> tuple[float, float, float, float]:
    """Where the deck may cover the photo (x0, y0, x1, y1, as image fractions): around and below the
    guide line, since the deck comes toward the camera and its railing rises above the line."""
    if not path:
        return DECK_AREA_FALLBACK
    xs, ys = [p[0] for p in path], [p[1] for p in path]
    pad = max(0.15, (max(xs) - min(xs)) * 0.35)
    return max(0.0, min(xs) - pad), max(0.0, min(ys) - 0.3), min(1.0, max(xs) + pad), 1.0


def pixel_check(photo_jpeg: bytes, result: bytes, path: list[list[float]] | None) -> dict:
    """Checks that need no model: the guide's magenta left in the result, and the photo changed
    outside where the deck can be (reframed, zoomed or recomposed). Returns the measurements and
    the problems found."""
    photo = Image.open(io.BytesIO(photo_jpeg)).convert("RGB")
    # The result comes back at the closest supported aspect ratio; compare it at the photo's
    out = Image.open(io.BytesIO(result)).convert("RGB").resize(photo.size, Image.BILINEAR)
    small = (PIXEL_SIZE, round(PIXEL_SIZE * photo.height / photo.width))
    magenta = _magenta_share(np.asarray(out.resize(small, Image.BOX))) - _magenta_share(np.asarray(photo.resize(small, Image.BOX)))

    diff = np.abs(_gray_grid(photo) - _gray_grid(out))
    x0, y0, x1, y1 = _deck_area(path)
    gh, gw = diff.shape
    outside = np.ones_like(diff, dtype=bool)
    outside[math.floor(y0 * gh):math.ceil(y1 * gh), math.floor(x0 * gw):math.ceil(x1 * gw)] = False
    changed = float((diff[outside] > PIXEL_CELL_CHANGED).mean()) if outside.any() else 0.0

    problems = []
    if magenta > MAGENTA_MAX_SHARE:
        problems.append("the guide line is still visible")
    if changed > OUTSIDE_MAX_CHANGED:
        problems.append("the photo was reframed or changed outside the deck")
    return {"magenta": round(magenta, 5), "outside_changed": round(changed, 3), "problems": problems}


def _judge(photo_jpeg: bytes, result: bytes, wall: str, stairs: str, height_ft: float,
           path: list[list[float]] | None) -> dict | None:
    """Check a design result: the pixel checks, then (if those pass) the text model. Returns
    {"matches", "severity" (0 ok, 1 off-design, 2 photo ruined), "problem"}, or None if the
    model's check itself failed (then the result is not held back)."""
    try:
        pixels = pixel_check(photo_jpeg, result, path)
    except Exception:  # an unreadable result fails the model check below too
        pixels = {"problems": []}
    if pixels["problems"]:
        return {"matches": False, "severity": 2, "problem": "; ".join(pixels["problems"])}
    try:
        result_jpeg, _ = prepare_photo(result)
        prompt = JUDGE_PROMPT.replace("{wall}", wall).replace("{stairs}", stairs).replace("{height}", f"{height_ft:g}")
        answer = gemini.analyze([prompt, (photo_jpeg, "image/jpeg"), (result_jpeg, "image/jpeg")], timeout=60)
        failed = [(text, severity) for field, (bad, text, severity) in JUDGE_FLAGS.items() if bool(answer.get(field, not bad)) == bad]
    except (gemini.GeminiError, PhotoError, TypeError, ValueError, AttributeError):
        return None
    problem = str(answer.get("problem") or "")[:200]
    if failed:
        # The model's own "matches" is not trusted over a flag it raised
        return {"matches": False, "severity": max(s for _, s in failed), "problem": problem or "; ".join(t for t, _ in failed)}
    if not answer.get("matches"):
        return {"matches": False, "severity": 1, "problem": problem or "it does not match the design"}
    return {"matches": True, "severity": 0, "problem": ""}


def design(
    generated: Path, model_id: str, photo: bytes, style: str, template: str | None = None, placement: dict | None = None
) -> tuple[bytes, str, View, dict | None, dict]:
    """The photo with the model's deck added, the viewpoint and placement used, and the check of the result:
    {"verified": True/False/None (check unavailable), "attempts": results generated, "problem": text}.
    `placement` is as sent by the photo screen (see parse_placement). Without one the deck goes on the
    recommended candidate (or, if none was found, a straight-on view without a guide).
    Raises FileNotFoundError, PhotoError or gemini.GeminiError (if every generation failed)."""
    _check(style, template)
    request = _load_request(generated, model_id)
    kind = KIND_FOR_SHAPE[request["shape"]]
    layout = make_layout(request["shape"], request["dims"])
    jpeg, (w, h) = prepare_photo(photo)
    if placement is None:
        placement = default_placement(kind, analyze_photo(jpeg, kind), request, layout)
    else:
        placement = parse_placement(placement, kind)
    view = view_for(placement)
    deck = deck_description(request, layout, view.wall_length_ft)
    refs = ensure_references(generated, model_id)
    images = {
        "photo": (jpeg, "image/jpeg"),
        "guide": (guide_image(jpeg, placement["path"]), "image/jpeg") if placement else None,
        "matched": (matched_reference(generated, model_id, view.azimuth, view.elevation), "image/png"),
        "top": (refs["top"], "image/png"),
        "front": (refs["front"], "image/png"),
    }
    parts = design_parts(template or DESIGN_TEMPLATE, deck, style, view, images, kind)
    aspect = gemini.closest_aspect_ratio(w, h)
    stairs = _stairs_line(layout, request.get("stair_edge")).replace("{wall}", view.wall)

    def attempt():
        data, mime = gemini.edit_image(parts, aspect)
        return data, mime, _judge(jpeg, data, view.wall, stairs, request["height_ft"], placement and placement["path"])

    results, error = [], None
    for _ in range(MAX_ROUNDS):
        # Not a `with` block: return as soon as one passes, without waiting for the other generation
        pool = ThreadPoolExecutor(CANDIDATES_PER_ROUND)
        pending = {pool.submit(attempt) for _ in range(CANDIDATES_PER_ROUND)}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        data, mime, check = future.result()
                    except gemini.GeminiError as err:
                        error = err
                        continue
                    results.append((data, mime, check))
                    if check is None or check["matches"]:
                        verified = None if check is None else True
                        return data, mime, view, placement, {"verified": verified, "attempts": len(results), "problem": ""}
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    if not results:
        raise error or gemini.GeminiError("Generation failed.")
    # Nothing passed: prefer a result that is only off-design over one that spoiled the photo
    data, mime, check = min(results, key=lambda r: r[2]["severity"])
    return data, mime, view, placement, {"verified": False, "attempts": len(results), "problem": check["problem"]}


def restyle(master: bytes, style: str, template: str | None = None) -> tuple[bytes, str]:
    """A design result with only the deck's material changed."""
    _check(style, template)
    jpeg, (w, h) = prepare_photo(master)
    return gemini.edit_image([_fill(template or RESTYLE_TEMPLATE, "", style, FALLBACK_VIEW), (jpeg, "image/jpeg")], gemini.closest_aspect_ratio(w, h))

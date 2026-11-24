"""Extra inputs for experiments: the photo as the canvas prepares it, a reference picked by the photo's
camera angle, and a labelled plan of the deck."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .describe import FURNITURE_NAMES, Frame, _ft

HERE = Path(__file__).resolve().parent
ANGLES_AZ = [-60, -40, -20, 0, 20, 40, 60]
ANGLES_EL = [12, 30]


def prep_like_canvas(data: bytes, long_edge: int = 1536, quality: int = 85) -> bytes:
    """What the canvas uploads: upright, long edge 1536, JPEG 85 (prepareDeckPhoto)."""
    from PIL import ImageOps

    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    scale = min(1.0, long_edge / max(im.size))
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality)
    return out.getvalue()


def photo_angle(name: str, jpeg: bytes) -> dict:
    """The camera's angle to the wall a deck would go on, from the same text model the analyze step used;
    cached per photo name under out/ so repeated runs agree."""
    cache_path = HERE / "out" / "angles.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if name not in cache:
        from app import photo as analysis

        result = analysis.analyze_photo(jpeg, "wall")
        best = next((c for c in result["candidates"] if c["recommended"]), None)
        cache[name] = {"azimuth": best["azimuth"] if best else 0, "elevation": result["elevation"],
                       "label": best["label"] if best else None, "analyzed": result["analyzed"]}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
    return cache[name]


def matched_ref(style: str, angle: dict) -> bytes:
    az = min(ANGLES_AZ, key=lambda a: abs(a - angle["azimuth"]))
    el = min(ANGLES_EL, key=lambda e: abs(e - angle["elevation"]))
    return (HERE / "refs" / style / "angles" / f"az{az}_el{el}.png").read_bytes()


def _font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def plan_png(deck: dict, width: int = 1280, height: int = 960) -> bytes:
    """The deck from above, house at the top, every edge and piece labelled in feet: a drawing, not a scene."""
    f = Frame(deck)
    pts = f.points
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    w, d = max(xs) - min(xs), max(zs) - min(zs)
    margin = 150
    stairs_ft = max([st["risers"] * 0.6 for st in deck.get("stairs", [])] or [0])
    scale = min((width - 2 * margin) / w, (height - margin - 60 - 120) / (d + stairs_ft))
    ox = (width - w * scale) / 2 - min(xs) * scale
    oy = margin + 60 - min(zs) * scale

    def at(x, z):
        return (ox + x * scale, oy + z * scale)

    im = Image.new("RGB", (width, height), (250, 250, 247))
    dr = ImageDraw.Draw(im)
    big, mid, small = _font(34), _font(26), _font(22)

    # The house along the main wall
    top_y = at(0, 0)[1]
    dr.rectangle([at(min(xs) - 3, 0)[0], top_y - 60, at(max(xs) + 3, 0)[0], top_y], fill=(120, 120, 120))
    dr.text((width / 2, top_y - 30), "HOUSE WALL", fill="white", font=big, anchor="mm")

    dr.polygon([at(*p) for p in pts], fill=(214, 190, 150), outline=(60, 40, 20), width=5)
    n = len(pts)
    house = {e if isinstance(e, int) else e[0] for e in deck.get("ledger", []) + deck.get("house", [])}
    railed = {e for g in deck.get("railing", []) for e in g["edges"]}
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if i in railed and i not in house:
            dr.line([at(*a), at(*b)], fill=(255, 255, 255), width=9)
            dr.line([at(*a), at(*b)], fill=(60, 40, 20), width=3)
        length = math.dist(a, b)
        mx, mz = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        nx, nz = (b[1] - a[1]) / length, -(b[0] - a[0]) / length  # outward for a clockwise frame outline
        cx, cz = sum(xs) / n, sum(zs) / n
        if (mx - cx) * nx + (mz - cz) * nz < 0:
            nx, nz = -nx, -nz
        # Labels sit just inside the edge, clear of the stairs and the house.
        # Side edges get theirs outside, edges facing the yard just inside (the stairs hang below).
        out_ft = 1.4 if abs(nx) > abs(nz) else -1.0
        lx, ly = at(mx + nx * out_ft, mz + nz * out_ft)
        label = f"{_ft(length)} ft along the house" if i in house else f"{_ft(length)} ft"
        dr.text((lx, ly), label, fill=(20, 20, 20), font=mid, anchor="mm")

    for s in deck.get("stairs", []):
        outline = deck["outline"]
        a, b = outline[s["edge"]], outline[(s["edge"] + 1) % len(outline)]
        length = math.dist(a, b)
        p0 = f.map((a[0] + (b[0] - a[0]) * s["from"] / length, a[1] + (b[1] - a[1]) * s["from"] / length))
        p1 = f.map((a[0] + (b[0] - a[0]) * (s["from"] + s["width"]) / length,
                    a[1] + (b[1] - a[1]) * (s["from"] + s["width"]) / length))
        run = s["risers"] * 0.6
        x0, x1 = sorted([p0[0], p1[0]])
        z0 = max(p0[1], p1[1])
        dr.rectangle([at(x0, z0), at(x1, z0 + run)], fill=(190, 160, 120), outline=(60, 40, 20), width=3)
        for k in range(1, s["risers"]):
            yy = at(0, z0 + run * k / s["risers"])[1]
            dr.line([at(x0, 0)[0], yy, at(x1, 0)[0], yy], fill=(60, 40, 20), width=2)
        dr.text((at(x1, 0)[0] + 16, at(0, z0 + run / 2)[1]), f"STAIRS DOWN ({s['risers']} steps)",
                fill=(20, 20, 20), font=small, anchor="lm")

    for p in deck.get("furniture", []):
        x, z = f.map(p["center"])
        hw, hd = p["width"] / 2, p["depth"] / 2
        name = FURNITURE_NAMES.get(p["kind"], p["kind"]).upper()
        box = [at(x - hw, z - hd), at(x + hw, z + hd)]
        if "round" in name or "FIRE PIT" in name:
            dr.ellipse(box, fill=(150, 150, 150), outline=(40, 40, 40), width=3)
        else:
            dr.rectangle(box, fill=(120, 170, 190), outline=(40, 40, 40), width=3)
        dr.text(at(x, z), name.replace(" ", "\n", 1) if len(name) > 12 else name, fill=(10, 10, 10), font=small,
                anchor="mm", align="center")

    dr.text((width / 2, height - 28), "YARD  (seen from above; left and right as seen from the yard facing the house)",
            fill=(20, 20, 20), font=small, anchor="mm")
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def engine_image(kind: str, style: str, photo: str, jpeg: bytes) -> bytes:
    if kind == "composite-clean":
        return engine_image("composite", style, photo, cleaned(photo, jpeg))
    """The engine's deck render from the photo's solved camera (tuning/engine/<style>/<photo>.png, transparent, the
    photo's pixel size): "render" = on a plain grey ground, "composite" = pasted onto the photo."""
    deck = Image.open(HERE / "engine" / style / f"{photo}.png").convert("RGBA")
    base = Image.open(io.BytesIO(jpeg)).convert("RGBA")
    if deck.size != base.size:
        deck = deck.resize(base.size, Image.LANCZOS)
    under = base if kind == "composite" else Image.new("RGBA", base.size, (200, 200, 200, 255))
    out = io.BytesIO()
    Image.alpha_composite(under, deck).convert("RGB").save(out, "JPEG", quality=92)
    return out.getvalue()


CLEAN_PROMPT = """\
Edit this photo of a house: remove the deck, patio or porch that is built against the house, with all of its floor, \
stairs, railings, posts, skirting and the furniture and pots standing on it. Where it was, show the bare foot of the \
house wall and the ground in front of it as lawn, matching the rest of the yard. The ground stays at its real height: \
where the deck's floor was above the ground, the wall's siding and foundation now run down below the door to the ground, \
and the door stays high in the wall, with no steps. Keep everything else exactly as it is: \
the same camera, framing and aspect ratio, the same house with its walls, doors (a door that opened onto the deck stays \
exactly where it is, at the same height), windows, roof and colours, the same trees, fence, sky and light. Do not add \
anything. Output the edited photo."""


def cleaned(photo: str, jpeg: bytes, model: str = "gemini-3.1-flash-image") -> bytes:
    """The photo with its existing deck or patio removed (one Gemini edit, cached under out/clean/), scaled back to the
    photo's pixel size so the engine render still lines up. Photos the wall analysis found no deck in come back as is."""
    from . import camera, providers

    marks = json.loads(camera.CACHE.read_text()).get(photo, {}) if camera.CACHE.exists() else {}
    if not marks.get("existing_deck"):
        return jpeg
    path = HERE / "out" / "clean" / f"{photo}.jpg"
    if not path.exists():
        size = Image.open(io.BytesIO(jpeg)).size
        image, _ = providers.edit(model, [CLEAN_PROMPT, ("photo", jpeg)], size)
        im = Image.open(io.BytesIO(image)).convert("RGB").resize(size, Image.LANCZOS)
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path, "JPEG", quality=92)
    return path.read_bytes()


def above_ground(deck_png: Path, ground_png: Path, tolerance: float = 40.0) -> Image.Image:
    """The deck render without what lies under the ground (pier footings): the same camera rendered once without the
    ground (transparent around the deck) and once with it. A deck pixel the ground render shows differently is hidden
    by the ground there, so it goes. Pixels past the ground disc, or on its faded rim, stay."""
    import numpy as np

    deck = np.asarray(Image.open(deck_png).convert("RGBA")).astype(np.float32)
    ground = np.asarray(Image.open(ground_png).convert("RGBA")).astype(np.float32)
    diff = np.linalg.norm(deck[..., :3] - ground[..., :3], axis=2)
    buried = (deck[..., 3] > 0) & (ground[..., 3] > 200) & (diff > tolerance)
    try:
        import cv2

        buried = cv2.morphologyEx(buried.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    except ImportError:
        pass
    out = deck.copy()
    out[buried, 3] = 0
    try:  # specks left where grass and concrete happened to match: drop tiny islands of the deck
        import cv2

        n, labels, stats, _ = cv2.connectedComponentsWithStats((out[..., 3] > 0).astype(np.uint8), connectivity=8)
        small = np.isin(labels, np.where(stats[:, cv2.CC_STAT_AREA] < 40)[0]) & (labels > 0)
        out[small, 3] = 0
    except ImportError:
        pass
    return Image.fromarray(out.astype(np.uint8), "RGBA")

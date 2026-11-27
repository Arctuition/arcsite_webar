"""The photo cleaned first, everything after built on it (experiment, 11-25).

  1+2. one image-model call: is there a deck / patio / porch against the house? if so remove it (DECK), else NONE and
       the photo is kept as it was
  3.   the wall analysed and the camera solved on the cleaned photo (and, to compare, on the original)
  4.   the engine renders the deck from that camera; composited locally with its shadow and the photo's colour
  5.   the image model polishes the composite instead of editing the photo

  .venv/bin/python -m tuning.pipeline2 detect [--repeat 3]      # step 1+2 on every photo, under out/p2/detect/
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path

from PIL import Image

from . import check, inputs, providers

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "p2"
MODEL = "gemini-3.1-flash-image"

DETECT_CLEAN_PROMPT = """\
Look at this photo of a house. Is there a deck, patio, porch platform, raised terrace or paved sitting area built \
against the house (with its floor, stairs, railings, posts or skirting)? A plain path, a single doorstep or a lawn \
does not count.

If there is one: edit the photo to remove it, with all of its floor, stairs, railings, posts, skirting and the \
furniture and pots standing on it. Where it was, show the bare foot of the house wall and the ground in front of it as \
lawn, matching the rest of the yard. The ground stays at its real height: where the floor was above the ground, the \
wall's siding and foundation now run down below the door to the ground, and the door stays high in the wall, with no \
steps. Keep everything else exactly as it is: the same camera, framing and aspect ratio, the same house with its walls, \
doors (each exactly where it is, at the same height), windows, roof and colours, the same trees, fence, sky and light. \
Do not add anything.

If there is none: output the photo unchanged.

Begin your text answer with one word, DECK if you removed one or NONE if there was none, then one short sentence \
saying what it was."""


def _verdict(text: str | None) -> str | None:
    m = re.search(r"\b(DECK|NONE)\b", text or "")
    return m.group(1) if m else None


class NoCleanImage(providers.ProviderError):
    """The model said the photo has a deck but sent no picture without it (about one call in seven on deck photos)."""


def detect_clean(jpeg: bytes, model: str = MODEL) -> dict:
    """{verdict: DECK|NONE|None, text, image: the cleaned photo (the photo itself unless DECK), seconds}."""
    size = Image.open(io.BytesIO(jpeg)).size
    t0 = time.time()
    image, text, reason = providers.gemini_answer(model, [DETECT_CLEAN_PROMPT, ("photo", jpeg)], size)
    verdict = _verdict(text)
    if verdict == "DECK" and image is None:
        raise NoCleanImage(f"DECK but no image ({reason}): {text}"[:400])
    out = jpeg
    if verdict == "DECK":
        im = Image.open(io.BytesIO(image)).convert("RGB").resize(size, Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        out = buf.getvalue()
    return {"verdict": verdict, "text": text, "image": out, "raw": image, "seconds": round(time.time() - t0, 1)}


def photo_jpeg(name: str) -> bytes:
    return inputs.prep_like_canvas((HERE / "photos" / f"{name}.jpg").read_bytes(), 2048, 92)


def cmd_detect(args) -> None:
    import concurrent.futures as cf

    d = OUT / "detect"
    d.mkdir(parents=True, exist_ok=True)
    jobs = [(p, k) for p in args.photos for k in range(args.repeat)]
    blank = None

    def work(job):
        p, k = job
        jpeg = photo_jpeg(p)
        r = detect_clean(jpeg)
        (d / f"{p}__{k + 1}.jpg").write_bytes(r["image"])
        kept = None  # how much of the photo a returned image kept, everywhere: ~1 = untouched
        if r["raw"]:
            (d / f"{p}__{k + 1}.raw.jpg").write_bytes(providers.as_jpeg(r["raw"]))
            size = Image.open(io.BytesIO(jpeg)).size
            kept = round(check.drift(providers.as_jpeg(r["raw"]), jpeg, Image.new("L", size, 0)), 2)
        return {"photo": p, "k": k + 1, "verdict": r["verdict"], "text": r["text"], "seconds": r["seconds"],
                "raw_kept": kept}

    with cf.ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(work, jobs))
    for r in rows:
        print(f"{r['photo']:14} #{r['k']} {r['verdict']}  kept {r['raw_kept']}  {r['seconds']:5.1f}s  {(r['text'] or '')[:110]!r}")
    (d / "detect.json").write_text(json.dumps(rows, indent=1))


def solve_wall(jpeg: bytes, deck: dict, wall: dict | None, marks: dict | None = None) -> dict:
    """Step 3: the wall's marks (asked afresh unless given), the camera solved from them and the deck placed."""
    from . import camera

    marks = marks if marks is not None else camera.mark_wall(jpeg)
    w, h = Image.open(io.BytesIO(jpeg)).size
    cam = camera.solve(marks, w, h, camera.detect_lines(jpeg))
    cam["placement"] = camera.place(deck, cam, wall)
    cam["engine"] = camera.engine_camera(deck, cam, cam["placement"])
    cam["marks"] = marks
    return cam


def cmd_wall(args) -> None:
    """Step 3 on the original photo and on the cleaned one, [repeat] times each: how far apart, and how steady."""
    import concurrent.futures as cf
    from . import camera

    deck = json.loads((HERE / "deck.json").read_text())
    wall = json.loads((HERE / "engine" / "wall.json").read_text())
    d = OUT / "wall"
    d.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in args.photos:
        cleaned = OUT / "detect" / f"{p}__{args.clean}.jpg"
        sources = {"original": photo_jpeg(p)}
        if cleaned.exists() and cleaned.read_bytes() != sources["original"]:
            sources["cleaned"] = cleaned.read_bytes()
        jobs += [(p, src, jpeg, k) for src, jpeg in sources.items() for k in range(args.repeat)]

    def work(job):
        p, src, jpeg, k = job
        try:
            cam = solve_wall(jpeg, deck, wall)
        except Exception as err:  # noqa: BLE001 -- an unsolvable answer is a result too
            return {"photo": p, "src": src, "k": k + 1, "error": f"{type(err).__name__}: {err}"[:200]}
        (d / f"{p}__{src}__{k + 1}.jpg").write_bytes(camera.overlay(jpeg, cam["marks"], cam, deck, cam["placement"]))
        keep = {key: cam[key] for key in cam if key not in ("snapped_lines",)}
        (d / f"{p}__{src}__{k + 1}.json").write_text(json.dumps(keep, indent=1, default=float))
        return {"photo": p, "src": src, "k": k + 1, "base": "ground" if cam["marks"].get("base_line") else
                ("deck surface" if cam["marks"].get("existing_deck") else "none"),
                **{key: round(float(cam[key]), 1) for key in ("hfov_deg", "yaw_deg", "pitch_deg", "camera_height_ft",
                                                               "wall_length_ft")},
                "x_ft": round(cam["placement"]["x_ft"], 1), "fit_cost": round(float(cam["fit_cost"]), 5),
                "problems": camera.fit_problems(deck, cam, cam["placement"])}

    with cf.ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(work, jobs))
    for r in rows:
        print(r)
    (d / "wall.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------- step 4: the local composite

ENGINE = HERE / "engine" / "p2"  # <photo>.deck.png (house + ground hidden), .ground.png (house hidden), .bare.png
#                                   (house and every deck part hidden): the engine's renders from out/p2/cameras.json


def clean_of(photo: str) -> bytes:
    """The photo step 1+2 left: detect run #1's cleaned photo, or the photo itself."""
    path = OUT / "detect" / f"{photo}__1.jpg"
    return path.read_bytes() if path.exists() else photo_jpeg(photo)


def shadow_map(ground: Image.Image, bare: Image.Image, deck_alpha, strength: float = 0.85):
    """How much darker the engine's ground is with the deck than without it (its shadow), 1 where it is not."""
    import cv2
    import numpy as np

    g = np.asarray(ground.convert("RGBA")).astype(np.float32)
    b = np.asarray(bare.convert("RGBA")).astype(np.float32)
    lum = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2] + 1.0
    ratio = np.ones(g.shape[:2], np.float32)
    on = (g[..., 3] > 200) & (b[..., 3] > 200) & (np.asarray(deck_alpha) < 8)
    ratio[on] = np.clip(lum(g)[on] / lum(b)[on], 0.25, 1.0)
    ratio = cv2.GaussianBlur(ratio, (0, 0), 2.0)
    return ratio ** strength


def match_colour(deck: Image.Image, photo: Image.Image, amount: float = 0.5, ring: float = 0.08,
                 gain: tuple[float, float] = (0.8, 1.25), shift: float = 15.0) -> Image.Image:
    """The deck render's brightness moved [amount] of the way to the photo's around it (a ring [ring] of the width
    wide): mean and spread of Lab L, the spread's gain held to [gain] and the mean's shift to [shift]. The hue is left
    alone. Matching a and b to the whole photo (until 11-27) scaled them up to 3.3x, the render's near-grey a and b
    against the photo's lawn and sky: white rails came out mint green and stair risers olive and brick red."""
    import cv2
    import numpy as np

    d = np.asarray(deck.convert("RGBA"))
    a = d[..., 3] > 128
    if a.sum() < 50:
        return deck
    dl = cv2.cvtColor(np.ascontiguousarray(d[..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
    pl = cv2.cvtColor(np.asarray(photo.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    k = max(3, int(ring * d.shape[1]) | 1)
    around = (cv2.dilate(a.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & ~a
    ref = pl[..., 0][around] if around.sum() >= 50 else pl[..., 0].ravel()
    dm, ds = dl[..., 0][a].mean(), dl[..., 0][a].std() + 1e-3
    pm, ps = ref.mean(), ref.std() + 1e-3
    tm = dm + float(np.clip((pm - dm) * amount, -shift, shift))
    ts = ds * float(np.clip((ds + (ps - ds) * amount) / ds, *gain))
    out = dl.copy()
    out[..., 0] = (dl[..., 0] - dm) * (ts / ds) + tm
    rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(np.dstack([rgb, d[..., 3]]), "RGBA")


def _jpeg(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def shadow_from_ground(ground: Image.Image, deck_alpha, strength: float = 0.85, k: int = 61, cut: float = 0.75,
                       near: float = 0.02):
    """shadow_map without the bare render (the canvas sends only the deck and the deck with the ground): the local
    brightest of the engine's ground is the unshadowed ground, and what is over a quarter darker than that is the deck's
    shadow (vs shadow_map, 11-25: the same area to IoU 0.70-0.86). Only shadow touching the deck (within [near] of the
    width) is kept: on a drawing whose ground is textured grass (11-27) its dark blades came out as ~350 bits of shadow
    over the lawn, and the paste-back, which takes the model's pixels wherever there is shadow, took 40% of the photo
    from the model (now 2%; IoU vs shadow_map unchanged on the four harness photos, existing-deck 0.43 -> 0.84)."""
    import cv2
    import numpy as np

    g = np.asarray(ground.convert("RGBA")).astype(np.float32)
    lum = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2] + 1.0
    deck_on = np.asarray(deck_alpha) >= 8
    on = (g[..., 3] > 200) & ~deck_on
    env = cv2.GaussianBlur(cv2.dilate(np.where(on, lum, 0).astype(np.float32), np.ones((k, k), np.uint8)), (0, 0), k / 3)
    ratio = np.ones(lum.shape, np.float32)
    ratio[on] = np.clip(lum[on] / np.maximum(env[on], 1), 0.25, 1.0)
    shadow = cv2.morphologyEx((ratio < cut).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    reach = max(3, int(near * lum.shape[1]) | 1)
    by_deck = cv2.dilate(deck_on.astype(np.uint8), np.ones((reach, reach), np.uint8))
    n, labels = cv2.connectedComponents(shadow, connectivity=8)
    touching = np.unique(labels[(by_deck > 0) & (shadow > 0)])
    shadow = np.isin(labels, touching[touching > 0])
    ratio = np.where(shadow, ratio, 1.0).astype(np.float32)
    return cv2.GaussianBlur(ratio, (0, 0), 2.0) ** strength


def restore_photo(photo: bytes, clean: bytes, box=None, threshold: float = 10.0, grow: float = 0.015,
                  feather: float = 0.008, join: float = 0.025) -> tuple[bytes, dict]:
    """The photo with the cleaned photo only where the old deck was: the image model redraws the whole picture at ~1K
    when it removes a deck (an order of magnitude less detail than the photo, after upscaling) and changes things
    away from it too (a flower bed dropped, a fence carried on). Where it changed the photo is found by difference,
    after moving the cleaned photo onto the photo's pixels; of those changes only the ones that touch the old deck's
    box (marks.existing_deck.box: [ymin, xmin, ymax, xmax] in 0-1000) are taken, or without a box the largest one
    and what touches it, grown by [grow] and feathered by [feather] of the width. {share, moved, ...} says how much."""
    import cv2
    import numpy as np

    ph = np.asarray(Image.open(io.BytesIO(photo)).convert("RGB"))
    h, w = ph.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    if box:
        y0, x0, y1, x1 = [float(v) / 1000 for v in box]
        keep[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
    aligned, info = register(clean, photo, Image.fromarray(keep))
    cl = np.asarray(Image.open(io.BytesIO(aligned)).convert("RGB").resize((w, h), Image.LANCZOS))
    lab = lambda a: cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), 2.0), cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = lab(cl) - lab(ph)
    diff -= np.median(diff.reshape(-1, 3), axis=0)  # an overall change of tone is not a change of things
    changed = (np.linalg.norm(diff, axis=2) > threshold).astype(np.uint8)
    s = max(3, int(0.004 * w) | 1)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((s, s), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((3 * s, 3 * s), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
    if n <= 1:
        return photo, {**info, "share": 0.0}
    if box:
        g = max(3, int(0.05 * w) | 1)
        near = cv2.dilate(keep, np.ones((g, g), np.uint8)) > 0
        ids = np.unique(labels[near & (changed > 0)])
    else:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        g = max(3, int(0.05 * w) | 1)
        near = cv2.dilate((labels == big).astype(np.uint8), np.ones((g, g), np.uint8)) > 0
        ids = np.unique(labels[near & (changed > 0)])
    m = np.isin(labels, ids[ids > 0]).astype(np.uint8)
    # one region, holes and all: a chair on the old deck only partly changed (its top against dark glass) would
    # otherwise be left half there, a ghost through the feathered edge
    j = max(3, int(join * w) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((j, j), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = cv2.drawContours(np.zeros_like(m), contours, -1, 1, cv2.FILLED)
    g = max(3, int(grow * w) | 1)
    m = cv2.dilate(m, np.ones((g, g), np.uint8)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, feather * w))[..., None]
    out = cl.astype(np.float32) * m + ph.astype(np.float32) * (1 - m)
    info = {**info, "share": round(float((m > 0.5).mean()), 3)}
    return _jpeg(Image.fromarray(out.clip(0, 255).astype(np.uint8))), info


def compose_images(clean_jpeg: bytes, deck: Image.Image, ground: Image.Image, bare: Image.Image | None = None) -> dict:
    """Step 4: {plain: the deck render (above the ground, RGBA, photo-sized) on the clean photo, rich: with its shadow
    darkening the photo and the photo's colour, alpha: the deck's coverage, shadow: the shadow map}."""
    import numpy as np

    clean = Image.open(io.BytesIO(clean_jpeg)).convert("RGBA")
    alpha = deck.getchannel("A")
    plain = Image.alpha_composite(clean, deck)
    shade = shadow_map(ground, bare, alpha) if bare is not None else shadow_from_ground(ground, alpha)
    shaded = np.asarray(clean).astype(np.float32)
    shaded[..., :3] *= shade[..., None]
    rich = Image.alpha_composite(Image.fromarray(shaded.clip(0, 255).astype(np.uint8), "RGBA"),
                                 match_colour(deck, clean))
    return {"plain": _jpeg(plain), "rich": _jpeg(rich), "alpha": alpha, "shadow": shade}


def compose(photo: str) -> dict:
    """Step 4 for a harness [photo], from its engine renders under ENGINE; with clean: the clean photo."""
    clean = clean_of(photo)
    size = Image.open(io.BytesIO(clean)).size
    load = lambda kind: Image.open(ENGINE / f"{photo}.{kind}.png").convert("RGBA").resize(size, Image.LANCZOS)
    deck = inputs.above_ground(load("deck"), load("ground"))
    return {**compose_images(clean, deck, load("ground"), load("bare")), "clean": clean}


def register(output: bytes, reference: bytes, alpha) -> tuple[bytes, dict]:
    """The output moved onto the reference's pixels: a model that was asked to keep the framing still zooms or shifts it
    a little (the front photo: up to 36%), and a paste-back of a moved output puts the deck off its place. The move is a
    similarity (scale, turn, shift) fitted on features away from the deck, which the output is meant to change; where
    the moved output does not reach, the reference shows."""
    import cv2
    import numpy as np

    ref = np.asarray(Image.open(io.BytesIO(reference)).convert("RGB"))
    out = np.asarray(Image.open(io.BytesIO(output)).convert("RGB").resize((ref.shape[1], ref.shape[0]), Image.LANCZOS))
    keep = (cv2.dilate((np.asarray(alpha) > 8).astype(np.uint8), np.ones((31, 31), np.uint8)) == 0).astype(np.uint8) * 255
    orb = cv2.ORB_create(5000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), keep)
    kb, db = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), keep)
    info = {"moved": False}
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return output, info
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return output, info
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl.sum() < 15:
        return output, info
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    shift = float(np.hypot(M[0, 2], M[1, 2]))
    info = {"scale": round(scale, 3), "turn_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2),
            "shift_px": round(shift, 1), "inliers": int(inl.sum())}
    if abs(scale - 1) < 0.004 and shift < 2:
        return output, info
    h, w = ref.shape[:2]
    moved = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    reach = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    moved = np.where((reach > 250)[..., None], moved, ref)
    info["moved"] = True
    return _jpeg(Image.fromarray(moved)), info


def whole_objects(out, base, m, band: int = 9, min_area: int = 80, max_share: float = 0.04):
    """The paste mask grown to take in, whole, anything the output added that its edge would cut through (a shrub the
    model put before the deck): a cut plant reads as a flower floating on the lawn. Things it added clear of the deck
    stay out, as the photo has them."""
    import cv2
    import numpy as np

    lab = lambda a: cv2.cvtColor(cv2.GaussianBlur(a, (0, 0), 1.5), cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.linalg.norm(lab(out) - lab(base), axis=2)
    diff -= cv2.GaussianBlur(diff, (0, 0), 25)  # an overall change of light is not an object
    added = (diff > 18).astype(np.uint8)
    added = cv2.morphologyEx(added, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    added = cv2.morphologyEx(added, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    edge = cv2.dilate(m, np.ones((band, band), np.uint8)) - cv2.erode(m, np.ones((band, band), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(added, connectivity=8)
    grown = m.copy()
    limit = max_share * m.size
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > limit:
            continue
        comp = labels == i
        inside = int((m[comp] > 0).sum())
        outside = area - inside
        # a plant cut by the edge is mostly in the mask already; a lawn the model re-grew is mostly out of it
        if (edge[comp] > 0).any() and 0 < outside <= inside:
            grown[comp] = 1
    return cv2.dilate(grown, np.ones((5, 5), np.uint8))


def paste_back(output: bytes, clean: bytes, alpha, shade, grow: int = 25, feather: float = 8.0,
               objects: bool = False) -> bytes:
    """The output only where the deck and its shadow are (grown and feathered), the clean photo everywhere else; with
    [objects], also whatever the output added across that edge, whole."""
    import cv2
    import numpy as np

    base = Image.open(io.BytesIO(clean)).convert("RGB")
    out = Image.open(io.BytesIO(output)).convert("RGB").resize(base.size, Image.LANCZOS)
    m = ((np.asarray(alpha) > 8) | (shade < 0.97)).astype(np.uint8)
    m = cv2.dilate(m, np.ones((grow, grow), np.uint8))
    if objects:
        m = whole_objects(np.asarray(out), np.asarray(base), m)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), feather)[..., None]
    mixed = np.asarray(out).astype(np.float32) * m + np.asarray(base).astype(np.float32) * (1 - m)
    return _jpeg(Image.fromarray(mixed.clip(0, 255).astype(np.uint8)))


def cmd_compose(args) -> None:
    d = OUT / "compose"
    d.mkdir(parents=True, exist_ok=True)
    for p in args.photos:
        c = compose(p)
        (d / f"{p}.plain.jpg").write_bytes(c["plain"])
        (d / f"{p}.rich.jpg").write_bytes(c["rich"])
        print(p, "ok")


# ---------------------------------------------------------------- step 5: the polish

VARIANTS = {  # name: (template, the composite it gets, paste back)
    "base": ("auto-v10", "plain", False),
    "g1": ("p2g1", "rich", False),
    "g2": ("p2g2", "rich", False),
    "l1": ("p2l1", "rich", True),
    "l2": ("p2l2", "rich", True),  # l1 telling the model harder to add no plants; footings concrete
}


def furniture_block(furniture: str) -> str:
    return ("\nFURNITURE ON THE DECK\n" + furniture + "\nKeep each piece as exactly that kind of object, where the "
            "picture shows it. Do not swap any piece for planters, tables, chairs or sofas.\n")


def cmd_polish(args) -> None:
    import concurrent.futures as cf
    from . import camera
    from .describe import describe
    from .run import FLOOR_SEEN_FT, build_parts

    deck = json.loads((HERE / "deck.json").read_text())
    cams = json.loads((OUT / "cameras.json").read_text())
    looks = json.loads((HERE / "styles.json").read_text())
    deck_text, furniture = describe(deck)
    style = "english-country-cottage"
    run = OUT / "polish" / args.name
    run.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in args.photos:
        c = compose(p)
        cam_dir = OUT / "wall" / f"{p}__{cams[p]['src']}__{cams[p]['k']}.json"
        cam = json.loads(cam_dir.read_text())
        stairs = camera.stairs_note(deck, cam, cams[p]["placement"])
        seen = cams[p]["camera_height_ft"] > deck["height_in"] / 12 + FLOOR_SEEN_FT
        for v in args.variants:
            template, comp_kind, paste = VARIANTS[v]
            if template == "auto-v10":
                template = "v10k" if seen else "v10j"
            text = (HERE / "prompts" / f"{template}.txt").read_text()
            if not template.startswith("v10"):
                text = text.replace("{furniture}", furniture_block(furniture) if seen else "")
            refs = {"composite": c[comp_kind], "composite-clean": c[comp_kind], "clean": c["clean"]}
            parts = build_parts(text, c["clean"], refs, deck_text, furniture, looks[style], stairs)
            for k in range(args.repeat):
                jobs.append((p, v, template, parts, c, comp_kind, paste, k))

    def work(job):
        p, v, template, parts, c, comp_kind, paste, k = job
        size = Image.open(io.BytesIO(c["clean"])).size
        t0 = time.time()
        try:
            image, _ = providers.edit(MODEL, parts, size)
        except Exception as err:  # noqa: BLE001
            return {"photo": p, "variant": v, "k": k + 1, "error": str(err)[:200]}
        jpeg = providers.as_jpeg(image)
        name = f"{p}__{v}__{k + 1}"
        (run / f"{name}.jpg").write_bytes(jpeg)
        seconds = round(time.time() - t0, 1)
        moved = {}
        if paste:  # a moved output pastes the deck off its place; the checks too read it on the composite's pixels
            jpeg, moved = register(jpeg, c[comp_kind], c["alpha"])
        row = {"photo": p, "variant": v, "template": template, "k": k + 1, "seconds": seconds, "moved": moved,
               "photo_score": round(check.drift(jpeg, c[comp_kind], c["alpha"]), 2),
               "deck_score": round(check.deck(jpeg, c[comp_kind], c["alpha"]), 2)}
        if paste:
            (run / f"{name}.pasted.jpg").write_bytes(paste_back(jpeg, c["clean"], c["alpha"], c["shadow"],
                                                                objects=True))
        return row

    with cf.ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(work, jobs))
    for r in rows:
        print(r)
    (run / "run.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("detect")
    a.add_argument("--photos", nargs="+", default=["angled", "existing-deck", "regular"])
    a.add_argument("--repeat", type=int, default=3)
    w = sub.add_parser("wall")
    w.add_argument("--photos", nargs="+", default=["angled", "existing-deck", "regular"])
    w.add_argument("--repeat", type=int, default=3)
    w.add_argument("--clean", type=int, default=1, help="which detect run's cleaned photo to use")
    c = sub.add_parser("compose")
    c.add_argument("--photos", nargs="+", default=["angled", "existing-deck", "regular"])
    o = sub.add_parser("polish")
    o.add_argument("--photos", nargs="+", default=["angled", "existing-deck", "regular"])
    o.add_argument("--variants", nargs="+", default=list(VARIANTS))
    o.add_argument("--repeat", type=int, default=6)
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--name", default="d1")
    args = ap.parse_args()
    {"detect": cmd_detect, "wall": cmd_wall, "compose": cmd_compose, "polish": cmd_polish}[args.cmd](args)


if __name__ == "__main__":
    main()

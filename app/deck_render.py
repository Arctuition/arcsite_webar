"""Deck photo rendering for the canvas, while the prompt is being tuned (see tuning/README.md).

The engine draws the deck from the photo's own camera, in two calls:
  POST /api/deck/analyze  the photo and the deck: the wall marked by Gemini, the camera solved, the deck placed along
                          the wall; answers with a job and the camera for AR_Web_RenderDeckPhotoDiagram. A photo with
                          a deck already on it starts being cleaned (that deck removed) in the background.
  POST /api/deck/render   the job and the engine's two transparent renders (the deck alone, and with the ground): the
                          deck pasted on the clean photo and made real by Gemini, retried while tuning.check says the
                          photo or the deck was redrawn.

The p2 flow (DECK_FLOW, default; tuning/pipeline2.py), on the same two routes and the same canvas renders:
  /analyze  one image-model call says whether the photo has a deck and removes it, while the wall is marked on the
            photo; with a deck, the wall is marked on the cleaned photo too, and the camera that fits best (lowest
            line cost) and can hold the deck (camera.fit_problems) is kept.
  /render   the deck render pasted on the clean photo with its shadow and the photo's colour; the model polishes only the
            deck (p2l2), its answer is moved back onto the composite's pixels and pasted on the clean photo around the
            deck and its shadow, the rest of the photo kept as it was.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures as cf
import datetime as dt
import io
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, Field

from app import gemini
from tuning import camera, check, inputs, pipeline2, providers
from tuning.describe import _edge_no, describe
from tuning.run import FLOOR_SEEN_FT, HERE as TUNING_DIR, build_parts, prompt_text

log = logging.getLogger(__name__)

router = APIRouter()


def _decode(name: str, value: str) -> bytes:
    try:
        return base64.b64decode(value.split(",", 1)[-1], validate=False)
    except Exception as err:
        raise HTTPException(400, f"{name} is not base64") from err


CLEAN_MODEL = "gemini-3.1-flash-image"
COMPOSE_MODEL = "gemini-3.1-flash-image"
WALL_TRIES = 2
DETECT_TRIES = 2  # the model says DECK but sends no image about one call in seven; asked once more
FLOW = os.environ.get("DECK_FLOW", "p2")  # "p2", or "two-step": the flow before 11-25 (v10j/v10k on the composite)
P2_PROMPT = "p2l2"
JOB_TTL_S = 30 * 60
PROMPT_NAME = re.compile(r"[\w-]+")

_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()
_cleaner = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="deck-clean")


@dataclass
class Job:
    created: float
    photo: bytes
    size: tuple[int, int]
    deck: dict
    marks: dict
    cam: dict
    placement: dict
    engine: dict
    clean: cf.Future | None  # None: the photo had no deck to remove
    out: Path
    composes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    flow: str = "two-step"


def _sweep() -> None:
    now = time.time()
    with _jobs_lock:
        for key in [k for k, j in _jobs.items() if now - j.created > JOB_TTL_S]:
            del _jobs[key]  # a clean still running finishes and is dropped; a compose holding the job still gets it


def _as_jpeg_photo(data: bytes) -> tuple[bytes, tuple[int, int]]:
    """The photo as sent (a JPEG, as the canvas prepares it); anything else is re-encoded as JPEG 92."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as err:
        raise HTTPException(400, "photo is not an image") from err
    if im.format == "JPEG":
        return data, im.size
    out = io.BytesIO()
    im.convert("RGB").save(out, "JPEG", quality=92)
    return out.getvalue(), im.size


def _house_wall(deck: dict) -> dict | None:
    """The engine's exported wall run on the deck's ledger edge, else the first one, else none (old engines)."""
    walls = [w for w in deck.get("walls") or [] if isinstance(w, dict) and "a" in w and "b" in w]
    ledger = {_edge_no(e) for e in deck.get("ledger") or []}
    return next((w for w in walls if w.get("edge") in ledger), walls[0] if walls else None)


def _finite(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _solve(marks: dict, jpeg: bytes, size: tuple[int, int], deck: dict) -> tuple[dict, dict, dict]:
    """camera.solve + place + engine_camera; raises ValueError with a reason when no usable camera comes out."""
    w, h = size
    try:
        if not marks.get("base_line") and not (marks.get("existing_deck") or {}).get("surface_line"):
            raise ValueError("the analysis found no wall ground line (nor an existing deck's floor line) in the photo")
        cam = camera.solve(marks, w, h, camera.detect_lines(jpeg))
        if not _finite([cam["wall_length_ft"], cam["camera_height_ft"], cam["hfov_deg"]]) or cam["wall_length_ft"] <= 0:
            raise ValueError("the solved camera is degenerate (wall length or camera height is not usable)")
        placement = camera.place(deck, cam, _house_wall(deck))
        eng = camera.engine_camera(deck, cam, placement)
    except ValueError:
        raise
    except Exception as err:  # marks of the wrong shape (AttributeError, TypeError, ...), cv2.error from the lines
        log.warning("deck wall: the marks could not be solved", exc_info=True)
        raise ValueError(f"the wall marks could not be solved: {err!r}") from err
    engine = {"projection": "perspective", "eye": [float(v) for v in eng["eye"]],
              "target": [float(v) for v in eng["target"]], "up": [float(v) for v in eng["up"]],
              "fovDeg": float(eng["fovDeg"])}
    if not _finite(engine["eye"] + engine["target"] + engine["up"] + [engine["fovDeg"]]):
        raise ValueError("the solved camera is degenerate (not finite)")
    placement = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in placement.items()}
    problems = camera.fit_problems(deck, cam, placement)
    if problems:  # a deck mostly out of the picture, or shrunk to a dot by a broken camera, looks wrong however drawn
        raise PhotoDoesNotFit("this photo cannot show this deck: " + "; ".join(problems))
    return cam, placement, engine


class PhotoDoesNotFit(ValueError):
    """The camera is fine but the deck cannot stand where this photo puts it: ask for another photo."""


def _write(out: Path, files: dict[str, bytes | str | dict]) -> None:
    """Keep what went in and came out under tuning/out/live/ (not committed); never fails the request."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            if isinstance(data, dict):
                data = json.dumps(data, indent=2, default=str)
            (out / name).write_bytes(data if isinstance(data, bytes) else data.encode())
    except OSError as err:
        log.warning("could not keep %s: %s", out, err)


class WallRequest(BaseModel):
    photo: str = Field(description="the house photo, base64 JPEG (long side 2048, quality 92)")
    deck: dict = Field(description="the deck as the engine exports it (DeckPhotoDeck.json)")


@router.post("/api/deck/analyze")
async def analyze(req: WallRequest) -> dict:
    _sweep()
    t0 = time.time()
    photo, size = await run_in_threadpool(_as_jpeg_photo, _decode("photo", req.photo))
    try:
        describe(req.deck)
        float(req.deck["height_in"])
    except (KeyError, ValueError, TypeError, IndexError) as err:
        raise HTTPException(400, f"deck: {err}") from err

    if FLOW == "p2":
        return await _wall_p2(req, photo, size, t0)

    # The model's marks differ from call to call, and now and then give no camera (one in two angled tries, 11-24):
    # asked once more before the photo is given up on.
    for attempt in range(WALL_TRIES):
        try:
            marks = await run_in_threadpool(camera.mark_wall, photo)
        except gemini.GeminiError as err:
            raise HTTPException(502, f"wall analysis: {err}") from err
        if not isinstance(marks, dict):
            raise HTTPException(502, "wall analysis: the answer is not a JSON object")
        try:
            cam, placement, engine = await run_in_threadpool(_solve, marks, photo, size, req.deck)
            break
        except ValueError as err:  # PhotoDoesNotFit too: another answer may mark a longer wall
            log.warning("deck wall: try %d of %d gave no camera: %s", attempt + 1, WALL_TRIES, err)
            if attempt + 1 == WALL_TRIES:
                raise HTTPException(422, str(err)) from err
    existing = bool(marks.get("existing_deck"))  # the paid clean starts only once the camera is usable
    clean = _cleaner.submit(inputs.clean_photo, photo, CLEAN_MODEL) if existing else None

    key = secrets.token_urlsafe(12)
    out = TUNING_DIR / "out" / "live" / f"{dt.datetime.now():%m%d-%H%M%S}-two-step"
    wall = {"label": marks.get("label"), "hfov_deg": float(cam["hfov_deg"]),
            "camera_height_ft": float(cam["camera_height_ft"]), "wall_length_ft": float(cam["wall_length_ft"]),
            "placement": placement, "existing_deck": existing}
    with _jobs_lock:
        _jobs[key] = Job(time.time(), photo, size, req.deck, marks, cam, placement, engine, clean, out)
    seconds = round(time.time() - t0, 1)
    solved = {k: v for k, v in cam.items() if k not in ("snapped_lines",)}
    await run_in_threadpool(_write, out, {"photo.jpg": photo, "deck.json": req.deck, "wall.json": {
        "job": key, "size": size, "marks": marks, "solved": solved, "wall": wall, "camera": engine,
        "seconds": seconds}})
    log.info("deck wall %s: %s, %.1f s", key, wall["label"], seconds)
    return {"job": key, "width": size[0], "height": size[1], "camera": engine, "wall": wall, "seconds": seconds}


def _detect(photo: bytes) -> dict:
    """pipeline2.detect_clean, asked again while it says DECK with no image. {verdict: DECK|NONE|None, text, image,
    tries}; verdict DECK with image None when every try said DECK and sent no image, None when detection failed."""
    said_deck, last = False, None
    for k in range(DETECT_TRIES):
        try:
            return {**pipeline2.detect_clean(photo), "tries": k + 1}
        except pipeline2.NoCleanImage as err:
            said_deck, last = True, err
        except Exception as err:  # noqa: BLE001 -- the wall marks then say whether there is a deck
            last = err
        log.warning("deck wall p2: deck detection, try %d of %d: %s", k + 1, DETECT_TRIES, last)
    return {"verdict": "DECK" if said_deck else None, "text": str(last)[:200], "image": None, "tries": DETECT_TRIES}


def _p2_cameras(photo: bytes, size: tuple[int, int], deck: dict) -> dict:
    """Steps 1-3 of the p2 flow: {verdict, clean (None without a deck), src, marks, cam, placement, engine}, raising
    HTTPException when no source gives a camera that can hold the deck."""
    def solve(jpeg: bytes):
        marks = camera.mark_wall(jpeg)
        if not isinstance(marks, dict):
            raise ValueError("the wall analysis answer is not a JSON object")
        return (marks, *_solve(marks, jpeg, size, deck))

    detect = _cleaner.submit(_detect, photo)
    sources = {"original": photo}
    futures = {"original": _cleaner.submit(solve, photo)}
    det = detect.result()
    clean = det["image"] if det["verdict"] == "DECK" else None
    if clean is None and det["verdict"] != "NONE":
        # The detection said DECK and sent no picture, or failed outright: the wall marks, made at the same time,
        # say whether the photo has a deck. Going on with the deck still in the photo puts the new deck on top
        # of it, and only the new deck's part is pasted back, the old one showing around it.
        has_deck = det["verdict"] == "DECK"
        if not has_deck:
            try:
                has_deck = bool(futures["original"].result()[0].get("existing_deck"))
            except Exception:  # noqa: BLE001 -- the wall loop below reports it
                pass
        if has_deck:
            try:
                clean = inputs.clean_photo(photo, CLEAN_MODEL)
                det["cleaned_by"] = "inputs.clean_photo"
            except Exception as err:  # noqa: BLE001
                log.warning("deck wall p2: the photo's deck could not be removed: %s", err)
                raise HTTPException(502, "the photo has a deck that could not be removed; try again") from err
    if clean is not None:
        sources["cleaned"] = clean
        futures["cleaned"] = _cleaner.submit(solve, clean)

    errors, unfit, found = [], [], {}
    for attempt in range(WALL_TRIES):  # the marks differ from call to call; a source that failed is asked once more
        for src, fut in futures.items():
            try:
                found[src] = fut.result()
            except PhotoDoesNotFit as err:
                unfit.append(str(err))
            except (ValueError, gemini.GeminiError) as err:
                errors.append(f"{src}: {err}")
        if found or attempt + 1 == WALL_TRIES:
            break
        log.warning("deck wall p2: try %d of %d gave no camera: %s", attempt + 1, WALL_TRIES, errors + unfit)
        futures = {src: _cleaner.submit(solve, jpeg) for src, jpeg in sources.items()}
    if not found:
        raise HTTPException(422, unfit[-1] if unfit else "; ".join(errors[-2:]))
    src = min(found, key=lambda k: float(found[k][1]["fit_cost"]))
    marks, cam, placement, engine = found[src]
    old_box = None  # where the photo's old deck is, by the marks on the original photo (the cleaned one has none)
    if "original" in found:
        old_box = (found["original"][0].get("existing_deck") or {}).get("box")
    return {"verdict": det["verdict"], "detect_text": det.get("text"), "detect_tries": det.get("tries"), "old_box": old_box,
            "cleaned_by": det.get("cleaned_by", "detect_clean" if clean is not None else None),
            "clean": clean, "src": src, "marks": marks,
            "cam": cam, "placement": placement, "engine": engine,
            "fit_costs": {k: round(float(v[1]["fit_cost"]), 5) for k, v in found.items()}}


async def _wall_p2(req: "WallRequest", photo: bytes, size: tuple[int, int], t0: float) -> dict:
    r = await run_in_threadpool(_p2_cameras, photo, size, req.deck)
    clean, base, restored = None, None, None
    if r["clean"] is not None:  # the photo itself, with the model's picture only where the old deck was
        base, restored = await run_in_threadpool(pipeline2.restore_photo, photo, r["clean"], r["old_box"])
        clean = cf.Future()
        clean.set_result(base)
    key = secrets.token_urlsafe(12)
    out = TUNING_DIR / "out" / "live" / f"{dt.datetime.now():%m%d-%H%M%S}-p2"
    cam, placement, engine = r["cam"], r["placement"], r["engine"]
    wall = {"label": r["marks"].get("label"), "hfov_deg": float(cam["hfov_deg"]),
            "camera_height_ft": float(cam["camera_height_ft"]), "wall_length_ft": float(cam["wall_length_ft"]),
            "placement": placement, "existing_deck": r["clean"] is not None, "solved_on": r["src"]}
    with _jobs_lock:
        _jobs[key] = Job(time.time(), photo, size, req.deck, r["marks"], cam, placement, engine, clean, out, flow="p2")
    seconds = round(time.time() - t0, 1)
    solved = {k: v for k, v in cam.items() if k not in ("snapped_lines",)}
    files = {"photo.jpg": photo, "deck.json": req.deck, "wall.json": {
        "job": key, "flow": "p2", "size": size,
        "detect": {"verdict": r["verdict"], "text": r["detect_text"], "tries": r["detect_tries"],
                   "cleaned_by": r["cleaned_by"], "old_box": r["old_box"], "restored": restored},
        "fit_costs": r["fit_costs"], "marks": r["marks"], "solved": solved, "wall": wall, "camera": engine,
        "seconds": seconds}}
    if r["clean"] is not None:
        files["clean.jpg"] = r["clean"]
        files["base.jpg"] = base
    await run_in_threadpool(_write, out, files)
    log.info("deck wall p2 %s: %s on the %s photo (deck: %s), %.1f s", key, wall["label"], r["src"], r["verdict"], seconds)
    return {"job": key, "width": size[0], "height": size[1], "camera": engine, "wall": wall, "seconds": seconds}


class ComposeRequest(BaseModel):
    job: str
    deck_png: str = Field(description="the engine's render of the deck alone, transparent, base64 PNG")
    ground_png: str = Field(description="the same camera with the ground shown, base64 PNG")
    style: str
    model: str | None = None
    prompt: str | None = None  # "auto" (default): v10k when the camera sees the deck floor, else v10j
    retry: int = Field(2, ge=0, le=4)


def _render_png(name: str, data: bytes, size: tuple[int, int]) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as err:
        raise HTTPException(400, f"{name} is not an image") from err
    im = im.convert("RGBA")
    return im if im.size == size else im.resize(size, Image.LANCZOS)


def _composite(clean: bytes, deck: Image.Image) -> bytes:
    out = io.BytesIO()
    Image.alpha_composite(Image.open(io.BytesIO(clean)).convert("RGBA"), deck).convert("RGB").save(out, "JPEG", quality=92)
    return out.getvalue()


def _png(im: Image.Image) -> bytes:
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


def _attempts(model: str, parts: list, size: tuple[int, int], composite: bytes, alpha: Image.Image, retry: int,
              fix=None):
    """Up to 1 + retry edits; stop at the first that keeps the photo and the deck, else keep the best by check.rank.
    A provider failure after a scored result keeps that result; before one, it is raised."""
    scores, best = [], None
    for _ in range(retry + 1):
        try:
            image, _note = providers.edit(model, parts, size)
        except providers.ProviderError:
            if best is None:
                raise
            log.warning("deck compose: a retry failed, keeping the best so far", exc_info=True)
            break
        if fix is not None:  # p2: the answer moved back onto the composite's pixels before it is scored
            image = fix(image)
        sc = {"photo": round(check.drift(image, composite, alpha), 2), "deck": round(check.deck(image, composite, alpha), 2)}
        scores.append(sc)
        if best is None or check.rank(sc) > best[0]:
            best = (check.rank(sc), image, sc)
        if check.passes(sc):
            break
    return best[1], best[2], scores


@router.post("/api/deck/render")
async def render(req: ComposeRequest) -> dict:
    _sweep()
    t0 = time.time()
    looks = json.loads((TUNING_DIR / "styles.json").read_text())
    if req.style not in looks:
        raise HTTPException(400, f"unknown style {req.style}")
    with _jobs_lock:
        job = _jobs.get(req.job)
    if job is None:
        raise HTTPException(404, "unknown or expired job; send the photo to /api/deck/analyze again")
    prompt_name = req.prompt or "auto"
    if prompt_name == "auto" and job.flow == "p2":
        prompt_name = P2_PROMPT
    if prompt_name == "auto":
        seen = job.cam["camera_height_ft"] > float(job.deck["height_in"]) / 12 + FLOOR_SEEN_FT
        prompt_name = "v10k" if seen else "v10j"
    template_path = TUNING_DIR / "prompts" / f"{prompt_name}.txt"
    if not PROMPT_NAME.fullmatch(prompt_name) or not template_path.is_file():
        raise HTTPException(400, f"unknown prompt {prompt_name}")
    model = req.model or COMPOSE_MODEL
    deck_raw, ground_raw = _decode("deck_png", req.deck_png), _decode("ground_png", req.ground_png)

    def above() -> Image.Image:
        return inputs.above_ground(_render_png("deck_png", deck_raw, job.size), _render_png("ground_png", ground_raw, job.size))

    deck_im = await run_in_threadpool(above)
    if deck_im.getchannel("A").getbbox() is None:  # nothing to polish: the model would only be asked to keep the photo
        raise HTTPException(422, "the 3D view drew no deck from this camera; send the photo to /api/deck/analyze again")
    t_clean = time.time()
    if job.clean is None:
        clean = job.photo
    else:
        try:
            clean = await asyncio.wrap_future(job.clean)
        except Exception as err:  # providers.ProviderError, or an answer that is not an image
            raise HTTPException(502, f"removing the existing deck: {err}") from err
    clean_wait = round(time.time() - t_clean, 1)
    p2 = None
    if job.flow == "p2":
        ground_im = await run_in_threadpool(_render_png, "ground_png", ground_raw, job.size)
        p2 = await run_in_threadpool(pipeline2.compose_images, clean, deck_im, ground_im)
        composite = p2["rich"]
    else:
        composite = await run_in_threadpool(_composite, clean, deck_im)
    alpha = deck_im.getchannel("A")

    deck_text, furniture = describe(job.deck)
    try:  # where the stairs fall in this photo; a flight running out of the frame is named, so none is made up
        stairs = camera.stairs_note(job.deck, job.cam, job.placement)
    except Exception:  # noqa: BLE001 -- an odd stairs entry must not cost the rendering
        log.warning("deck compose: no stairs note", exc_info=True)
        stairs = camera.IN_VIEW_STAIRS
    template = template_path.read_text()
    if "{furniture}" in template and not prompt_name.startswith("v10"):  # the p2 prompts list it only when seen
        seen = job.cam["camera_height_ft"] > float(job.deck["height_in"]) / 12 + FLOOR_SEEN_FT
        template = template.replace("{furniture}", pipeline2.furniture_block(furniture) if seen else "")
    try:
        parts = build_parts(template, clean, {"composite": composite, "composite-clean": composite, "clean": clean},
                            deck_text, furniture, looks[req.style], stairs)
    except KeyError as err:
        raise HTTPException(400, f"prompt {prompt_name} needs an image this flow does not have: {err}") from err

    try:
        fix = (lambda im: pipeline2.register(im, composite, alpha)[0]) if p2 else None
        image, scores, attempts = await run_in_threadpool(_attempts, model, parts, job.size, composite, alpha, req.retry,
                                                          fix)
    except providers.ProviderError as err:
        raise HTTPException(502, str(err)) from err
    polished = None
    if p2:  # only the deck and its shadow come from the answer; the rest of the photo stays the photo
        polished = image
        image = await run_in_threadpool(pipeline2.paste_back, image, clean, alpha, p2["shadow"], 25, 8.0, True)
    jpeg = await run_in_threadpool(providers.as_jpeg, image)
    width, height = Image.open(io.BytesIO(jpeg)).size
    passed = check.passes(scores)
    seconds = round(time.time() - t0, 1)
    text = prompt_text(parts)

    with job.lock:
        job.composes += 1
        out = job.out if job.composes == 1 else job.out / f"compose-{job.composes}"
    files = {"deck.png": deck_raw, "ground.png": ground_raw, "deck-above-ground.png": await run_in_threadpool(_png, deck_im),
             "composite.jpg": composite, "result.jpg": jpeg, "meta.json": {
                 "job": req.job, "flow": job.flow, "style": req.style, "model": model, "prompt_version": prompt_name,
                 "retry": req.retry,
                 "camera": job.engine, "camera_height_ft": job.cam["camera_height_ft"], "placement": job.placement,
                 "attempts": attempts, "kept": scores, "passed": passed, "cleaned": job.clean is not None,
                 "seconds": seconds, "clean_wait_s": clean_wait, "prompt": text}}
    if job.clean is not None:  # p2: the photo with the old deck's place from clean.jpg, which analyze kept
        files["base.jpg" if job.flow == "p2" else "clean.jpg"] = clean
    if polished is not None:
        files["polished.jpg"] = await run_in_threadpool(providers.as_jpeg, polished)
    await run_in_threadpool(_write, out, files)
    log.info("deck compose %s: %s %s %s, %s, %.1f s", req.job, req.style, model, prompt_name, attempts, seconds)
    return {
        "image": base64.b64encode(jpeg).decode(),
        "mime": "image/jpeg",
        "width": width,
        "height": height,
        "seconds": seconds,
        "model": model,
        "prompt_version": prompt_name,
        "prompt": text,
        "attempts": attempts,
        "passed": passed,
        "cleaned": job.clean is not None,
    }

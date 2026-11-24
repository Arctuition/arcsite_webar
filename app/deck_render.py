"""One-step deck photo rendering, for the canvas to call while the prompt is being tuned.

POST /api/deck/render with the photo, the three 3D-view references, the canvas's deck JSON and the
style; answers with the rendering. One model call, no photo analysis and no placement: the model
puts the deck on the house itself (see tuning/README.md).
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel, Field

from tuning import providers
from tuning.describe import describe
from tuning.run import HERE as TUNING_DIR, build_parts, prompt_text

router = APIRouter()

DEFAULT_MODEL = "gpt-image-2.6-sunburst"
DEFAULT_PROMPT = "v5p"
VIEWS = ("top", "left-front", "right-front")


class RenderRequest(BaseModel):
    photo: str = Field(description="the house photo, base64 JPEG or PNG")
    refs: dict[str, str] = Field(description="base64 PNGs keyed top, left-front, right-front")
    deck: dict
    style: str
    model: str | None = None
    prompt: str | None = None


def _decode(name: str, value: str) -> bytes:
    try:
        return base64.b64decode(value.split(",", 1)[-1], validate=False)
    except Exception as err:
        raise HTTPException(400, f"{name} is not base64") from err


def _keep(photo: bytes, refs: dict[str, bytes], req: RenderRequest, result: bytes, prompt: str,
          model: str, prompt_name: str, size: tuple[int, int], seconds: float) -> None:
    """Every rendering the canvas asked for, with what went in, under tuning/out/live/ (not committed)."""
    out = TUNING_DIR / "out" / "live" / f"{dt.datetime.now():%m%d-%H%M%S}-{req.style}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "photo.jpg").write_bytes(photo)
    for name, data in refs.items():
        (out / f"{name}.png").write_bytes(data)
    (out / "result.jpg").write_bytes(result)
    (out / "deck.json").write_text(json.dumps(req.deck, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "style": req.style, "model": model, "prompt_version": prompt_name, "photo_size": size,
        "seconds": seconds, "prompt": prompt}, indent=2))


@router.post("/api/deck/render")
async def render(req: RenderRequest) -> dict:
    missing = [v for v in VIEWS if v not in req.refs]
    if missing:
        raise HTTPException(400, f"missing references: {', '.join(missing)}")
    looks = json.loads((TUNING_DIR / "styles.json").read_text())
    if req.style not in looks:
        raise HTTPException(400, f"unknown style {req.style}")
    prompt_name = req.prompt or DEFAULT_PROMPT
    template_path = TUNING_DIR / "prompts" / f"{prompt_name}.txt"
    if not template_path.is_file():
        raise HTTPException(400, f"unknown prompt {prompt_name}")
    try:
        deck_text, furniture = describe(req.deck)
    except (KeyError, ValueError, TypeError) as err:
        raise HTTPException(400, f"deck: {err}") from err

    photo = _decode("photo", req.photo)
    refs = {v: _decode(v, req.refs[v]) for v in VIEWS}
    size = Image.open(io.BytesIO(photo)).size
    parts = build_parts(template_path.read_text(), photo, refs, deck_text, furniture, looks[req.style])
    model = req.model or DEFAULT_MODEL

    t0 = time.time()
    try:
        image, _ = await run_in_threadpool(providers.edit, model, parts, size)
    except providers.ProviderError as err:
        raise HTTPException(502, str(err)) from err
    jpeg = providers.as_jpeg(image)
    width, height = Image.open(io.BytesIO(jpeg)).size
    _keep(photo, refs, req, jpeg, prompt_text(parts), model, prompt_name, size, round(time.time() - t0, 1))
    return {
        "image": base64.b64encode(jpeg).decode(),
        "mime": "image/jpeg",
        "width": width,
        "height": height,
        "seconds": round(time.time() - t0, 1),
        "model": model,
        "prompt_version": prompt_name,
        "prompt": prompt_text(parts),
    }

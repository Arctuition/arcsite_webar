"""One image edit through Gemini or OpenAI. Keys come from GEMINI_API_KEY / OPENAI_API_KEY and are never printed."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
import uuid

from PIL import Image

Part = str | tuple[str, bytes]  # text, or (image name, PNG/JPEG bytes)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENAI_URL = "https://api.openai.com/v1/images/edits"
TIMEOUT_S = 300
GEMINI_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


class ProviderError(RuntimeError):
    pass


def _key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProviderError(f"{name} is not set")
    return value


def _mime(data: bytes) -> str:
    return "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"


def _open(request: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as err:
        body = err.read().decode(errors="replace")
        try:
            message = json.loads(body)["error"]["message"]
        except Exception:
            message = body[:300]
        raise ProviderError(f"HTTP {err.code}: {message}") from err
    except (urllib.error.URLError, TimeoutError) as err:
        raise ProviderError(f"network: {err}") from err


def gemini(model: str, parts: list[Part], photo_size: tuple[int, int]) -> tuple[bytes, str | None]:
    image, text, reason = gemini_answer(model, parts, photo_size)
    if image:
        return image, text
    raise ProviderError(f"no image ({reason}): {text}"[:400])


def gemini_answer(model: str, parts: list[Part], photo_size: tuple[int, int]) -> tuple[bytes | None, str | None, str]:
    """The image (or None: some prompts let the model answer in words only), all the text, and the finish reason."""
    target = photo_size[0] / photo_size[1]
    ratio = min(GEMINI_RATIOS, key=lambda r: abs(int(r.split(":")[0]) / int(r.split(":")[1]) - target))
    body = {
        "contents": [{"parts": [
            {"text": p} if isinstance(p, str)
            else {"inline_data": {"mime_type": _mime(p[1]), "data": base64.b64encode(p[1]).decode()}}
            for p in parts
        ]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": ratio}},
    }
    request = urllib.request.Request(
        GEMINI_URL.format(model=model), data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": _key("GEMINI_API_KEY")},
    )
    payload = _open(request)
    texts, image = [], None  # all the answer's text, before or after the image (a detect-and-edit call reads it)
    for part in ((payload.get("candidates") or [{}])[0].get("content") or {}).get("parts", []):
        if part.get("thought"):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            image = image or base64.b64decode(inline["data"])
        elif part.get("text"):
            texts.append(part["text"])
    text = "\n".join(texts) or None
    reason = (payload.get("candidates") or [{}])[0].get("finishReason")
    return image, text or (None if image else json.dumps(payload.get("promptFeedback"))), str(reason)


def _openai_size(photo_size: tuple[int, int]) -> str:
    w, h = photo_size
    return "1536x1024" if w / h > 1.15 else "1024x1536" if h / w > 1.15 else "1024x1024"


def _pad_to(data: bytes, ratio: float) -> tuple[bytes, tuple[float, float, float, float]]:
    """The photo on a canvas of [ratio], its borders filled with a blurred stretch of the photo so the model
    sees no hard edge; returns the PNG and where the photo sits, as fractions of the canvas."""
    from PIL import ImageFilter

    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    if abs(w / h - ratio) < 0.01:
        return data, (0, 0, 1, 1)
    cw, ch = (w, round(w / ratio)) if w / h > ratio else (round(h * ratio), h)
    canvas = im.resize((cw, ch)).filter(ImageFilter.GaussianBlur(40))
    x, y = (cw - w) // 2, (ch - h) // 2
    canvas.paste(im, (x, y))
    out = io.BytesIO()
    canvas.save(out, "PNG")
    return out.getvalue(), (x / cw, y / ch, (x + w) / cw, (y + h) / ch)


def _crop(data: bytes, box: tuple[float, float, float, float]) -> bytes:
    if box == (0, 0, 1, 1):
        return data
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    out = io.BytesIO()
    im.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h))).save(out, "PNG")
    return out.getvalue()


def fit_aspect(data: bytes, photo_size: tuple[int, int]) -> bytes:
    """Centre-crop a result whose aspect drifted from the photo's back to it."""
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    want = photo_size[0] / photo_size[1]
    if abs(w / h - want) / want < 0.02:
        return data
    cw, ch = (round(h * want), h) if w / h > want else (w, round(w / want))
    x, y = (w - cw) // 2, (h - ch) // 2
    out = io.BytesIO()
    im.crop((x, y, x + cw, y + ch)).save(out, "PNG")
    return out.getvalue()


def openai(model: str, parts: list[Part], photo_size: tuple[int, int]) -> tuple[bytes, str | None]:
    size = _openai_size(photo_size)
    sw, sh = (int(v) for v in size.split("x"))
    box = (0, 0, 1, 1)
    images = []
    for p in parts:
        if isinstance(p, str):
            continue
        if p[0] == "photo":  # padded to the output's shape, then cropped back, so the framing survives
            padded, box = _pad_to(p[1], sw / sh)
            images.append((p[0], padded))
        else:
            images.append(p)
    prompt = "\n".join(p for p in parts if isinstance(p, str))
    fields = {"model": model, "prompt": prompt, "size": size, "quality": "high",
              "input_fidelity": "high", "n": "1"}
    for attempt in range(2):
        boundary = uuid.uuid4().hex
        buf = io.BytesIO()
        for k, v in fields.items():
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for name, data in images:
            mime = _mime(data)
            ext = "png" if mime == "image/png" else "jpg"
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"{name}.{ext}\"\r\n"
                      f"Content-Type: {mime}\r\n\r\n".encode())
            buf.write(data)
            buf.write(b"\r\n")
        buf.write(f"--{boundary}--\r\n".encode())
        request = urllib.request.Request(
            OPENAI_URL, data=buf.getvalue(), method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Authorization": f"Bearer {_key('OPENAI_API_KEY')}"},
        )
        try:
            payload = _open(request)
            break
        except ProviderError as err:
            # Newer models may not take input_fidelity; try once without it.
            if attempt == 0 and "input_fidelity" in str(err):
                fields.pop("input_fidelity")
                continue
            raise
    data = (payload.get("data") or [{}])[0]
    if data.get("b64_json"):
        return _crop(base64.b64decode(data["b64_json"]), box), data.get("revised_prompt")
    raise ProviderError(f"no image: {json.dumps(payload)[:300]}")


def edit(model: str, parts: list[Part], photo_size: tuple[int, int]) -> tuple[bytes, str | None]:
    image, note = (gemini if model.startswith("gemini") else openai)(model, parts, photo_size)
    return fit_aspect(image, photo_size), note


def as_jpeg(data: bytes, quality: int = 90) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    out = io.BytesIO()
    im.save(out, "JPEG", quality=quality)
    return out.getvalue()

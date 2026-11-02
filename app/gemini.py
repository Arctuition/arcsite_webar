"""Server-side Gemini image editing. The API key never leaves the server.

Configure with environment variables (or a `.env` file in the project root):
  GEMINI_API_KEY      required
  GEMINI_IMAGE_MODEL  optional, defaults to DEFAULT_MODEL
  GEMINI_TEXT_MODEL   optional, defaults to DEFAULT_TEXT_MODEL (photo viewpoint analysis)
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "gemini-3.1-flash-image"
# Reads the photo's viewpoint. In testing only this model got the camera side right every time
# (lighter models were consistently or randomly mirrored), at "low" thinking.
DEFAULT_TEXT_MODEL = "gemini-3.8-flash"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT_S = 180

# Aspect ratios the image models accept; the output matches the closest one to the photo
ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


class GeminiError(Exception):
    pass


def _load_dotenv() -> None:
    """Minimal .env support: KEY=VALUE lines, existing environment variables win."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv()


def is_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def closest_aspect_ratio(width: int, height: int) -> str:
    target = width / height
    return min(ASPECT_RATIOS, key=lambda r: abs(int(r.split(":")[0]) / int(r.split(":")[1]) - target))


Part = str | tuple[bytes, str]  # text, or (image bytes, mime type)


def _post(model: str, body: dict, timeout: float) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not configured on the server.")
    request = urllib.request.Request(
        API_URL.format(model=urllib.request.quote(model, safe="")),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as err:
        try:
            message = json.loads(err.read())["error"]["message"]
        except Exception:
            message = f"Request failed (HTTP {err.code})."
        raise GeminiError(message) from err
    except (urllib.error.URLError, TimeoutError) as err:
        raise GeminiError(f"Couldn't reach Gemini: {err}") from err


def _encode(parts: list[Part]) -> list[dict]:
    return [
        {"text": p} if isinstance(p, str) else {"inline_data": {"mime_type": p[1], "data": base64.b64encode(p[0]).decode()}}
        for p in parts
    ]


def analyze(parts: list[Part], timeout: float = 45) -> dict:
    """Ask the text model about images; returns its JSON answer."""
    model = os.environ.get("GEMINI_TEXT_MODEL", "").strip() or DEFAULT_TEXT_MODEL
    body = {
        "contents": [{"parts": _encode(parts)}],
        "generationConfig": {"responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}},
    }
    payload = _post(model, body, timeout)
    texts = [p["text"] for p in ((payload.get("candidates") or [{}])[0].get("content") or {}).get("parts", []) if p.get("text") and not p.get("thought")]
    try:
        return json.loads(texts[-1])
    except (IndexError, ValueError) as err:
        raise GeminiError("The analysis returned no JSON.") from err


def edit_image(parts: list[Part], aspect_ratio: str, model: str | None = None) -> tuple[bytes, str]:
    """Send text and images interleaved, in order; return the first generated image."""
    model = model or os.environ.get("GEMINI_IMAGE_MODEL", "").strip() or DEFAULT_MODEL
    body = {
        "contents": [{"parts": _encode(parts)}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": aspect_ratio}},
    }
    payload = _post(model, body, TIMEOUT_S)

    text = None
    candidates = payload.get("candidates") or [{}]
    for part in (candidates[0].get("content") or {}).get("parts", []):
        if part.get("thought"):
            continue
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"]), inline.get("mimeType") or inline.get("mime_type") or "image/png"
        text = part.get("text") or text
    raise GeminiError(f"The model returned no image: {text}" if text else "The model returned no image.")

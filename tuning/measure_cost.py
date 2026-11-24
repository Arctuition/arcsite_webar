"""One real request per model with the tuned inputs, printing the token usage each API reports.

  .venv/bin/python -m tuning.measure_cost --models gpt-image-2.6-sunburst gemini-3.1-flash-image
"""

from __future__ import annotations

import argparse
import io
import json
import time

from PIL import Image

from . import inputs, providers
from .describe import describe
from .run import HERE, build_parts

USAGE: list[dict] = []
_open = providers._open


def _recording_open(request):
    payload = _open(request)
    USAGE.append(payload.get("usage") or payload.get("usageMetadata") or {})
    return payload


providers._open = _recording_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--photo", default="angled")
    ap.add_argument("--prompt", default="v5p")
    ap.add_argument("--style", default="english-country-cottage")
    args = ap.parse_args()

    deck = json.loads((HERE / "deck.json").read_text())
    deck_text, furniture = describe(deck)
    looks = json.loads((HERE / "styles.json").read_text())
    photo = inputs.prep_like_canvas((HERE / "photos" / f"{args.photo}.jpg").read_bytes(), 2048, 92)
    size = Image.open(io.BytesIO(photo)).size
    template = (HERE / "prompts" / f"{args.prompt}.txt").read_text()
    parts = build_parts(template, photo, lambda n: (HERE / "refs" / args.style / f"{n}.png").read_bytes(),
                        deck_text, furniture, looks[args.style])
    for model in args.models:
        USAGE.clear()
        t0 = time.time()
        try:
            image, _ = providers.edit(model, parts, size)
            out = Image.open(io.BytesIO(image)).size
        except providers.ProviderError as err:
            out = f"error: {err}"
        print(json.dumps({"model": model, "seconds": round(time.time() - t0, 1), "output": out, "usage": USAGE[-1] if USAGE else None}))


if __name__ == "__main__":
    main()

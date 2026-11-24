"""Did the image model keep the photo? Compares an output with the composite it was given, outside the pasted deck.

The score is the correlation of the two pictures' brightness gradients (direction and strength) on a 320 px wide grey
copy: new light, lawn and planting change it a little, a redrawn camera or house moves every edge and drops it to near 0.
On the judged rounds 13-15 (18 images) the five the judges marked as redrawn scored 0.01-0.45, all others 0.60-0.93.

  .venv/bin/python -m tuning.check        # scores the judged rounds again
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
WIDTH = 320
THRESHOLD = 0.55
DECK_THRESHOLD = 0.30  # inside the deck: rounds 13-21 (33 judged images), every image with judged geometry >= 1.5
#                        scored >= 0.33; 13 of the 16 with geometry <= 1 scored < 0.30


def _gradients(grey: np.ndarray, sigma: float = 1.5):
    import cv2

    g = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), sigma)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)


def drift(output: bytes, composite: bytes, deck_alpha: Image.Image) -> float:
    """How much of the composite's structure, outside the deck, the output kept: about 1 = all, about 0 = redrawn."""
    import cv2

    ref = Image.open(io.BytesIO(composite))
    size = (WIDTH, round(WIDTH * ref.height / ref.width))
    deck = np.asarray(deck_alpha.convert("L").resize(size)) > 8
    keep = ~(cv2.dilate(deck.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    rx, ry = _gradients(np.asarray(ref.convert("L").resize(size, Image.LANCZOS)))
    ox, oy = _gradients(np.asarray(Image.open(io.BytesIO(output)).convert("L").resize(size, Image.LANCZOS)))
    dot = (rx * ox + ry * oy)[keep].sum()
    norm = np.sqrt(((rx**2 + ry**2)[keep]).sum() * ((ox**2 + oy**2)[keep]).sum())
    return float(dot / norm) if norm > 0 else 0.0


def deck(output: bytes, composite: bytes, deck_alpha: Image.Image) -> float:
    """How well the output keeps the pasted deck's own lines (edges, railing, stairs) where the render drew them:
    the same gradient correlation, inside the deck, on a softer blur since real materials add texture."""
    import cv2

    ref = Image.open(io.BytesIO(composite))
    size = (WIDTH, round(WIDTH * ref.height / ref.width))
    inside = np.asarray(deck_alpha.convert("L").resize(size)) > 128
    inside = cv2.dilate(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    rx, ry = _gradients(np.asarray(ref.convert("L").resize(size, Image.LANCZOS)), 3.0)
    ox, oy = _gradients(np.asarray(Image.open(io.BytesIO(output)).convert("L").resize(size, Image.LANCZOS)), 3.0)
    dot = (rx * ox + ry * oy)[inside].sum()
    norm = np.sqrt(((rx**2 + ry**2)[inside]).sum() * ((ox**2 + oy**2)[inside]).sum())
    return float(dot / norm) if norm > 0 else 0.0


def passes(scores: dict) -> bool:
    return scores["photo"] >= THRESHOLD and scores["deck"] >= DECK_THRESHOLD


def rank(scores: dict) -> float:
    """For keeping the best of several failed tries: how far the worse of the two is from its threshold."""
    return min(scores["photo"] / THRESHOLD, scores["deck"] / DECK_THRESHOLD)


def main() -> None:
    from . import inputs

    style = "english-country-cottage"
    for run, base in (("r13-v9c", "composite"), ("r14-v9r", "composite"), ("r15-v10c", "composite-clean")):
        for photo in ("angled", "existing-deck", "regular"):
            jpeg = inputs.prep_like_canvas((HERE / "photos" / f"{photo}.jpg").read_bytes(), 2048, 92)
            comp = inputs.engine_image(base, style, photo, jpeg)
            alpha = Image.open(HERE / "engine" / style / f"{photo}.png").getchannel("A")
            for k in ("", "__2"):
                path = HERE / "out" / run / f"{photo}__{style}__gemini-3.1-flash-image{k}.jpg"
                if path.exists():
                    s = drift(path.read_bytes(), comp, alpha)
                    print(f"{run:9} {photo:14} {k or '#1':4} {s:.2f} {'ok' if s >= THRESHOLD else 'REDRAWN'}")


if __name__ == "__main__":
    main()

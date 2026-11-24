"""Runs one prompt over photos x styles x models and writes a comparison page.

  .venv/bin/python -m tuning.run --prompt v1 --styles english-country-cottage modern-minimalist \
      --models gemini-3-pro-image gpt-image-2.6-sunburst

Results go to tuning/out/<run>/: one JPEG per cell, run.json (prompt text, timings, errors) and index.html.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import html
import io
import json
import re
import time
from pathlib import Path

from PIL import Image

from . import providers
from . import inputs
from . import check
from .describe import describe

HERE = Path(__file__).resolve().parent
FLOOR_SEEN_FT = 1.0  # camera this far above the deck floor: the floor and its furniture are in view
VIEWS = ["top", "left-front", "right-front"]


def build_parts(template: str, photo: bytes, refs, deck_text: str, furniture: str, look: str, stairs: str = "") -> list:
    """[refs]: the images by name, a dict or a function of the name ([IMAGE <name>] in the template)."""
    text = (template.replace("{deck}", deck_text).replace("{furniture}", furniture).replace("{look}", look)
            .replace("{stairs}", stairs))
    image = refs if callable(refs) else refs.__getitem__
    parts: list = []
    for chunk in re.split(r"(\[IMAGE [\w-]+\])", text):
        m = re.fullmatch(r"\[IMAGE ([\w-]+)\]", chunk)
        if m:
            name = m.group(1)
            parts.append((name, photo if name == "photo" else image(name)))
        elif chunk.strip():
            parts.append(chunk.strip())
    return parts


def prompt_text(parts: list) -> str:
    return "\n\n".join(p if isinstance(p, str) else f"[image: {p[0]}]" for p in parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--photos", nargs="*", default=None)
    ap.add_argument("--styles", nargs="+", default=["english-country-cottage", "modern-minimalist"])
    ap.add_argument("--models", nargs="+", default=["gemini-3-pro-image", "gemini-3.1-flash-image"])
    ap.add_argument("--deck", default=str(HERE / "deck.json"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=1, help="samples per cell, to see how much one prompt varies")
    ap.add_argument("--scale", action="store_true", help="add the door-size scale line to the deck description")
    ap.add_argument("--retry", type=int, default=0,
                    help="redo a result up to N times when it redrew the photo or reshaped the deck (composite prompts, tuning.check)")
    ap.add_argument("--prep", choices=["none", "canvas", "hi"], default="none",
                    help="canvas: send the photo as the canvas prepares it (long edge 1536, JPEG 85)")
    args = ap.parse_args()

    def template_for(photo_stem: str) -> str:
        """"auto": the furniture list (v10k) only when the camera sees the deck floor, else none (v10j): told to keep
        furniture it cannot see, the model raises the camera to show it."""
        name = args.prompt
        if name == "auto":
            cams = json.loads((HERE / "out" / "camera" / "cameras.json").read_text())
            seen = cams[photo_stem]["camera_height_ft"] > deck["height_in"] / 12 + FLOOR_SEEN_FT
            name = "v10k" if seen else "v10j"
        chosen[photo_stem] = name
        return (HERE / "prompts" / f"{name}.txt").read_text()

    def stairs_for(photo_stem: str) -> str:
        """Where the design's stairs fall in this photo ({stairs} in a template), from its solved camera."""
        from . import camera
        cams = json.loads((HERE / "out" / "camera" / "cameras.json").read_text())
        cam = cams.get(photo_stem)
        return camera.stairs_note(deck, cam, cam["placement"]) if cam and "placement" in cam else ""

    chosen: dict[str, str] = {}
    looks = json.loads((HERE / "styles.json").read_text())
    deck = json.loads(Path(args.deck).read_text())
    deck_text, furniture = describe(deck, scale=args.scale)
    photos = sorted((HERE / "photos").glob("*.jpg"))
    if args.photos:
        photos = [p for p in photos if p.stem in args.photos]

    run = args.name or f"{dt.datetime.now():%m%d-%H%M}-{args.prompt}"
    out = HERE / "out" / run
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for photo in photos:
        data = photo.read_bytes()
        if args.prep == "canvas":
            data = inputs.prep_like_canvas(data)
        elif args.prep == "hi":
            data = inputs.prep_like_canvas(data, long_edge=2048, quality=92)
        size = Image.open(io.BytesIO(data)).size
        for style in args.styles:
            def ref(name: str, style=style, photo=photo, data=data) -> bytes:
                if name == "matched":
                    return inputs.matched_ref(style, inputs.photo_angle(photo.stem, inputs.prep_like_canvas(data)))
                if name == "plan":
                    return inputs.plan_png(deck)
                if name == "clean":
                    return inputs.cleaned(photo.stem, data)
                if name in ("render", "composite", "composite-clean"):
                    return inputs.engine_image(name, style, photo.stem, data)
                return (HERE / "refs" / style / f"{name}.png").read_bytes()

            parts = build_parts(template_for(photo.stem), data, ref, deck_text, furniture, looks[style],
                                stairs_for(photo.stem))
            for model in args.models:
                for k in range(args.repeat):
                    jobs.append((photo.stem, style, model, parts, size, k))

    def work(job):
        photo, style, model, parts, size, k = job
        cell = f"{photo}__{style}__{model}" + (f"__{k + 1}" if k else "")
        t0 = time.time()
        comp_part = next((p for p in parts if not isinstance(p, str) and p[0].startswith("composite")), None)
        comp = comp_part[1] if comp_part else None
        ignore = None
        if comp_part and comp_part[0] == "composite":  # pasted on the photo itself: the model removes the old deck
            from . import camera
            old = (json.loads(camera.CACHE.read_text()).get(photo, {}) if camera.CACHE.exists() else {}).get("existing_deck")
            ignore = [old["box"]] if old else None
        try:
            scores, best = [], None
            for attempt in range(args.retry + 1):
                image, note = providers.edit(model, parts, size)
                if comp is None:
                    best = (0, image, note)
                    break
                alpha = Image.open(HERE / "engine" / style / f"{photo}.png").getchannel("A")
                sc = {"photo": round(check.drift(image, comp, alpha, ignore), 2), "deck": round(check.deck(image, comp, alpha), 2)}
                scores.append(sc)
                if best is None or check.rank(sc) > best[0]:
                    best = (check.rank(sc), image, note)
                if check.passes(sc):
                    break
            _, image, note = best  # every try redrew the photo: keep the one that kept the most
            (out / f"{cell}.jpg").write_bytes(providers.as_jpeg(image))
            return {"cell": cell, "photo": photo, "style": style, "model": model, "sample": k + 1, "ok": True,
                    "seconds": round(time.time() - t0, 1), "note": note, "drift": scores}
        except Exception as err:  # keep going: one failure is a result too
            return {"cell": cell, "photo": photo, "style": style, "model": model, "sample": k + 1, "ok": False,
                    "seconds": round(time.time() - t0, 1), "error": str(err)[:500]}

    results = []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for r in pool.map(work, jobs):
            results.append(r)
            print(("ok  " if r["ok"] else "ERR ") + r["cell"], r["seconds"], "s", r.get("drift", ""), r.get("error", ""), flush=True)

    sample = jobs[0][3] if jobs else []
    meta = {"run": run, "prompt": args.prompt, "prompts": chosen, "prep": args.prep, "retry": args.retry, "styles": args.styles, "models": args.models, "repeat": args.repeat,
            "photos": [p.stem for p in photos], "prompt_text": prompt_text(sample), "results": results}
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    write_page(out, meta)
    print("page:", out / "index.html")


def write_page(out: Path, meta: dict) -> None:
    by = {r["cell"]: r for r in meta["results"]}
    rows = []
    for photo in meta["photos"]:
        for style in meta["styles"]:
            cells = [f'<td class="src"><img src="../../photos/{photo}.jpg"><div>{html.escape(photo)}<br>{html.escape(style)}</div></td>']
            for model, k in [(m, k) for m in meta["models"] for k in range(meta.get("repeat", 1))]:
                r = by.get(f"{photo}__{style}__{model}" + (f"__{k + 1}" if k else ""), {})
                if r.get("ok"):
                    cells.append(f'<td><a href="{r["cell"]}.jpg" target="_blank"><img src="{r["cell"]}.jpg"></a><div>{r["seconds"]} s</div></td>')
                else:
                    cells.append(f'<td class="err">{html.escape(r.get("error", "not run"))}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "<tr><th>photo / style</th>" + "".join(
        f"<th>{html.escape(m)}{f' #{k + 1}' if meta.get('repeat', 1) > 1 else ''}</th>"
        for m in meta["models"] for k in range(meta.get("repeat", 1))) + "</tr>"
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(meta['run'])}</title>
<style>body{{font:13px system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px;vertical-align:top}}
img{{width:340px;display:block}}td.src img{{width:200px}}td.err{{color:#b00;width:340px;white-space:pre-wrap}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}</style>
<h1>{html.escape(meta['run'])}</h1><table>{head}{''.join(rows)}</table>
<h2>Prompt</h2><pre>{html.escape(meta['prompt_text'])}</pre>"""
    (out / "index.html").write_text(page)


if __name__ == "__main__":
    main()

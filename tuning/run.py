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
from .describe import describe

HERE = Path(__file__).resolve().parent
VIEWS = ["top", "left-front", "right-front"]


def build_parts(template: str, photo: bytes, refs: dict[str, bytes], deck_text: str, furniture: str, look: str) -> list:
    text = template.replace("{deck}", deck_text).replace("{furniture}", furniture).replace("{look}", look)
    parts: list = []
    for chunk in re.split(r"(\[IMAGE [\w-]+\])", text):
        m = re.fullmatch(r"\[IMAGE ([\w-]+)\]", chunk)
        if m:
            name = m.group(1)
            parts.append((name, photo if name == "photo" else refs[name]))
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
    args = ap.parse_args()

    template = (HERE / "prompts" / f"{args.prompt}.txt").read_text()
    looks = json.loads((HERE / "styles.json").read_text())
    deck_text, furniture = describe(json.loads(Path(args.deck).read_text()), scale=args.scale)
    photos = sorted((HERE / "photos").glob("*.jpg"))
    if args.photos:
        photos = [p for p in photos if p.stem in args.photos]

    run = args.name or f"{dt.datetime.now():%m%d-%H%M}-{args.prompt}"
    out = HERE / "out" / run
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for photo in photos:
        data = photo.read_bytes()
        size = Image.open(io.BytesIO(data)).size
        for style in args.styles:
            refs = {v: (HERE / "refs" / style / f"{v}.png").read_bytes() for v in VIEWS}
            parts = build_parts(template, data, refs, deck_text, furniture, looks[style])
            for model in args.models:
                for k in range(args.repeat):
                    jobs.append((photo.stem, style, model, parts, size, k))

    def work(job):
        photo, style, model, parts, size, k = job
        cell = f"{photo}__{style}__{model}" + (f"__{k + 1}" if k else "")
        t0 = time.time()
        try:
            image, note = providers.edit(model, parts, size)
            (out / f"{cell}.jpg").write_bytes(providers.as_jpeg(image))
            return {"cell": cell, "photo": photo, "style": style, "model": model, "sample": k + 1, "ok": True,
                    "seconds": round(time.time() - t0, 1), "note": note}
        except Exception as err:  # keep going: one failure is a result too
            return {"cell": cell, "photo": photo, "style": style, "model": model, "sample": k + 1, "ok": False,
                    "seconds": round(time.time() - t0, 1), "error": str(err)[:500]}

    results = []
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for r in pool.map(work, jobs):
            results.append(r)
            print(("ok  " if r["ok"] else "ERR ") + r["cell"], r["seconds"], "s", r.get("error", ""), flush=True)

    sample = jobs[0][3] if jobs else []
    meta = {"run": run, "prompt": args.prompt, "styles": args.styles, "models": args.models, "repeat": args.repeat,
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

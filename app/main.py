"""FastAPI server for the Web AR deck preview PoC."""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal
from urllib.parse import quote, unquote

import segno
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from . import photo
from .deck import FT, build_deck, make_layout
from .gemini import GeminiError
from .gltf import write_glb
from .usdz import write_usdz

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
GENERATED_DIR = ROOT / "generated"
GEOMETRY_VERSION = 6  # bump to invalidate cached models after geometry changes

USDZ_MEDIA_TYPE = "model/vnd.usdz+zip"
GLB_MEDIA_TYPE = "model/gltf-binary"
MAX_PHOTO_BYTES = 25 * 1024 * 1024
# AR troubleshooting: what the page reports after AR Preview, and every model download (Scene
# Viewer and Quick Look fetch the model themselves, so a download means the viewer did open)
DIAG_PATH = GENERATED_DIR / "diagnostics.jsonl"
MAX_DIAG_BYTES = 32 * 1024
_diag_lock = threading.Lock()

app = FastAPI(title="Deck Web AR PoC")


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    """Make browsers revalidate the page and its assets so UI changes show up on phones right away."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class ModelRequest(BaseModel):
    shape: Literal["rect", "l", "l_left", "u", "angled"]
    dims: dict[str, int]  # feet, keyed by edge label (plus "wall" for the U-shape house depth)
    height_ft: int = Field(ge=2, le=8)
    stair_edge: str | None = Field(default=None, pattern=r"^[A-Z]$")  # None: no stairs

    @model_validator(mode="after")
    def check_layout(self):
        layout = make_layout(self.shape, self.dims)  # ValueError -> 422
        if self.stair_edge and (problem := layout.stair_problem(layout.edge_index(self.stair_edge))):
            raise ValueError(f"{self.stair_edge}: {problem}")
        return self


class ModelResponse(BaseModel):
    id: str
    url: str  # USDZ, for AR Quick Look on iOS
    glb_url: str  # GLB, for Scene Viewer on Android
    size_m: tuple[float, float, float]  # width, height incl. railing, depth as placed in AR
    bytes: int


def _model_id(req: ModelRequest) -> str:
    key = json.dumps({"v": GEOMETRY_VERSION, **req.model_dump()}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@app.post("/api/models", response_model=ModelResponse)
def create_model(req: ModelRequest, background: BackgroundTasks) -> ModelResponse:
    model_id = _model_id(req)
    out_path = GENERATED_DIR / f"{model_id}.usdz"
    glb_path = GENERATED_DIR / f"{model_id}.glb"
    meta_path = GENERATED_DIR / f"{model_id}.json"

    if not (out_path.exists() and glb_path.exists() and meta_path.exists()):
        layout = make_layout(req.shape, req.dims)
        stair = layout.edge_index(req.stair_edge) if req.stair_edge else None
        model = build_deck(layout, req.height_ft * FT, stair)
        write_usdz(model, out_path)
        write_glb(model, glb_path)
        lo, hi = model.bounds()  # includes stairs and railing overhang
        size = tuple(round(hi[i] - lo[i], 4) for i in range(3))
        meta_path.write_text(json.dumps({"size_m": size, "request": req.model_dump()}))
    # Render the photo references now, so "Show in Photo" doesn't wait for them
    background.add_task(photo.ensure_references, GENERATED_DIR, model_id)

    meta = json.loads(meta_path.read_text())
    return ModelResponse(
        id=model_id,
        url=f"/models/{model_id}.usdz",
        glb_url=f"/models/{model_id}.glb",
        size_m=tuple(meta["size_m"]),
        bytes=out_path.stat().st_size,
    )


def _client(request: Request) -> dict:
    """Who sent a request; behind a tunnel or proxy the real address is in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return {"ip": forwarded or (request.client.host if request.client else None),
            "ua": request.headers.get("user-agent"), "referer": request.headers.get("referer")}


def _log_diag(record: dict) -> None:
    line = json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}, ensure_ascii=False)
    with _diag_lock:
        DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DIAG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


@app.post("/api/diag")
async def diagnostics(request: Request):
    body = await request.body()
    if len(body) > MAX_DIAG_BYTES:
        raise HTTPException(413, "Report too large.")
    try:
        report = json.loads(body)
    except ValueError:
        raise HTTPException(400, "Report must be JSON.")
    _log_diag({"event": "ar_report", **_client(request), "report": report})
    return {"ok": True}


@app.api_route("/models/{model_id}.{ext}", methods=["GET", "HEAD"])
def get_model(model_id: str, ext: str, request: Request):
    media_type = {"usdz": USDZ_MEDIA_TYPE, "glb": GLB_MEDIA_TYPE}.get(ext)
    if not model_id.isalnum() or not media_type:
        raise HTTPException(404)
    path = GENERATED_DIR / f"{model_id}.{ext}"
    _log_diag({"event": "model_download", "file": f"{model_id}.{ext}", "method": request.method,
               "found": path.exists(), "query": request.url.query or None, "range": request.headers.get("range"),
               **_client(request)})
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type=media_type, filename=f"deck-{model_id}.{ext}")


async def _photo_body(request: Request) -> bytes:
    body = await request.body()
    if not body:
        raise HTTPException(400, "No photo received.")
    if len(body) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "That photo is too large.")
    return body


async def _gemini_response(fn, *args) -> Response:
    try:
        data, mime = await run_in_threadpool(fn, *args)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found. Go back and generate it again.")
    except photo.PhotoError as err:
        return JSONResponse({"detail": str(err)}, status_code=400)
    except GeminiError as err:
        return JSONResponse({"detail": str(err)}, status_code=502)
    return Response(data, media_type=mime)


def _prompt_template(request: Request) -> str | None:
    """An edited prompt template from the photo screen, sent URI-encoded in a header next to the image body."""
    value = request.headers.get("x-prompt-template")
    return unquote(value) if value else None


@app.get("/api/models/{model_id}/prompts")
def model_prompts(model_id: str):
    """Default prompt templates and what {deck} and {style} expand to for this model."""
    if not model_id.isalnum():
        raise HTTPException(404)
    try:
        return photo.prompt_info(GENERATED_DIR, model_id)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found.")


@app.get("/models/{model_id}/ref/{view}.png")
async def reference_image(model_id: str, view: str, az: int = 0, el: int = 10):
    """A reference image exactly as sent to Gemini (for the prompt panel). view "matched" takes the photo's az/el."""
    if not model_id.isalnum() or view not in (*photo.REFERENCE_VIEWS, "matched"):
        raise HTTPException(404)
    try:
        if view == "matched":
            data = await run_in_threadpool(photo.matched_reference, GENERATED_DIR, model_id, max(-75, min(75, az)), max(0, min(35, el)))
        else:
            data = (await run_in_threadpool(photo.ensure_references, GENERATED_DIR, model_id))[view]
    except FileNotFoundError:
        raise HTTPException(404, "Model not found.")
    return Response(data, media_type="image/png")


@app.post("/api/photo/analyze")
async def analyze_photo(request: Request, kind: str = "wall"):
    """Body: the homeowner's photo. Returns where a deck of this snap kind (wall, corner, bump) could go
    (see photo.analyze_photo)."""
    body = await _photo_body(request)
    try:
        return await run_in_threadpool(lambda: photo.analyze_photo(photo.prepare_photo(body)[0], kind))
    except photo.PhotoError as err:
        return JSONResponse({"detail": str(err)}, status_code=400)


@app.get("/api/photo/styles")
def photo_styles():
    return photo.style_list()


@app.post("/api/models/{model_id}/photo")
async def design_in_photo(model_id: str, request: Request, style: str = "cedar"):
    """Body: the homeowner's photo. Returns it with this model's deck added in `style`."""
    if not model_id.isalnum():
        raise HTTPException(404)
    body = await _photo_body(request)
    template = _prompt_template(request)
    try:
        # Where the homeowner placed the deck on the photo (URI-encoded JSON); without it, the recommended wall
        raw = request.headers.get("x-placement")
        placement = json.loads(unquote(raw)) if raw else None
        data, mime, view, placement, check = await run_in_threadpool(photo.design, GENERATED_DIR, model_id, body, style, template, placement)
    except FileNotFoundError:
        raise HTTPException(404, "Model not found. Go back and generate it again.")
    except (photo.PhotoError, ValueError) as err:
        return JSONResponse({"detail": str(err)}, status_code=400)
    except GeminiError as err:
        return JSONResponse({"detail": str(err)}, status_code=502)
    # The viewpoint analysis, for the prompt panel (URI-encoded JSON: headers must be ASCII)
    analysis = {"wall": view.wall, "azimuth": view.azimuth, "elevation": view.elevation,
                "wall_length_ft": view.wall_length_ft, "analyzed": view.analyzed, "viewpoint": view.phrase()}
    headers = {"X-View-Analysis": quote(json.dumps(analysis)), "X-Check": quote(json.dumps(check))}
    return Response(data, media_type=mime, headers=headers)


@app.post("/api/photo/restyle")
async def restyle_photo(request: Request, style: str):
    """Body: a design result. Returns it with only the deck's material changed to `style`."""
    return await _gemini_response(photo.restyle, await _photo_body(request), style, _prompt_template(request))


def _lan_ip() -> str:
    """A private (Wi-Fi/Ethernet) IPv4 address on macOS, Linux or Windows, preferring 192.168.x.x."""
    found: list[str] = []
    try:  # the interface of the default route (may be a VPN, so not trusted alone)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # UDP connect sends no packets
            found.append(s.getsockname()[0])
    except OSError:
        pass
    if platform.system() == "Windows":
        cmd, pattern = ["ipconfig"], r"IPv4[^:\n]*:\s*(\d+\.\d+\.\d+\.\d+)"
    elif shutil.which("ip"):
        cmd, pattern = ["ip", "-4", "-o", "addr"], r"inet (\d+\.\d+\.\d+\.\d+)"
    else:
        cmd, pattern = ["ifconfig"], r"inet (\d+\.\d+\.\d+\.\d+)"
    try:
        found += re.findall(pattern, subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout)
    except (OSError, subprocess.SubprocessError):
        pass

    lan = {ipaddress.ip_address(ip) for ip in found}
    lan = [ip for ip in lan if ip.is_private and not ip.is_loopback and not ip.is_link_local]
    # Home/office Wi-Fi is usually 192.168.x.x, then 10.x / 172.16-31.x
    lan.sort(key=lambda ip: (not str(ip).startswith("192.168."), str(ip)))
    return str(lan[0]) if lan else "127.0.0.1"


@app.get("/api/share")
def share(request: Request):
    """The URL a phone on the same Wi-Fi can open (override with WEBAR_PUBLIC_URL)."""
    if public_url := os.environ.get("WEBAR_PUBLIC_URL"):
        return {"url": public_url}
    host = request.url.hostname
    if host in ("localhost", "127.0.0.1", "::1"):
        host = _lan_ip()
    port = request.url.port
    url = f"{request.url.scheme}://{host}{f':{port}' if port else ''}/"
    return {"url": url}


@app.get("/api/qr.svg")
def qr(url: str):
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=6, dark="#1d2a23", light="#ffffff", border=2)
    return Response(buf.getvalue(), media_type="image/svg+xml")


@app.get("/")
def index():
    # Version asset URLs by modification time so phones never run a stale script
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for name in ("app.js", "style.css"):
        version = int((STATIC_DIR / name).stat().st_mtime)
        html = html.replace(f"/static/{name}", f"/static/{name}?v={version}")
    return Response(html, media_type="text/html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

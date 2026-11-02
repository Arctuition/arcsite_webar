"""Software renderer for reference images (top view and 45° view) of a USDZ deck.

Reads the meshes and their UsdPreviewSurface colors back out of the USDZ, adds
a gray block for the house, and z-buffers flat-shaded triangles with
numpy. No GPU or OpenGL needed, so it runs the same on macOS, Linux and Windows.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

from .deck import Layout, extrude, layout_center

HOUSE_COLOR = (0.62, 0.64, 0.68)
BACKGROUND = (222, 224, 228)  # light gray, so the white railings still read
HOUSE_ABOVE_DECK = 1.2  # meters; low enough not to hide the deck in the 45° view
LIGHT = np.array([-0.35, 1.0, 0.55]) / np.linalg.norm([-0.35, 1.0, 0.55])
SUPERSAMPLE = 2


def _linear_to_srgb(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def load_usdz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """All mesh triangles (N, 3, 3) in world space and their sRGB colors (N, 3)."""
    stage = Usd.Stage.Open(path)
    tris, colors = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        xform = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        pts = np.c_[pts, np.ones(len(pts))] @ xform  # USD matrices are row-major, points are row vectors
        pts = pts[:, :3]

        color = (0.7, 0.7, 0.7)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            shader = material.ComputeSurfaceSource()[0]
            diffuse = shader.GetInput("diffuseColor") if shader else None
            if diffuse and diffuse.Get() is not None:
                color = tuple(_linear_to_srgb(c) for c in diffuse.Get())

        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        k = 0
        for n in counts:
            face = indices[k : k + n]
            k += n
            for i in range(1, n - 1):  # fan triangulation (faces are convex)
                tris.append(pts[[face[0], face[i], face[i + 1]]])
                colors.append(color)
    return np.array(tris), np.array(colors)


def house_geometry(layout: Layout, deck_top: float) -> tuple[np.ndarray, np.ndarray]:
    """The house as a plain gray block, in the model's centered frame."""
    cx, cz = layout_center(layout)
    house = [(x - cx, z - cz) for x, z in layout.house_footprint()]
    tris = [[f[0], f[i], f[i + 1]] for f in extrude(house, 0.0, deck_top + HOUSE_ABOVE_DECK) for i in range(1, len(f) - 1)]
    return np.array(tris, dtype=float), np.tile(HOUSE_COLOR, (len(tris), 1))


# Perspective views as (azimuth from straight in front, elevation) in degrees; "top" is an orthographic plan
VIEW_ANGLES = {
    "oblique": (45, 45),  # from the front-right corner, looking 45° down
    "front": (0, 20),  # standing in the yard facing the house, like most photos
}


def render(tris: np.ndarray, colors: np.ndarray, view: str | tuple[float, float], fit: np.ndarray, size=(1280, 960)) -> Image.Image:
    """view: "top" (orthographic plan, house at the top), a key of VIEW_ANGLES, or (azimuth, elevation) in degrees.
    The camera frames the triangles in `fit`."""
    w, h = size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE
    lo, hi = fit.reshape(-1, 3).min(axis=0), fit.reshape(-1, 3).max(axis=0)
    target = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2

    if view == "top":
        forward, up, perspective = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0]), False
        eye = target - forward * radius * 3
    else:
        azimuth, elevation = (math.radians(a) for a in (VIEW_ANGLES[view] if isinstance(view, str) else view))
        to_eye = np.array([math.sin(azimuth) * math.cos(elevation), math.sin(elevation), math.cos(azimuth) * math.cos(elevation)])
        forward, up, perspective = -to_eye, np.array([0.0, 1.0, 0.0]), True
        eye = target + to_eye * radius * 2.6

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)

    # Flat shading and back-face culling from the outward face normals
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-12
    tris, colors, normals = tris[keep], colors[keep], normals[keep] / lengths[keep, None]
    view_dirs = (tris.mean(axis=1) - eye) if perspective else np.broadcast_to(forward, normals.shape)
    facing = np.einsum("ij,ij->i", normals, view_dirs) < 0
    tris, colors, normals = tris[facing], colors[facing], normals[facing]
    shade = 0.62 + 0.38 * np.clip(normals @ LIGHT, 0, 1)
    shaded = np.clip(colors * shade[:, None], 0, 1)

    # Camera space, then project; depth key grows toward the camera
    rel = tris - eye
    cx, cy, cz = rel @ right, rel @ cam_up, rel @ forward
    if perspective:
        sx, sy, key = cx / cz, cy / cz, 1 / cz
    else:
        sx, sy, key = cx, cy, -cz

    # Fit the projection of `fit` into the image with a margin
    frel = fit.reshape(-1, 3) - eye
    fx, fy, fz = frel @ right, frel @ cam_up, frel @ forward
    if perspective:
        fx, fy = fx / fz, fy / fz
    margin = 0.06
    span = max((fx.max() - fx.min()) / (w * (1 - 2 * margin)), (fy.max() - fy.min()) / (h * (1 - 2 * margin)))
    mid_x, mid_y = (fx.max() + fx.min()) / 2, (fy.max() + fy.min()) / 2
    px = (sx - mid_x) / span + w / 2
    py = h / 2 - (sy - mid_y) / span

    zbuf = np.full((h, w), -np.inf)
    img = np.empty((h, w, 3))
    img[:] = np.array(BACKGROUND) / 255
    for t in range(len(px)):
        x, y, k = px[t], py[t], key[t]
        x0, x1 = max(int(x.min()), 0), min(int(math.ceil(x.max())), w - 1)
        y0, y1 = max(int(y.min()), 0), min(int(math.ceil(y.max())), h - 1)
        if x0 > x1 or y0 > y1:
            continue
        area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((x[1] - gx) * (y[2] - gy) - (x[2] - gx) * (y[1] - gy)) / area
        w1 = ((x[2] - gx) * (y[0] - gy) - (x[0] - gx) * (y[2] - gy)) / area
        w2 = 1 - w0 - w1
        depth = w0 * k[0] + w1 * k[1] + w2 * k[2]
        region = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        hit = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (depth > region)
        region[hit] = depth[hit]
        img[y0 : y1 + 1, x0 : x1 + 1][hit] = shaded[t]

    image = Image.fromarray((img * 255).astype(np.uint8))
    return image.resize(size, Image.LANCZOS)


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _scene(usdz_path: Path, layout: Layout, deck_top: float) -> tuple[np.ndarray, np.ndarray]:
    tris, colors = load_usdz(str(usdz_path))
    house, house_colors = house_geometry(layout, deck_top)
    return np.concatenate([tris, house]), np.concatenate([colors, house_colors])


def reference_images(usdz_path: Path, layout: Layout, deck_top: float, views=("top", "oblique")) -> dict:
    """PNG reference images keyed by view (see render): the deck and a gray house block on a plain background.
    Deliberately diagram-like (no ground, no scenery) so an image model treats them as a design, not a scene."""
    scene, scene_colors = _scene(usdz_path, layout, deck_top)
    return {view: png_bytes(render(scene, scene_colors, view, scene)) for view in views}

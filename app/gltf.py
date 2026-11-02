"""Write a DeckModel to a binary glTF (GLB) for Android's Scene Viewer.

Same geometry and colors as the USDZ: one mesh per material (Scene Viewer recommends a
single mesh per material), flat face normals, meters, +Y up. Faces are convex polygons,
so each is split into a fan of triangles.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

from .deck import DeckModel
from .usdz import MATERIALS, _srgb_to_linear

GLB_MAGIC, GLB_VERSION = 0x46546C67, 2  # "glTF"
CHUNK_JSON, CHUNK_BIN = 0x4E4F534A, 0x004E4942
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
FLOAT, UNSIGNED_INT = 5126, 5125


def _newell(face: np.ndarray) -> np.ndarray:
    nxt = np.roll(face, -1, axis=0)
    n = np.array([
        np.sum((face[:, 1] - nxt[:, 1]) * (face[:, 2] + nxt[:, 2])),
        np.sum((face[:, 2] - nxt[:, 2]) * (face[:, 0] + nxt[:, 0])),
        np.sum((face[:, 0] - nxt[:, 0]) * (face[:, 1] + nxt[:, 1])),
    ])
    return n / (np.linalg.norm(n) or 1.0)


def _mesh_arrays(solids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, normals and triangle indices; faces don't share vertices, so normals stay flat."""
    positions, normals, indices = [], [], []
    for solid in solids:
        for face in solid.faces():
            pts = np.asarray(face, dtype=np.float64)
            base = sum(len(p) for p in positions)
            positions.append(pts)
            normals.append(np.repeat(_newell(pts)[None, :], len(pts), axis=0))
            for i in range(1, len(pts) - 1):  # counter-clockwise fan keeps the outward winding
                indices.extend((base, base + i, base + i + 1))
    return (np.concatenate(positions).astype(np.float32), np.concatenate(normals).astype(np.float32),
            np.asarray(indices, dtype=np.uint32))


def write_glb(model: DeckModel, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_material = defaultdict(list)
    for solid in model.solids:
        by_material[solid.material].append(solid)

    blob = bytearray()
    views, accessors, materials, meshes = [], [], [], []

    def add_view(data: bytes, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(views) - 1

    def add_accessor(array: np.ndarray, target: int, kind: str, component: int, bounds: bool = False) -> int:
        accessor = {"bufferView": add_view(array.tobytes(), target), "componentType": component,
                    "count": len(array), "type": kind}
        if bounds:  # required for POSITION
            accessor["min"], accessor["max"] = array.min(axis=0).tolist(), array.max(axis=0).tolist()
        accessors.append(accessor)
        return len(accessors) - 1

    for name, solids in by_material.items():
        srgb, roughness = MATERIALS[name]
        materials.append({
            "name": name,
            # glTF base color factors are linear, like UsdPreviewSurface's diffuseColor
            "pbrMetallicRoughness": {"baseColorFactor": [*(_srgb_to_linear(c) for c in srgb), 1.0],
                                     "metallicFactor": 0.0, "roughnessFactor": roughness},
        })
        positions, normals, indices = _mesh_arrays(solids)
        meshes.append({"name": name, "primitives": [{
            "attributes": {"POSITION": add_accessor(positions, ARRAY_BUFFER, "VEC3", FLOAT, bounds=True),
                           "NORMAL": add_accessor(normals, ARRAY_BUFFER, "VEC3", FLOAT)},
            "indices": add_accessor(indices, ELEMENT_ARRAY_BUFFER, "SCALAR", UNSIGNED_INT),
            "material": len(materials) - 1,
        }]})

    gltf = {
        "asset": {"version": "2.0", "generator": "deck-ar"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Deck", "children": list(range(1, len(meshes) + 1))}]
                 + [{"name": m["name"], "mesh": i} for i, m in enumerate(meshes)],
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    bin_chunk = bytes(blob) + b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out_path.write_bytes(
        struct.pack("<III", GLB_MAGIC, GLB_VERSION, total)
        + struct.pack("<II", len(json_chunk), CHUNK_JSON) + json_chunk
        + struct.pack("<II", len(bin_chunk), CHUNK_BIN) + bin_chunk
    )

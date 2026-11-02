"""Write a DeckModel to an ARKit-compatible USDZ package."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

from .deck import DeckModel

# sRGB colors; converted to linear for UsdPreviewSurface
MATERIALS = {
    "decking_a": ((0.56, 0.37, 0.23), 0.75),
    "decking_b": ((0.50, 0.32, 0.20), 0.75),
    "decking_c": ((0.60, 0.41, 0.26), 0.75),
    "framing": ((0.64, 0.56, 0.42), 0.9),
    "railing": ((0.93, 0.92, 0.89), 0.6),
    "concrete": ((0.62, 0.62, 0.60), 0.95),
}

def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _define_material(stage: Usd.Stage, path: str, srgb, roughness: float) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(_srgb_to_linear(c) for c in srgb)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _face_normal(face) -> Gf.Vec3f:
    """Newell's method; faces are counter-clockwise seen from outside."""
    nx = ny = nz = 0.0
    for (x0, y0, z0), (x1, y1, z1) in zip(face, face[1:] + face[:1]):
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return Gf.Vec3f(nx, ny, nz).GetNormalized()


def _define_mesh(stage: Usd.Stage, path: str, solids) -> UsdGeom.Mesh:
    # Faces don't share vertices so every face gets a flat normal
    points, counts, indices, normals = [], [], [], []
    for solid in solids:
        for face in solid.faces():
            counts.append(len(face))
            indices.extend(range(len(points), len(points) + len(face)))
            points.extend(Gf.Vec3f(*p) for p in face)
            normals.extend([_face_normal(face)] * len(face))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateNormalsAttr(Vt.Vec3fArray(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(Vt.Vec3fArray(points)))
    return mesh


def write_usdz(model: DeckModel, out_path: Path, scale: float = 1.0) -> None:
    """Write `model` (meters) to `out_path`, uniformly scaled by `scale` (e.g. 1/72)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        usdc_path = str(Path(tmp) / "deck.usdc")
        stage = Usd.Stage.CreateNew(usdc_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root = UsdGeom.Xform.Define(stage, "/Deck")
        stage.SetDefaultPrim(root.GetPrim())
        Usd.ModelAPI(root.GetPrim()).SetKind(Kind.Tokens.component)
        if scale != 1.0:
            root.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

        # AR Quick Look anchoring hint: place on a horizontal plane (floor / table)
        prim = root.GetPrim()
        prim.CreateAttribute("preliminary:anchoring:type", Sdf.ValueTypeNames.Token, custom=False).Set("plane")
        prim.CreateAttribute("preliminary:planeAnchoring:alignment", Sdf.ValueTypeNames.Token, custom=False).Set(
            "horizontal"
        )

        materials = {
            name: _define_material(stage, f"/Deck/Materials/{name}", color, rough)
            for name, (color, rough) in MATERIALS.items()
        }

        by_material = defaultdict(list)
        for solid in model.solids:
            by_material[solid.material].append(solid)

        UsdGeom.Scope.Define(stage, "/Deck/Geom")
        for name, solids in by_material.items():
            mesh = _define_mesh(stage, f"/Deck/Geom/{name}", solids)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[name])

        stage.GetRootLayer().Save()
        if not UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(usdc_path), str(out_path)):
            raise RuntimeError("USDZ packaging failed")

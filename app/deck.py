"""Procedural deck geometry.

A deck is a footprint polygon (plan coordinates x, z in meters; z grows away
from the house) whose edges are either house walls or open edges labeled A, B,
C... Open edges get a railing; one open edge can carry stairs. Edges may be
axis-aligned or at 45 degrees.

All parts are solids made of planar faces. Members that meet are mitered or
butted so no two faces are coplanar and overlapping; coplanar overlaps
z-fight (flicker) in AR. The final model is re-centered so the footprint center
sits at the origin, with the ground at y = 0 where AR Quick Look anchors it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

FT = 0.3048
IN = 0.0254

# Nominal lumber -> actual dimensions (meters)
DECK_BOARD_W = 5.5 * IN
DECK_BOARD_T = 1.0 * IN
DECK_BOARD_GAP = 0.125 * IN
JOIST_T = 1.5 * IN  # 2x8
JOIST_H = 7.25 * IN
JOIST_SPACING = 16 * IN
BEAM_T = 3.0 * IN  # doubled 2x10
BEAM_H = 9.25 * IN
BEAM_SETBACK = 1 * FT  # joist cantilever past a beam at an open edge
MAX_BEAM_SPACING = 10 * FT
POST_SIZE = 5.5 * IN  # 6x6
MAX_POST_SPACING = 8 * FT
FOOTING_SIZE = 16 * IN
FOOTING_H = 2 * IN

RAIL_POST_SIZE = 3.5 * IN  # 4x4
RAIL_INSET = RAIL_POST_SIZE / 2 + 0.5 * IN  # post centerline from the deck edge
RAIL_HEIGHT = 36 * IN
MAX_RAIL_POST_SPACING = 6 * FT
TOP_RAIL_W = 5.5 * IN
TOP_RAIL_T = 1.5 * IN
BOTTOM_RAIL_W = 2.5 * IN  # narrower than the posts so faces never line up
BOTTOM_RAIL_T = 1.5 * IN
BOTTOM_RAIL_GAP = 3.5 * IN
BALUSTER_SIZE = 1.5 * IN
BALUSTER_SPACING = 5.5 * IN  # on center -> 4" clear gap
EMBED = 0.25 * IN  # how far members sink into the member they sit on

STAIR_WIDTH = 4 * FT
STAIR_MIN_EDGE = STAIR_WIDTH + 1.5 * FT  # room for the stair opening plus railing posts
STAIR_MAX_RISE = 7.5 * IN
STAIR_RUN = 10 * IN
STAIR_NOSING = 1 * IN
STAIR_TREAD_T = 1.0 * IN
STAIR_RISER_T = 0.75 * IN
STRINGER_T = 1.5 * IN  # 2x12, closed stringer on each side
STRINGER_ABOVE_NOSING = 2 * IN
STRINGER_DEPTH = 12 * IN  # measured vertically

DECKING_MATERIALS = ("decking_a", "decking_b", "decking_c")

Vec2 = tuple[float, float]  # plan (x, z)
Vec3 = tuple[float, float, float]


# ---------- shapes ----------

# Allowed range (feet) for each dimension key, before cross-checks
SHAPE_DIMS: dict[str, dict[str, tuple[int, int]]] = {
    "rect": {"A": (6, 30), "B": (8, 40)},  # A = C = depth, B = width
    "l": {"A": (4, 26), "B": (12, 40), "C": (10, 30), "D": (4, 36)},  # wraps to the right: A short side, C full side
    "l_left": {"A": (10, 30), "B": (12, 40), "C": (4, 26), "D": (4, 36)},  # mirror image: A full side, C short side
    "u": {"A": (10, 30), "B": (16, 48), "D": (4, 20), "E": (4, 20), "wall": (4, 26)},  # A = C
    "angled": {"A": (4, 24), "B": (2, 10), "C": (6, 30)},  # A = E sides, B = D 45° corners
}
MIN_HOUSE_FT = 4
MAX_WIDTH_FT = 48


@dataclass(frozen=True)
class Edge:
    kind: str  # "open" | "house"
    label: str | None = None


@dataclass(frozen=True)
class Layout:
    points: list[Vec2]  # meters
    edges: list[Edge]  # edge i runs from points[i] to points[i + 1]

    def edge_length(self, i: int) -> float:
        a, b = self.points[i], self.points[(i + 1) % len(self.points)]
        return math.dist(a, b)

    def house_footprint(self, behind: float = 2.0) -> list[Vec2]:
        """Rectangle (meters) for the house: it fills any notch the deck wraps and runs `behind` past z = 0."""
        walls = [p for i, e in enumerate(self.edges) if e.kind == "house" for p in (self.points[i], self.points[(i + 1) % len(self.points)])]
        x0, x1 = min(p[0] for p in walls), max(p[0] for p in walls)
        z1 = max(p[1] for p in walls)
        if z1 < 1e-9:  # a single wall along z = 0: show the house a bit wider than the deck
            x0, x1 = x0 - 2 * FT, x1 + 2 * FT
        return [(x0, -behind), (x1, -behind), (x1, z1), (x0, z1)]

    def is_diagonal(self, i: int) -> bool:
        a, b = self.points[i], self.points[(i + 1) % len(self.points)]
        return abs(a[0] - b[0]) > 1e-9 and abs(a[1] - b[1]) > 1e-9

    def stair_problem(self, i: int) -> str | None:
        """Why edge i can't take stairs, or None if it can."""
        if self.edges[i].kind != "open":
            return "stairs must go on an open side"
        if self.is_diagonal(i):
            return "stairs can't go on an angled side"
        if self.edge_length(i) < STAIR_MIN_EDGE - 1e-6:
            return "side is too short for stairs"
        return None

    def edge_index(self, label: str) -> int:
        for i, e in enumerate(self.edges):
            if e.label == label:
                return i
        raise ValueError(f"no edge labeled {label}")


def make_layout(shape: str, dims: dict[str, int]) -> Layout:
    """Footprint for a shape and its dimensions (feet). Raises ValueError if out of range."""
    if shape not in SHAPE_DIMS:
        raise ValueError(f"unknown shape {shape}")
    spec = SHAPE_DIMS[shape]
    if set(dims) != set(spec):
        raise ValueError(f"{shape} needs dimensions {sorted(spec)}")
    for key, (lo, hi) in spec.items():
        if not lo <= dims[key] <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi} ft")

    H = Edge("house")
    if shape == "rect":
        d, w = dims["A"], dims["B"]
        pts = [(0, 0), (0, d), (w, d), (w, 0)]
        edges = [Edge("open", "A"), Edge("open", "B"), Edge("open", "C"), H]
    elif shape == "l":
        # House sits in the top-left notch; the deck wraps its outside corner
        w, d = dims["B"], dims["C"]
        house_w, house_d = w - dims["D"], d - dims["A"]
        if house_w < MIN_HOUSE_FT or house_d < MIN_HOUSE_FT:
            raise ValueError(f"A must be at most C - {MIN_HOUSE_FT} and D at most B - {MIN_HOUSE_FT}")
        pts = [(0, house_d), (0, d), (w, d), (w, 0), (house_w, 0), (house_w, house_d)]
        edges = [Edge("open", "A"), Edge("open", "B"), Edge("open", "C"), Edge("open", "D"), H, H]
    elif shape == "l_left":
        # Mirror image: house in the top-right notch, the deck wraps around to its left
        w, d = dims["B"], dims["A"]
        house_x, house_d = dims["D"], d - dims["C"]
        if w - house_x < MIN_HOUSE_FT or house_d < MIN_HOUSE_FT:
            raise ValueError(f"C must be at most A - {MIN_HOUSE_FT} and D at most B - {MIN_HOUSE_FT}")
        pts = [(0, 0), (0, d), (w, d), (w, house_d), (house_x, house_d), (house_x, 0)]
        edges = [Edge("open", "A"), Edge("open", "B"), Edge("open", "C"), H, H, Edge("open", "D")]
    elif shape == "u":
        # House bump sits in the top-center notch; the deck wraps three of its walls
        w, d, wall = dims["B"], dims["A"], dims["wall"]
        left, right = dims["E"], w - dims["D"]
        if right - left < MIN_HOUSE_FT + 2:
            raise ValueError("B must leave at least 6 ft of house between D and E")
        if wall > d - MIN_HOUSE_FT:
            raise ValueError(f"house wall depth must be at most A - {MIN_HOUSE_FT}")
        pts = [(0, 0), (0, d), (w, d), (w, 0), (right, 0), (right, wall), (left, wall), (left, 0)]
        edges = [Edge("open", "A"), Edge("open", "B"), Edge("open", "C"), Edge("open", "D"), H, H, H, Edge("open", "E")]
    else:  # angled: straight sides, 45° corners, then the front
        side, c, front = dims["A"], dims["B"] / math.sqrt(2), dims["C"]
        w = front + 2 * c
        if w > MAX_WIDTH_FT:
            raise ValueError(f"total width must be at most {MAX_WIDTH_FT} ft")
        pts = [(0, 0), (0, side), (c, side + c), (w - c, side + c), (w, side), (w, 0)]
        edges = [Edge("open", "A"), Edge("open", "B"), Edge("open", "C"), Edge("open", "D"), Edge("open", "E"), H]
    return Layout([(x * FT, z * FT) for x, z in pts], edges)


# ---------- solids ----------


@dataclass
class Solid:
    face_list: list[list[Vec3]]  # each face counter-clockwise seen from outside
    material: str

    def faces(self) -> list[list[Vec3]]:
        return self.face_list

    def mapped(self, fn) -> Solid:
        return Solid([[fn(p) for p in f] for f in self.face_list], self.material)


def _area(poly: list[Vec2]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1])) / 2


def extrude(poly: list[Vec2], y0: float, y1: float) -> list[list[Vec3]]:
    """Faces of a convex plan polygon extruded from y0 up to y1."""
    if _area(poly) > 0:  # top face must wind so its normal points up
        poly = poly[::-1]
    top = [(x, y1, z) for x, z in poly]
    faces = [top, [(x, y0, z) for x, z in poly][::-1]]
    for (xa, za), (xb, zb) in zip(poly, poly[1:] + poly[:1]):
        faces.append([(xa, y0, za), (xb, y0, zb), (xb, y1, zb), (xa, y1, za)])
    return faces


def profile_x(profile: list[tuple[float, float]], x0: float, x1: float) -> list[list[Vec3]]:
    """Faces of a convex (z, y) profile extruded along X from x0 to x1."""
    if _area(profile) < 0:
        profile = profile[::-1]
    left = [(x0, y, z) for z, y in profile]
    right = [(x1, y, z) for z, y in profile]
    faces = [left, right[::-1]]
    n = len(profile)
    for i in range(n):
        j = (i + 1) % n
        faces.append([left[j], left[i], right[i], right[j]])
    return faces


def rect_plan(cx: float, cz: float, u: Vec2, half_len: float, half_wid: float) -> list[Vec2]:
    """Plan rectangle centered at (cx, cz), long axis along unit vector u."""
    n = (-u[1], u[0])
    return [
        (cx + u[0] * sl + n[0] * sw, cz + u[1] * sl + n[1] * sw)
        for sl, sw in ((-half_len, -half_wid), (half_len, -half_wid), (half_len, half_wid), (-half_len, half_wid))
    ]


@dataclass
class DeckModel:
    solids: list[Solid] = field(default_factory=list)

    def add(self, faces, material):
        self.solids.append(Solid(faces, material))

    def box(self, x0, y0, z0, x1, y1, z1, material):
        self.add(extrude([(x0, z0), (x1, z0), (x1, z1), (x0, z1)], min(y0, y1), max(y0, y1)), material)

    def bounds(self) -> tuple[Vec3, Vec3]:
        pts = [p for s in self.solids for f in s.faces() for p in f]
        return tuple(min(p[i] for p in pts) for i in range(3)), tuple(max(p[i] for p in pts) for i in range(3))


# ---------- polygon helpers ----------


def _inward_normals(pts: list[Vec2]) -> list[Vec2]:
    sign = 1 if _area(pts) > 0 else -1
    out = []
    for a, b in zip(pts, pts[1:] + pts[:1]):
        length = math.dist(a, b)
        dx, dz = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        out.append((-dz * sign, dx * sign))
    return out


def _offset(pts: list[Vec2], d: float) -> list[Vec2]:
    """Move every edge inward by d and re-intersect neighbours (works for reflex corners too)."""
    normals = _inward_normals(pts)
    n = len(pts)
    out = []
    for i in range(n):
        p0 = pts[i - 1]
        np_, nn = normals[i - 1], normals[i]
        # Previous edge line: point on it + direction; current edge likewise
        a = (p0[0] + np_[0] * d, p0[1] + np_[1] * d)
        ua = (pts[i][0] - p0[0], pts[i][1] - p0[1])
        b = (pts[i][0] + nn[0] * d, pts[i][1] + nn[1] * d)
        ub = (pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1])
        cross = ua[0] * ub[1] - ua[1] * ub[0]
        if abs(cross) < 1e-12:
            out.append(b)
            continue
        t = ((b[0] - a[0]) * ub[1] - (b[1] - a[1]) * ub[0]) / cross
        out.append((a[0] + ua[0] * t, a[1] + ua[1] * t))
    return out


def _intervals(pts: list[Vec2], z: float) -> list[tuple[float, int, float, int]]:
    """Inside spans of the polygon along the line z = const: (x_left, edge_left, x_right, edge_right)."""
    hits = []
    for i, (a, b) in enumerate(zip(pts, pts[1:] + pts[:1])):
        lo, hi = min(a[1], b[1]), max(a[1], b[1])
        if lo <= z < hi:
            hits.append((_x_on_edge(pts, i, z), i))
    hits.sort()
    return [(hits[k][0], hits[k][1], hits[k + 1][0], hits[k + 1][1]) for k in range(0, len(hits) - 1, 2)]


def _x_on_edge(pts: list[Vec2], i: int, z: float) -> float:
    a, b = pts[i], pts[(i + 1) % len(pts)]
    return a[0] + (z - a[1]) * (b[0] - a[0]) / (b[1] - a[1])


def _even_positions(a: float, b: float, max_spacing: float) -> list[float]:
    """Positions from a to b inclusive with spacing no larger than max_spacing."""
    n = max(1, math.ceil((b - a) / max_spacing - 1e-9))
    return [a + (b - a) * i / n for i in range(n + 1)]


# ---------- builder ----------


def layout_center(layout: Layout) -> Vec2:
    """Center of the footprint's bounding box; the built model is shifted so this is the origin."""
    xs = [p[0] for p in layout.points]
    zs = [p[1] for p in layout.points]
    return (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2


def build_deck(layout: Layout, height: float, stair_edge: int | None) -> DeckModel:
    model = DeckModel()
    top = height
    joist_top = top - DECK_BOARD_T
    joist_bottom = joist_top - JOIST_H

    _build_framing(model, layout, joist_top, joist_bottom)
    _build_decking(model, layout, joist_top, top)
    _build_railing(model, layout, stair_edge, top)
    if stair_edge is not None:
        _build_stairs(model, layout, stair_edge, top)

    # Center the deck footprint on the origin
    cx, cz = layout_center(layout)
    model.solids = [s.mapped(lambda p: (p[0] - cx, p[1], p[2] - cz)) for s in model.solids]
    return model


def _build_framing(model, layout, joist_top, joist_bottom):
    pts = layout.points
    n = len(pts)
    inner = _offset(pts, JOIST_T)
    beam_bottom = joist_bottom - BEAM_H

    # Rim joists (ledgers on house edges), mitered at every corner
    for i in range(n):
        j = (i + 1) % n
        model.add(extrude([pts[i], pts[j], inner[j], inner[i]], joist_bottom, joist_top), "framing")

    # Field joists, 16" on center, running away from the house (along z); ends cut to the rim angle
    swapped = [(z, x) for x, z in inner]
    min_x, max_x = min(p[0] for p in inner), max(p[0] for p in inner)
    corner_xs = [p[0] for p in inner]
    x = min_x + JOIST_SPACING
    while x < max_x - JOIST_T:
        x0, x1 = x - JOIST_T / 2, x + JOIST_T / 2
        if any(x0 - 0.1 * IN < cx < x1 + 0.1 * IN for cx in corner_xs):
            x += JOIST_SPACING  # straddles a rim corner, where the end cut can't follow two edges
            continue
        for _, el, _, er in _intervals(swapped, x):
            quad = [
                (x0, _x_on_edge(swapped, el, x0)),
                (x1, _x_on_edge(swapped, el, x1)),
                (x1, _x_on_edge(swapped, er, x1)),
                (x0, _x_on_edge(swapped, er, x0)),
            ]
            model.add(extrude(quad, joist_bottom, joist_top), "framing")
        x += JOIST_SPACING

    # Beams parallel to the house. If the far side of the deck (z = min) is an open edge,
    # it needs a beam too; otherwise the ledger carries that end.
    min_z, max_z = min(p[1] for p in pts), max(p[1] for p in pts)
    open_back = any(
        e.kind == "open" and abs(pts[i][1] - min_z) < 1e-6 and abs(pts[(i + 1) % n][1] - min_z) < 1e-6
        for i, e in enumerate(layout.edges)
    )
    z_first = min_z + BEAM_SETBACK if open_back else min_z
    z_last = max_z - BEAM_SETBACK
    count = max(1, math.ceil((z_last - z_first) / MAX_BEAM_SPACING - 1e-9))
    ks = range(0, count + 1) if open_back else range(1, count + 1)
    beam_zs = [z_first + (z_last - z_first) * k / count for k in ks]

    posts: list[tuple[float, float, float]] = []  # x, z, top of post
    for bz in beam_zs:
        for x0, _, x1, _ in _intervals(inner, bz):
            x0, x1 = x0 + BEAM_T, x1 - BEAM_T  # keep square beam ends inside mitered rims
            if x1 - x0 < POST_SIZE:
                continue
            model.box(x0, beam_bottom, bz - BEAM_T / 2, x1, joist_bottom, bz + BEAM_T / 2, "framing")
            posts += [(px, bz, beam_bottom) for px in _even_positions(x0 + POST_SIZE / 2, x1 - POST_SIZE / 2, MAX_POST_SPACING)]

    # Extra posts under outside corners of the rim that no beam post is near
    corner = _offset(pts, POST_SIZE / 2)
    for i in range(n):
        prev_edge, next_edge = layout.edges[i - 1], layout.edges[i]
        if prev_edge.kind != "open" or next_edge.kind != "open":
            continue
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        turn = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if turn * _area(pts) <= 0:  # reflex corner
            continue
        if all(math.dist(corner[i], p[:2]) > 2.5 * FT for p in posts):
            posts.append((*corner[i], joist_bottom))  # directly under the rim

    h, f = POST_SIZE / 2, FOOTING_SIZE / 2
    for px, pz, post_top in posts:
        model.box(px - h, FOOTING_H, pz - h, px + h, max(post_top, FOOTING_H), pz + h, "framing")
        model.box(px - f, 0.0, pz - f, px + f, FOOTING_H, pz + f, "concrete")


def _build_decking(model, layout, joist_top, top):
    """Boards run parallel to the house; rows restart at every vertex height so cuts follow the edges."""
    pts = layout.points
    rng = random.Random(42)
    breaks = sorted({p[1] for p in pts})
    for za, zb in zip(breaks, breaks[1:]):
        z = za
        while z < zb - 1e-6:
            z1 = min(z + DECK_BOARD_W, zb)
            for _, el, _, er in _intervals(pts, (z + z1) / 2):
                quad = [
                    (_x_on_edge(pts, el, z), z),
                    (_x_on_edge(pts, er, z), z),
                    (_x_on_edge(pts, er, z1), z1),
                    (_x_on_edge(pts, el, z1), z1),
                ]
                model.add(extrude(quad, joist_top, top), rng.choice(DECKING_MATERIALS))
            z = z1 + DECK_BOARD_GAP


def _stair_gap(layout: Layout, i: int, center: list[Vec2]) -> tuple[float, float]:
    """Centerline positions (along edge i) of the two railing posts flanking the stairs."""
    pts = layout.points
    a, b = pts[i], pts[(i + 1) % len(pts)]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    c0, c1 = center[i], center[(i + 1) % len(pts)]
    length = math.dist(c0, c1)
    u = ((c1[0] - c0[0]) / length, (c1[1] - c0[1]) / length)
    s_mid = (mid[0] - c0[0]) * u[0] + (mid[1] - c0[1]) * u[1]
    half = STAIR_WIDTH / 2 - RAIL_POST_SIZE / 2
    return s_mid - half, s_mid + half


def _build_railing(model, layout, stair_edge, top):
    pts = layout.points
    n = len(pts)
    edges = layout.edges
    normals = _inward_normals(pts)
    center = _offset(pts, RAIL_INSET)
    outer = _offset(pts, RAIL_INSET - TOP_RAIL_W / 2)
    inner = _offset(pts, RAIL_INSET + TOP_RAIL_W / 2)

    rail_top = top + RAIL_HEIGHT
    post_top = rail_top + TOP_RAIL_T / 2
    bottom_rail_y0 = top + BOTTOM_RAIL_GAP
    bottom_rail_y1 = bottom_rail_y0 + BOTTOM_RAIL_T
    placed_posts: list[Vec2] = []

    def post(p: Vec2, u: Vec2):
        if any(math.dist(p, q) < 0.5 * IN for q in placed_posts):
            return  # shared corner post already placed by the neighbouring edge
        placed_posts.append(p)
        h = RAIL_POST_SIZE / 2
        model.add(extrude(rect_plan(p[0], p[1], u, h, h), top, post_top), "railing")

    for i, e in enumerate(edges):
        if e.kind != "open":
            continue
        j = (i + 1) % n
        c0, c1 = center[i], center[j]
        length = math.dist(c0, c1)
        u = ((c1[0] - c0[0]) / length, (c1[1] - c0[1]) / length)
        nin = normals[i]

        def at(s, side=0.0):
            """Point s along the centerline, shifted `side` toward the deck interior."""
            return (c0[0] + u[0] * s + nin[0] * side, c0[1] + u[1] * s + nin[1] * side)

        # Sections: (start s, end s, starts at a mitered corner?, ends at a mitered corner?, stair post s)
        miter_start = edges[i - 1].kind == "open"
        miter_end = edges[j].kind == "open"
        if i == stair_edge:
            g0, g1 = _stair_gap(layout, i, center)
            sections = [(0.0, g0, miter_start, False, g0), (g1, length, False, miter_end, g1)]
        else:
            sections = [(0.0, length, miter_start, miter_end, None)]

        for s0, s1, m0, m1, stair_post in sections:
            if s1 - s0 < RAIL_POST_SIZE * 1.5:
                # Too short for a railing: just the post the stair handrail needs
                post(at(stair_post if stair_post is not None else s0), u)
                continue
            stations = _even_positions(s0, s1, MAX_RAIL_POST_SPACING)
            for s in stations:
                post(at(s), u)

            # Top rail: mitered into the neighbouring railing; elsewhere square-cut just past the end
            # post so the cut face never lines up with the post face
            w = TOP_RAIL_W / 2
            past = RAIL_POST_SIZE / 2 + 2 * EMBED
            start = (outer[i], inner[i]) if m0 else (at(s0 - past, -w), at(s0 - past, w))
            end = (outer[j], inner[j]) if m1 else (at(s1 + past, -w), at(s1 + past, w))
            model.add(extrude([start[0], end[0], end[1], start[1]], rail_top, rail_top + TOP_RAIL_T), "railing")

            for k0, k1 in zip(stations, stations[1:]):
                clear0, clear1 = k0 + RAIL_POST_SIZE / 2, k1 - RAIL_POST_SIZE / 2
                mid = at((clear0 + clear1) / 2)
                model.add(
                    extrude(rect_plan(mid[0], mid[1], u, (clear1 - clear0) / 2, BOTTOM_RAIL_W / 2), bottom_rail_y0, bottom_rail_y1),
                    "railing",
                )
                count = max(0, math.floor((clear1 - clear0) / BALUSTER_SPACING))
                step = (clear1 - clear0) / (count + 1)
                b = BALUSTER_SIZE / 2
                for k in range(1, count + 1):
                    p = at(clear0 + step * k)
                    model.add(extrude(rect_plan(p[0], p[1], u, b, b), bottom_rail_y1 - EMBED, rail_top + EMBED), "railing")


def _build_stairs(model, layout, i, top):
    """Stairs centered on edge i, descending away from the deck. Built in a local frame, then placed."""
    pts = layout.points
    a, b = pts[i], pts[(i + 1) % len(pts)]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    length = math.dist(a, b)
    u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
    nin = _inward_normals(pts)[i]
    nout = (-nin[0], -nin[1])
    if u[0] * nout[1] - u[1] * nout[0] < 0:  # keep the local->world map a rotation, not a mirror
        u = (-u[0], -u[1])

    local = DeckModel()
    _build_stairs_local(local, top)
    for s in local.solids:
        model.solids.append(
            s.mapped(lambda p: (mid[0] + p[0] * u[0] + p[2] * nout[0], p[1], mid[1] + p[0] * u[1] + p[2] * nout[1]))
        )


def stair_risers(top: float) -> int:
    return max(2, math.ceil(top / STAIR_MAX_RISE - 1e-9))


def _build_stairs_local(model, top):
    """Local frame: x along the deck edge (stairs centered on x = 0), z out from the deck edge at z = 0."""
    n_risers = stair_risers(top)
    rise = top / n_risers
    slope = rise / STAIR_RUN

    def nosing(z):
        """Height of the line through the tread nosings."""
        return top - z * slope

    def z_at(line, y):
        """Where a line parallel to the nosing line reaches height y."""
        return (line(0.0) - y) / slope

    rng = random.Random(7)
    xl, xr = -STAIR_WIDTH / 2 + RAIL_POST_SIZE / 2, STAIR_WIDTH / 2 - RAIL_POST_SIZE / 2  # stringers & stair rails
    tread_x0, tread_x1 = xl + STRINGER_T / 2, xr - STRINGER_T / 2

    for k in range(1, n_risers + 1):
        tread_top = top - k * rise
        z_back = (k - 1) * STAIR_RUN
        # Riser above tread k (the bottom riser stands on the ground)
        model.box(tread_x0, tread_top, z_back, tread_x1, tread_top + rise - STAIR_TREAD_T, z_back + STAIR_RISER_T, "railing")
        if k < n_risers:
            model.box(
                tread_x0, tread_top - STAIR_TREAD_T, z_back,
                tread_x1, tread_top, z_back + STAIR_RUN + STAIR_RISER_T + STAIR_NOSING,
                rng.choice(DECKING_MATERIALS),
            )

    # Closed stringers: a sloped band from the rim joist down to the ground
    def stringer_top(z):
        return nosing(z) + STRINGER_ABOVE_NOSING

    def stringer_bottom(z):
        return stringer_top(z) - STRINGER_DEPTH

    stringer = [
        (0.0, stringer_top(0.0)),
        (0.0, max(stringer_bottom(0.0), 0.0)),
        (z_at(stringer_bottom, 0.0), 0.0),
        (z_at(stringer_top, 0.0), 0.0),
    ]
    for x in (xl, xr):
        model.add(profile_x(stringer, x - STRINGER_T / 2, x + STRINGER_T / 2), "framing")

    # Handrails: from just outside the deck's top rail down to a post at the bottom step.
    # The deck railing posts flanking the opening sit at z = -RAIL_INSET.
    h = RAIL_POST_SIZE / 2
    rail_start = -RAIL_INSET + TOP_RAIL_W / 2  # outer face of the deck's top rail
    post_face_top = -RAIL_INSET + h
    # Bottom posts sit on the last riser line, nudged so their faces miss the bottom tread's nosing
    z_bottom = (n_risers - 1) * STAIR_RUN + 2 * EMBED
    post_face_bottom = z_bottom - h

    def band(offset, thickness, z0, z1):
        """Sloped band following the nosing line, `offset` above it."""
        y0, y1 = nosing(z0) + offset, nosing(z1) + offset
        return [(z0, y0), (z1, y1), (z1, y1 + thickness), (z0, y0 + thickness)]

    for x in (xl, xr):
        model.box(x - h, 0.0, z_bottom - h, x + h, nosing(post_face_bottom) + RAIL_HEIGHT + TOP_RAIL_T * 1.5, z_bottom + h, "railing")
        model.add(profile_x(band(RAIL_HEIGHT, TOP_RAIL_T, rail_start, post_face_bottom), x - TOP_RAIL_W / 2, x + TOP_RAIL_W / 2), "railing")
        model.add(
            profile_x(band(BOTTOM_RAIL_GAP, BOTTOM_RAIL_T, post_face_top, post_face_bottom), x - BOTTOM_RAIL_W / 2, x + BOTTOM_RAIL_W / 2),
            "railing",
        )

        count = max(0, math.floor((post_face_bottom - post_face_top) / BALUSTER_SPACING))
        step = (post_face_bottom - post_face_top) / (count + 1)
        bh = BALUSTER_SIZE / 2
        for k in range(1, count + 1):
            z = post_face_top + step * k
            y0 = nosing(z) + BOTTOM_RAIL_GAP + BOTTOM_RAIL_T - EMBED
            y1 = nosing(z) + RAIL_HEIGHT + EMBED
            model.box(x - bh, y0, z - bh, x + bh, y1, z + bh, "railing")

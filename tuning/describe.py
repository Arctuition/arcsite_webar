"""The deck in a few plain lines, as seen from the yard facing the house.

Input is the canvas's deck JSON (feet, outline counter-clockwise with y up, edge i
from point i to point i + 1; `ledger` / `house` the edges against the house).
"""

from __future__ import annotations

import math

FURNITURE_NAMES = {
    "hot_tub_square": "square hot tub",
    "hot_tub_rect": "rectangular hot tub",
    "hot_tub_round": "round hot tub",
    "lounge_chair": "lounge chair",
    "two_seater_sofa": "two-seater sofa",
    "three_seater_sofa": "three-seater sofa",
    "bistro_set": "small round bistro table with two chairs",
    "round_dining_set": "round dining table with four chairs",
    "dining_set": "rectangular dining table with chairs",
    "fire_pit": "round fire pit",
    "grill": "barbecue grill",
    "patio_umbrella": "patio umbrella",
}


def _edge_no(ref) -> int:
    return ref[0] if isinstance(ref, list) else ref


class Frame:
    """x to the viewer's right along the main house wall, z out from the house toward the yard."""

    def __init__(self, deck: dict):
        pts = deck["outline"]
        house = [_edge_no(e) for e in deck.get("ledger", []) + deck.get("house", [])]
        if not house:
            raise ValueError("the deck has no edge against the house")

        def length(i):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            return math.dist(a, b)

        main = max(house, key=length)
        a, b = pts[main], pts[(main + 1) % len(pts)]
        dx, dy = (b[0] - a[0]) / length(main), (b[1] - a[1]) / length(main)
        self.out = (-dy, dx)  # the deck's inside is left of a counter-clockwise edge: away from the house
        facing = (dy, -dx)  # toward the house
        self.right = (facing[1], -facing[0])
        xs = [self._x(p) for p in pts]
        self.x0 = min(xs)
        self.z0 = self._z(a)
        self.points = [self.map(p) for p in pts]

    def _x(self, p):
        return p[0] * self.right[0] + p[1] * self.right[1]

    def _z(self, p):
        return p[0] * self.out[0] + p[1] * self.out[1]

    def map(self, p):
        return (self._x(p) - self.x0, self._z(p) - self.z0)

    def direction(self, v):
        return (v[0] * self.right[0] + v[1] * self.right[1], v[0] * self.out[0] + v[1] * self.out[1])


def _ft(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _depth_profile(points: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    """(x from, x to, depth) for each run along the wall, merged where the depth is the same."""
    xs = sorted({round(p[0], 2) for p in points})
    runs = []
    n = len(points)
    for x0, x1 in zip(xs, xs[1:]):
        mid = (x0 + x1) / 2
        zs = []
        for i in range(n):
            (ax, az), (bx, bz) = points[i], points[(i + 1) % n]
            if (ax - mid) * (bx - mid) < 0:
                zs.append(az + (bz - az) * (mid - ax) / (bx - ax))
        if len(zs) >= 2:
            depth = max(zs) - min(zs)
            if runs and abs(runs[-1][2] - depth) < 0.05:
                runs[-1] = (runs[-1][0], x1, depth)
            else:
                runs.append((x0, x1, depth))
    return runs


def _where(x: float, z: float, width: float, depth: float) -> str:
    across = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "middle"
    out = "back" if z < depth / 3 else "front" if z > 2 * depth / 3 else "middle"
    if across == out == "middle":
        return "in the middle"
    if out == "middle":
        return f"on the {across} side"
    if across == "middle":
        return f"at the {out}, centred"
    return f"at the {out} {across}"


def describe(deck: dict, scale: bool = False) -> tuple[str, str]:
    """(deck lines, furniture lines)."""
    f = Frame(deck)
    pts = f.points
    width = max(p[0] for p in pts) - min(p[0] for p in pts)
    depth = max(p[1] for p in pts) - min(p[1] for p in pts)
    height = deck["height_in"] / 12
    lines = [
        f"- {_ft(width)} ft along the house wall and {_ft(depth)} ft out from it at its deepest; "
        f"the floor is {_ft(height)} ft above the ground.",
    ]
    if scale:
        # Scale against what every photo has: a door is about 3 ft wide and 7 ft tall.
        lines.append(
            f"- Scale it by the house's doors (about 3 ft wide, 7 ft tall): the deck is about {width / 3:.1f} door-widths "
            f"wide, sticks out about {depth / 7:.1f} door-heights at its deepest, and its floor is about "
            f"{height / 7:.1f} of a door's height above the ground. Do not make it wider or shallower than this.")
    runs = _depth_profile(pts)

    def run_name(i: int) -> str:
        return "left" if i == 0 else "right" if i == len(runs) - 1 else f"part {i + 1} from the left"

    if len(runs) > 1:
        parts = [f"the {run_name(i)} part is {_ft(x1 - x0)} ft wide and sticks out {_ft(d)} ft"
                 for i, (x0, x1, d) in enumerate(runs)]
        deepest = max(d for _, _, d in runs)
        steps = [f"the {run_name(i)} part is only about {round(100 * d / deepest)}% as deep as the deepest part"
                 for i, (_, _, d) in enumerate(runs) if d < deepest * 0.95]
        lines.append(f"- NOT a rectangle. Seen from the yard, {'; '.join(parts)}. So {', and '.join(steps)}: "
                     "this step in the outline must be clearly visible in the result, as in the renders.")
    else:
        lines.append("- A plain rectangle.")

    n = len(deck["outline"])
    for s in deck.get("stairs", []):
        a, b = deck["outline"][s["edge"]], deck["outline"][(s["edge"] + 1) % n]
        length = math.dist(a, b)
        t = (s["from"] + s["width"] / 2) / length
        mid = f.map((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        normal = f.direction(((b[1] - a[1]) / length, -(b[0] - a[0]) / length))  # outside of the deck
        toward = ("toward the yard" if normal[1] > 0.7 else "toward the house" if normal[1] < -0.7
                  else "to the right" if normal[0] > 0 else "to the left")
        where = ""
        if len(runs) > 1 and toward == "toward the yard":
            for i, (x0, x1, _) in enumerate(runs):
                if x0 <= mid[0] <= x1:
                    where = (f" from the front edge of the {run_name(i)} part, near its middle"
                             " (not from the inner corner where the parts meet)")
        lines.append(f"- One {_ft(s['width'])} ft wide stair with {s['risers']} steps goes down {toward}{where}; "
                     f"its middle is {_ft(mid[0])} ft from the deck's left end.")

    railed = {e for group in deck.get("railing", []) for e in group["edges"]}
    house = {_edge_no(e) for e in deck.get("ledger", []) + deck.get("house", [])}
    open_edges = set(range(n)) - house
    if railed and railed >= open_edges:
        lines.append("- Railing along every open edge, with a gap only where the stairs go down; none against the house.")
    elif railed:
        lines.append(f"- Railing on {len(railed & open_edges)} of the {len(open_edges)} open edges (see the renders); none against the house.")
    else:
        lines.append("- No railing.")

    pieces = deck.get("furniture", [])
    if not pieces:
        return "\n".join(lines), "There is no furniture on the deck; add none."
    counts: dict[str, int] = {}
    rows = []
    for p in pieces:
        name = FURNITURE_NAMES.get(p["kind"], p["kind"].replace("_", " "))
        counts[name] = counts.get(name, 0) + 1
        x, z = f.map(p["center"])
        near = " against the house" if z - min(p["width"], p["depth"]) / 2 < 1.5 else ""
        rows.append(f"- a {name}, about {_ft(p['width'])} x {_ft(p['depth'])} ft, {_where(x, z, width, depth)}{near}")
    total = ", ".join(f"{c} {n if c == 1 else n + 's'}" for n, c in counts.items())
    head = f"Exactly {len(pieces)} pieces on the deck ({total}), as in the renders; add no other furniture:"
    return "\n".join(lines), "\n".join([head, *rows])


if __name__ == "__main__":
    import json
    import sys

    d, furn = describe(json.load(open(sys.argv[1])))
    print(d)
    print(furn)

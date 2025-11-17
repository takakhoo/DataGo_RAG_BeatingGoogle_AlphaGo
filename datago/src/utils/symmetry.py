"""
utils/symmetry.py

Lightweight symmetry utilities mirroring KataGo's SymmetryHelpers for canonicalization
and move remapping. Supports the same 3-bit encoding: bit0=flipY, bit1=flipX, bit2=transpose.
"""
from __future__ import annotations

from typing import Tuple


def pos_to_xy(pos: int, x_size: int) -> Tuple[int, int]:
    x = pos % x_size
    y = pos // x_size
    return x, y


def xy_to_pos(x: int, y: int, x_size: int) -> int:
    return y * x_size + x


def apply_symmetry_xy(x: int, y: int, x_size: int, y_size: int, symmetry: int) -> Tuple[int, int]:
    """Apply symmetry to coordinates (x,y).

    symmetry bitmask: bit0=flipY, bit1=flipX, bit2=transpose
    Applied in order: flipX, flipY, then transpose (matching KataGo ordering comments).
    """
    transpose = (symmetry & 0x4) != 0
    flip_x = (symmetry & 0x2) != 0
    flip_y = (symmetry & 0x1) != 0
    if flip_x:
        x = x_size - x - 1
    if flip_y:
        y = y_size - y - 1
    if transpose:
        x, y = y, x
    return x, y


def invert_symmetry(symmetry: int) -> int:
    # inverse mapping as in KataGo: some symmetries swap 5 and 6
    if symmetry == 5:
        return 6
    if symmetry == 6:
        return 5
    return symmetry


def compose_symmetry(first: int, next_s: int) -> int:
    # Compose symmetries: if first has transpose, rotate bits of next
    if (first & 0x4) != 0:
        # rotate bits: keep transpose bit, swap flipX/flipY
        next_s = (next_s & 0x4) | ((next_s & 0x2) >> 1) | ((next_s & 0x1) << 1)
    return first ^ next_s


def map_pos_between_symmetries(pos: int, from_sym: int, to_sym: int, board_x: int = 19, board_y: int = 19) -> int:
    """Map a position index from a stored symmetry `from_sym` into target symmetry `to_sym`.

    This computes the composed symmetry s_map = compose(invert(from_sym), to_sym) and applies it to the
    coordinates.
    """
    inv = invert_symmetry(from_sym)
    s_map = compose_symmetry(inv, to_sym)
    x, y = pos_to_xy(pos, board_x)
    nx, ny = apply_symmetry_xy(x, y, board_x, board_y, s_map)
    return xy_to_pos(nx, ny, board_x)


def format_symmetry(sym: int) -> str:
    return f"sym={sym:03b}"


__all__ = [
    "pos_to_xy",
    "xy_to_pos",
    "apply_symmetry_xy",
    "invert_symmetry",
    "compose_symmetry",
    "map_pos_between_symmetries",
    "format_symmetry",
]

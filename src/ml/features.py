"""Milestone 2 -- turning a described side into tensors a network can read.

Both models are fed from the *same* geometry the classical pipeline already
computes (:mod:`src.piece_description`), so the comparison in the report is
between the **matchers**, not between two different front ends.

Two representations are provided, one per model family:

:func:`side_strip`
    A small image, ``(4, M, D)``: three colour channels sampled at ``D``
    depths just inside the side and ``M`` positions along it, plus a fourth
    channel carrying the shape profile.  This is what the Siamese CNN
    convolves over -- the strip really is a picture of the material either
    side of the cut, so a convolutional encoder is the natural choice.

:func:`side_vector`
    A flat descriptor of the same side: pooled colour, the shape profile at
    reduced resolution, the side type one-hot, and the normalised length.
    This is what the graph network puts on its nodes; it deliberately does
    *not* use a convolutional encoder, because the GNN's job is to exploit
    the relations between sides rather than the texture within one.

The reversal convention of the classical matcher is kept: when two pieces sit
side by side, one side is traversed in the opposite direction to the other,
so ``reverse=True`` flips a side along its length before it is compared.
"""

from __future__ import annotations

import numpy as np

from ..piece_description import SIDE_BLANK, SIDE_FLAT, SIDE_TAB

__all__ = [
    "STRIP_SAMPLES",
    "STRIP_DEPTHS",
    "VECTOR_DIM",
    "side_strip",
    "side_vector",
    "piece_side_strips",
    "side_type_index",
    "relative_orientation",
]

#: Samples along a side, and depths into the piece, used by :func:`side_strip`.
STRIP_SAMPLES = 64
STRIP_DEPTHS = 8

_TYPES = (SIDE_FLAT, SIDE_TAB, SIDE_BLANK)


def side_type_index(side_type: str) -> int:
    """Index of a side type in the one-hot encoding (flat, tab, blank)."""
    return _TYPES.index(side_type)


def _resample_rows(a: np.ndarray, n: int) -> np.ndarray:
    """Resample the first axis of ``a`` to ``n`` rows by linear interpolation."""
    m = a.shape[0]
    if m == n:
        return a
    src = np.linspace(0.0, m - 1.0, n)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, m - 1)
    w = (src - lo).reshape((-1,) + (1,) * (a.ndim - 1))
    return a[lo] * (1.0 - w) + a[hi] * w


def side_strip(side, samples: int = STRIP_SAMPLES, depths: int = STRIP_DEPTHS,
               reverse: bool = False) -> np.ndarray:
    """``(4, samples, depths)`` float32 tensor describing one side.

    Channels 0-2 are the RGB strip sampled inside the piece; channel 3 is the
    shape profile (deviation from the corner-to-corner chord, normalised by
    the chord length) broadcast across depth, so a convolution sees colour
    and geometry in register.
    """
    colours = np.asarray(side.colors, dtype=np.float32)      # (M, D, 3)
    profile = np.asarray(side.profile, dtype=np.float32)     # (M,)

    if colours.shape[1] != depths:
        # resample along depth as well, so every side has the same shape
        d = colours.shape[1]
        src = np.linspace(0.0, d - 1.0, depths)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, d - 1)
        w = (src - lo).reshape(1, -1, 1)
        colours = colours[:, lo] * (1.0 - w) + colours[:, hi] * w

    colours = _resample_rows(colours, samples)               # (S, D, 3)
    profile = _resample_rows(profile.reshape(-1, 1), samples)[:, 0]

    if reverse:
        colours = colours[::-1]
        profile = profile[::-1]

    out = np.empty((4, samples, depths), dtype=np.float32)
    out[:3] = np.transpose(colours, (2, 0, 1))
    out[3] = profile[:, None]
    return out


def side_vector(side, profile_bins: int = 16, colour_bins: int = 8
                ) -> np.ndarray:
    """Flat descriptor of one side, for the graph network's nodes.

    Layout: mean RGB per depth-pooled bin along the side (``colour_bins * 3``),
    the shape profile at ``profile_bins`` resolution, the three-way side-type
    one-hot, and the chord length scaled into a sensible range.
    """
    colours = np.asarray(side.colors, dtype=np.float32).mean(axis=1)   # (M, 3)
    profile = np.asarray(side.profile, dtype=np.float32)

    col = _resample_rows(colours, colour_bins).reshape(-1)
    prof = _resample_rows(profile.reshape(-1, 1), profile_bins)[:, 0]

    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[side_type_index(side.type)] = 1.0

    extra = np.array([float(side.length) / 100.0,
                      float(side.amplitude)], dtype=np.float32)
    return np.concatenate([col, prof, one_hot, extra]).astype(np.float32)


#: Dimension of :func:`side_vector` with its default arguments.
VECTOR_DIM = 8 * 3 + 16 + 3 + 2


def piece_side_strips(description, **kwargs) -> np.ndarray:
    """``(4, 4, S, D)`` -- the four sides of one piece, in clockwise order."""
    return np.stack([side_strip(s, **kwargs) for s in description.sides])


def relative_orientation(side_a: int, side_b: int) -> int:
    """Quarter-turns of B relative to A when side ``a`` meets side ``b``.

    Sides are stored clockwise and the grid directions N, E, S, W are also
    clockwise, so if A's side ``a`` faces direction ``d`` then B's side ``b``
    must face the opposite direction.  Writing each piece's rotation as the
    index of the side facing North, ``a = rot_a + d`` and
    ``b = rot_b + d + 2`` (mod 4).  Eliminating ``d``,

        ``rot_b - rot_a = b - a - 2 = b - a + 2  (mod 4)``

    which is the relative orientation the brief asks each model to report.
    (The two forms agree because ``-2 = +2`` modulo 4.)
    """
    return int((side_b - side_a + 2) % 4)

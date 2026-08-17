"""Milestone 2 / task 4 -- model scores into the Milestone 1 assembler.

The brief requires the compatibility scores of *each* model to be passed to
the **same** assembly algorithm, so that the comparison measures the matchers
and not three different searches.  That is what this module does: it turns a
model into a :class:`src.edge_matching.CompatibilityTable`, which
:func:`src.assembly.assemble` already knows how to consume -- together with
its border constraints, its tie-breaking rule, its dead-end relaxation and
its guarantee to return the best arrangement obtained.

Score to cost
-------------
The networks emit a probability ``p`` that two sides are neighbours; the
assembler minimises a cost.  The conversion is

    ``cost = -log(p + eps)``

which is the negative log-likelihood of the pairing: it is monotonically
decreasing in ``p`` (so the ordering the model intended is preserved), it is
additive across the seams of an arrangement (so the total cost the assembler
minimises is the negative log-likelihood of the whole arrangement, which is
the quantity that actually ought to be minimised), and it grows without bound
as ``p`` approaches zero, which keeps hopeless pairings out of the search.

Admissibility -- flats never form an interior seam, and a tab must meet a
blank -- is applied on top of the model's opinion, exactly as for the
classical table, so all three methods search the same space.

Predicted matches and relative orientation
------------------------------------------
:func:`predicted_matches` reports, for every side, the partner the model
ranks first, its score, and the **relative orientation** the pairing implies
(:func:`src.ml.features.relative_orientation`) -- the four outputs per
candidate pair that the brief asks each model to provide.
"""

from __future__ import annotations

import numpy as np
import torch

from ..edge_matching import CompatibilityTable, MatchWeights
from ..piece_description import SIDE_BLANK, SIDE_TAB
from .features import (VECTOR_DIM, relative_orientation, side_strip,
                       side_vector)
from .gnn import build_graph

__all__ = [
    "EPS",
    "admissibility_mask",
    "probabilities_to_table",
    "siamese_table",
    "gnn_table",
    "predicted_matches",
]

#: Floor on the probability, so ``-log p`` stays finite.
EPS = 1e-6


def admissibility_mask(descriptions) -> np.ndarray:
    """``(S, S)`` boolean: which side pairs may form an interior seam."""
    types = np.array([s.type for d in descriptions for s in d.sides])
    n = len(descriptions)
    piece_of = np.repeat(np.arange(n), 4)
    ok = ((types == SIDE_TAB)[:, None] & (types == SIDE_BLANK)[None, :]) | \
         ((types == SIDE_BLANK)[:, None] & (types == SIDE_TAB)[None, :])
    ok &= piece_of[:, None] != piece_of[None, :]
    return ok


def probabilities_to_table(prob: np.ndarray, descriptions,
                           weights: MatchWeights | None = None
                           ) -> CompatibilityTable:
    """Wrap an ``(S, S)`` probability matrix as a compatibility table."""
    n = len(descriptions)
    s = 4 * n
    prob = np.asarray(prob, dtype=np.float64).reshape(s, s)
    prob = 0.5 * (prob + prob.T)                 # the relation is symmetric
    cost = -np.log(np.clip(prob, EPS, 1.0))

    ok = admissibility_mask(descriptions)
    cost = np.where(ok, cost, np.inf)

    shp = (n, 4, n, 4)
    zeros = np.zeros(shp)
    return CompatibilityTable(cost=cost.reshape(shp),
                              shape=zeros, colour=cost.reshape(shp),
                              length=zeros,
                              weights=weights or MatchWeights(0.0, 1.0, 0.0))


# ==========================================================================
# per-model tables
# ==========================================================================
def siamese_table(model, descriptions, batch: int = 2048) -> CompatibilityTable:
    """Score every admissible side pair of one puzzle with the Siamese CNN."""
    strips = np.stack([side_strip(s) for d in descriptions for s in d.sides])
    with torch.no_grad():
        prob = model.score_all(torch.as_tensor(strips), batch=batch).numpy()
    return probabilities_to_table(prob, descriptions)


def gnn_table(model, descriptions, classical_table=None,
              max_candidates: int = 24) -> CompatibilityTable:
    """Score one puzzle's candidate graph with the graph network.

    Pairs outside the candidate set keep probability 0, i.e. infinite cost;
    the candidate set is the admissible pairs capped per side, ranked by the
    classical cost when one is supplied (the same shortlist the model was
    trained on).
    """
    vecs = np.stack([side_vector(s) for d in descriptions for s in d.sides])
    types = [s.type for d in descriptions for s in d.sides]
    n_sides = vecs.shape[0]

    prior = None
    if classical_table is not None:
        prior = classical_table.cost.reshape(n_sides, n_sides)
        prior = np.where(np.isfinite(prior), prior, 1e6)

    x, edge_index, pairs = build_graph(vecs, types, len(descriptions),
                                       max_candidates=max_candidates,
                                       prior=prior)
    prob = model.score_matrix(x, edge_index, pairs, n_sides)
    return probabilities_to_table(prob, descriptions)


# ==========================================================================
# the four outputs the brief asks for
# ==========================================================================
def predicted_matches(table: CompatibilityTable, descriptions,
                      top_k: int = 1) -> list:
    """For every side, the partner(s) the model ranks best.

    Each entry carries the four things the brief requires of a model:
    whether the two sides are predicted to be neighbours, the numerical
    compatibility score, which sides matched, and the relative orientation
    the pairing implies.
    """
    n = len(descriptions)
    cost = table.cost.reshape(n * 4, n * 4)
    out = []
    for a in range(n * 4):
        row = cost[a]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        order = np.argsort(np.where(finite, row, np.inf))[:top_k]
        for b in order:
            if not np.isfinite(row[b]):
                continue
            c = float(row[b])
            out.append({
                "piece_a": a // 4, "side_a": a % 4,
                "piece_b": int(b) // 4, "side_b": int(b) % 4,
                "score": float(np.exp(-c)),          # back to a probability
                "cost": c,
                "is_neighbour": bool(np.exp(-c) >= 0.5),
                "relative_orientation": relative_orientation(a % 4, int(b) % 4),
            })
    return out

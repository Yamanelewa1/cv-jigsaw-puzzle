"""Milestone 2 / model 2 -- Graph neural network.

How this differs from the Siamese CNN
-------------------------------------
The Siamese network judges a pair **in isolation**: two strips go in, one
probability comes out, and nothing about the rest of the puzzle is visible.
That is exactly where it is weakest, because a jigsaw is a competition -- a
side does not need a partner that looks plausible, it needs the partner that
looks more plausible than every other candidate, and whose own best choice is
this side in return.

The graph network is built on that observation and is a different kind of
model, not a re-tuned one:

* **the unit of inference is the whole puzzle, not the pair.**  Every side of
  every piece is a node; every admissible tab/blank pairing is an edge.  One
  forward pass scores all candidate seams together.
* **the representation is relational, not convolutional.**  Nodes carry a
  flat 45-number descriptor (pooled colour, the shape profile, the side type,
  the length) and there is no convolution anywhere in the model.  What
  refines a node is *message passing*: each side sees what its rival
  candidates look like and updates its own embedding accordingly, so the
  score it eventually gives a partner is a contextual judgement rather than
  an absolute one.
* **two sides of the same piece are linked**, so a side also knows what the
  rest of its own piece looks like -- information the Siamese model never
  receives.

Architecture
    ``SAGEConv`` layers (mean aggregation, with a residual from the node's own
    features) widening 45 -> 96 -> 96 -> 96, ReLU between, then an edge head
    that scores a candidate ``(a, b)`` from
    ``[h_a + h_b, |h_a - h_b|, h_a * h_b, x_a + x_b]`` -- symmetric in the
    pair, like the Siamese head, and given the raw descriptors as well as the
    message-passed ones so early layers cannot wash out the fine detail.
Loss
    Binary cross-entropy over the labelled edges, positive-weighted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                       # torch_geometric is optional
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except Exception:                          # pragma: no cover - fallback path
    SAGEConv = None
    HAS_PYG = False

__all__ = ["PuzzleGNN", "GNNConfig", "build_graph", "HAS_PYG"]


class GNNConfig:
    """Hyper-parameters, kept in one place so the report can quote them."""
    hidden = 96
    layers = 3
    head_hidden = 128
    dropout = 0.15

    def as_dict(self) -> dict:
        return {"hidden": self.hidden, "layers": self.layers,
                "head_hidden": self.head_hidden, "dropout": self.dropout}


class _MeanConv(nn.Module):
    """Mean-aggregation graph convolution, used when PyG is unavailable.

    ``h_i' = W_self x_i + W_neigh * mean_{j in N(i)} x_j`` -- the same update
    rule as GraphSAGE with mean aggregation, written with ``index_add_`` so
    the package has no hard dependency on torch_geometric.
    """

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.lin_self = nn.Linear(cin, cout)
        self.lin_neigh = nn.Linear(cin, cout, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(src, dtype=x.dtype))
        agg = agg / deg.clamp(min=1.0).unsqueeze(1)
        return self.lin_self(x) + self.lin_neigh(agg)


def build_graph(vectors: np.ndarray, side_types, n_pieces: int,
                max_candidates: int = 24, prior: np.ndarray | None = None):
    """Build the candidate graph of one puzzle.

    Nodes are the ``4 * n_pieces`` sides.  Edges are

    * every **admissible** tab/blank pairing between different pieces, capped
      at ``max_candidates`` per side (keeping those the ``prior`` -- normally
      the classical cost -- likes best), which keeps the graph sparse enough
      to message-pass over without discarding plausible partners;
    * the pairings **within** each piece, so a side can see its own piece.

    Returns ``(x, edge_index, candidate_pairs)`` where ``candidate_pairs`` is
    the ``(E, 2)`` array of inter-piece candidates that the head will score.
    """
    x = torch.as_tensor(np.asarray(vectors, dtype=np.float32))
    types = np.asarray(side_types)
    s = len(types)
    piece_of = np.repeat(np.arange(n_pieces), 4)

    ok = ((types == "tab")[:, None] & (types == "blank")[None, :]) | \
         ((types == "blank")[:, None] & (types == "tab")[None, :])
    ok &= piece_of[:, None] != piece_of[None, :]

    pairs = []
    for a in range(s):
        cand = np.flatnonzero(ok[a])
        if cand.size == 0:
            continue
        if prior is not None and cand.size > max_candidates:
            cand = cand[np.argsort(prior[a, cand])[:max_candidates]]
        elif cand.size > max_candidates:
            cand = cand[:max_candidates]
        for b in cand:
            pairs.append((a, int(b)))
    candidate_pairs = (np.array(pairs, dtype=np.int64) if pairs
                       else np.zeros((0, 2), dtype=np.int64))

    edges = list(map(tuple, candidate_pairs.tolist()))
    for p in range(n_pieces):                      # sides of the same piece
        for u in range(4):
            for v in range(4):
                if u != v:
                    edges.append((p * 4 + u, p * 4 + v))
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        e = np.array(edges, dtype=np.int64)
        e = np.vstack([e, e[:, ::-1]])             # make it undirected
        edge_index = torch.as_tensor(e.T)
    return x, edge_index, candidate_pairs


class PuzzleGNN(nn.Module):
    """Message passing over the candidate graph, then an edge score."""

    def __init__(self, in_dim: int, cfg: GNNConfig | None = None,
                 use_pyg: bool | None = None):
        super().__init__()
        self.cfg = cfg or GNNConfig()
        self.use_pyg = HAS_PYG if use_pyg is None else (use_pyg and HAS_PYG)
        h = self.cfg.hidden

        def conv(cin, cout):
            if self.use_pyg:
                return SAGEConv(cin, cout, aggr="mean")
            return _MeanConv(cin, cout)

        dims = [in_dim] + [h] * self.cfg.layers
        self.convs = nn.ModuleList([conv(dims[i], dims[i + 1])
                                    for i in range(self.cfg.layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h)
                                    for _ in range(self.cfg.layers)])
        self.head = nn.Sequential(
            nn.Linear(3 * h + in_dim, self.cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, self.cfg.head_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.head_hidden // 2, 1),
        )

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, edge_index)))
        return h

    def score_edges(self, h: torch.Tensor, x: torch.Tensor,
                    pairs: torch.Tensor) -> torch.Tensor:
        """Logits for the given ``(E, 2)`` candidate pairs."""
        a, b = pairs[:, 0], pairs[:, 1]
        ha, hb = h[a], h[b]
        feat = torch.cat([ha + hb, (ha - hb).abs(), ha * hb, x[a] + x[b]],
                         dim=1)
        return self.head(feat).squeeze(1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                pairs: torch.Tensor) -> torch.Tensor:
        return self.score_edges(self.encode(x, edge_index), x, pairs)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_matrix(self, x: torch.Tensor, edge_index: torch.Tensor,
                     candidate_pairs: np.ndarray, n_sides: int) -> np.ndarray:
        """``(S, S)`` probabilities, ``0`` where a pair was never a candidate."""
        self.eval()
        out = np.zeros((n_sides, n_sides), dtype=np.float32)
        if len(candidate_pairs) == 0:
            return out
        pairs = torch.as_tensor(candidate_pairs)
        p = torch.sigmoid(self.forward(x, edge_index, pairs)).cpu().numpy()
        out[candidate_pairs[:, 0], candidate_pairs[:, 1]] = p
        # the head is symmetric in the pair, so mirror rather than recompute
        both = np.maximum(out, out.T)
        return both

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

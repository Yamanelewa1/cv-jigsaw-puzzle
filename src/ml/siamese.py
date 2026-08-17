"""Milestone 2 / model 1 -- Siamese convolutional network.

Why this architecture
---------------------
A side's descriptor really is a small **picture**: three colour channels
sampled at eight depths just inside the cut and sixty-four positions along
it, with the shape profile in a fourth channel and in register with them.
Deciding whether two such strips continue into one another is a texture and
edge-continuation question, which is what convolutions are for -- and the
kernels should be the *same* for both sides, since "does A continue into B"
must not depend on which side was called A.  That is exactly a Siamese
network: one shared encoder, applied twice, followed by a head that sees
only the pair.

Input representation
    ``(4, 64, 8)`` per side -- RGB at eight depths, plus the shape profile.
    The second side is reversed along its length before encoding, because
    mating sides are traversed in opposite directions.
Encoder
    Four convolutional blocks (3x3, BatchNorm, ReLU) widening 4 -> 32 -> 64
    -> 128, pooling only along the *length* axis so the eight depths stay
    resolved until the end, then global average pooling to a 128-vector.
Head
    The two embeddings are combined symmetrically as
    ``[|a - b|, a * b]`` -- both invariant to swapping the pair, which bakes
    the symmetry of the problem into the model instead of hoping it is
    learned -- and passed through a two-layer MLP to a single logit.
Output
    ``sigmoid(logit)`` is the probability that the two sides are neighbours;
    the compatibility cost handed to the assembler is ``-log p``.

Loss
    Binary cross-entropy with positive weighting, because negatives outnumber
    positives about six to one by construction (see :mod:`src.ml.dataset`).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SiameseCNN", "SiameseConfig"]


class SiameseConfig:
    """Hyper-parameters, kept in one place so the report can quote them."""
    channels = (32, 64, 128)
    embedding = 128
    head_hidden = 128
    dropout = 0.15

    def as_dict(self) -> dict:
        return {"channels": list(self.channels), "embedding": self.embedding,
                "head_hidden": self.head_hidden, "dropout": self.dropout}


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        # pool along the length of the side only; depth is just 8 deep
        nn.MaxPool2d(kernel_size=(2, 1)),
    )


class SiameseCNN(nn.Module):
    """Shared-weight encoder plus a symmetric pair head."""

    def __init__(self, in_channels: int = 4, cfg: SiameseConfig | None = None):
        super().__init__()
        self.cfg = cfg or SiameseConfig()
        c1, c2, c3 = self.cfg.channels
        self.encoder = nn.Sequential(
            _block(in_channels, c1),
            _block(c1, c2),
            _block(c2, c3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, self.cfg.embedding),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * self.cfg.embedding, self.cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.head_hidden, self.cfg.head_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.head_hidden // 2, 1),
        )

    # ------------------------------------------------------------------
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of side strips into embeddings."""
        return self.encoder(x)

    def combine(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        """Symmetric combination: unchanged if the two arguments are swapped."""
        return torch.cat([(ea - eb).abs(), ea * eb], dim=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Logit that sides ``a`` and ``b`` are neighbours."""
        return self.head(self.combine(self.embed(a), self.embed(b))).squeeze(1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_pairs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Probability that each pair is a true seam."""
        self.eval()
        return torch.sigmoid(self.forward(a, b))

    @torch.no_grad()
    def score_all(self, strips: torch.Tensor, batch: int = 4096
                  ) -> torch.Tensor:
        """Full ``(S, S)`` probability matrix for one puzzle's sides.

        ``strips`` is ``(S, 4, L, D)`` in *forward* orientation; the reversal
        that mating sides require is applied here, so entry ``(i, j)`` is the
        probability that side ``i`` meets side ``j``.
        """
        self.eval()
        s = strips.shape[0]
        emb_fwd = self.embed(strips)
        emb_rev = self.embed(torch.flip(strips, dims=[2]))

        out = torch.empty((s, s), dtype=torch.float32)
        idx = torch.arange(s)
        for start in range(0, s, max(1, batch // max(s, 1))):
            rows = idx[start:start + max(1, batch // max(s, 1))]
            if rows.numel() == 0:
                break
            ea = emb_fwd[rows].repeat_interleave(s, dim=0)
            eb = emb_rev.repeat(rows.numel(), 1)
            logit = self.head(self.combine(ea, eb)).squeeze(1)
            out[rows] = torch.sigmoid(logit).view(rows.numel(), s)
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

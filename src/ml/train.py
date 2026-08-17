"""Milestone 2 / task 3 -- training both models under identical conditions.

Both models see the **same puzzles, the same split and the same labels**; the
only thing that differs is the model and the representation it needs.  Every
choice the brief asks the report to state is set here, in one place:

======================  ==========================================================
Loss                    binary cross-entropy on the "are these neighbours?"
                        logit, positive-weighted by the actual class ratio,
                        because negatives outnumber positives ~6:1 by
                        construction
Optimiser               Adam -- the usual choice for small, noisy, sparse
                        problems, and it removes learning-rate schedule
                        tuning from an already large comparison
Learning rate           1e-3, halved when validation AUC stalls for 3 epochs
Batch size              256 pairs (Siamese); one whole puzzle graph (GNN),
                        which is the natural unit there
Epochs                  30 with early stopping on validation AUC, patience 8
Augmentation            illumination, contrast, colour balance and sensor
                        noise on the colour channels only (see
                        :func:`src.ml.dataset.augment_strip`)
Model selection         the epoch with the best **validation** AUC is kept;
                        the test split is not looked at until the end
======================  ==========================================================

Validation uses AUC rather than accuracy because the classes are deliberately
unbalanced and because what the assembler actually needs is a good *ranking*
of candidate partners, which is what AUC measures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .dataset import PairDataset
from .features import VECTOR_DIM, side_vector
from .gnn import PuzzleGNN, build_graph
from .siamese import SiameseCNN

__all__ = [
    "TrainConfig",
    "TrainHistory",
    "roc_auc",
    "train_siamese",
    "train_gnn",
]


@dataclass
class TrainConfig:
    """Every training hyper-parameter the report has to quote."""
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    lr_patience: int = 3
    augment: bool = True
    seed: int = 0

    def as_dict(self) -> dict:
        return {"epochs": self.epochs, "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "early_stopping_patience": self.patience,
                "lr_halved_after": self.lr_patience,
                "augmentation": self.augment, "seed": self.seed,
                "loss": "BCEWithLogits (positive-weighted)",
                "optimiser": "Adam",
                "model_selection": "best validation AUC"}


@dataclass
class TrainHistory:
    """Per-epoch curves, so over/under-fitting can be seen and reported."""
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_auc: list = field(default_factory=list)
    val_auc: list = field(default_factory=list)
    best_epoch: int = -1
    best_val_auc: float = 0.0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"train_loss": self.train_loss, "val_loss": self.val_loss,
                "train_auc": self.train_auc, "val_auc": self.val_auc,
                "best_epoch": self.best_epoch,
                "best_val_auc": self.best_val_auc,
                "training_seconds": round(self.seconds, 1)}

    def diagnosis(self) -> str:
        """A one-line read of the curves: over-fitting, under-fitting or fine."""
        if not self.val_auc:
            return "no validation data"
        gap = (self.train_auc[self.best_epoch] - self.val_auc[self.best_epoch]
               if self.best_epoch >= 0 else 0.0)
        best = self.best_val_auc
        if best < 0.70:
            return f"under-fitting (best val AUC {best:.3f})"
        if gap > 0.10:
            return f"over-fitting (train-val AUC gap {gap:.3f} at best epoch)"
        return f"healthy (val AUC {best:.3f}, train-val gap {gap:.3f})"


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve, computed from rank statistics.

    Equivalent to the probability that a random positive outranks a random
    negative (the Mann-Whitney U form), which needs no threshold sweep.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = float((labels > 0.5).sum())
    neg = float((labels <= 0.5).sum())
    return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)


# ==========================================================================
# Siamese CNN
# ==========================================================================
def train_siamese(samples, split, cfg: TrainConfig | None = None,
                  verbose: bool = True):
    """Train the Siamese CNN.  Returns ``(model, history)``."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_ds = PairDataset(samples, split.train, mode="strip",
                           augment=cfg.augment, seed=cfg.seed)
    val_ds = PairDataset(samples, split.val, mode="strip",
                         augment=False, seed=cfg.seed + 1)
    va, vb, vy, _ = val_ds.arrays()
    va_t = torch.as_tensor(va)
    vb_t = torch.as_tensor(vb)
    vy_t = torch.as_tensor(vy)

    model = SiameseCNN()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=cfg.lr_patience)
    hist = TrainHistory()
    best_state, bad = None, 0
    t_start = time.time()

    for epoch in range(cfg.epochs):
        train_ds.resample()                       # fresh negatives each epoch
        a, b, y, _ = train_ds.arrays()
        crit = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(y))
        perm = np.random.permutation(len(y))
        a_t = torch.as_tensor(a[perm])
        b_t = torch.as_tensor(b[perm])
        y_t = torch.as_tensor(y[perm])

        model.train()
        total, seen = 0.0, 0
        preds = []
        for k in range(0, len(y_t), cfg.batch_size):
            xa = a_t[k:k + cfg.batch_size]
            xb = b_t[k:k + cfg.batch_size]
            yy = y_t[k:k + cfg.batch_size]
            opt.zero_grad()
            logit = model(xa, xb)
            loss = crit(logit, yy)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yy)
            seen += len(yy)
            preds.append(torch.sigmoid(logit).detach().numpy())
        train_loss = total / max(seen, 1)
        train_auc = roc_auc(y_t.numpy(), np.concatenate(preds))

        model.eval()
        with torch.no_grad():
            vlogit = torch.cat([model(va_t[k:k + 1024], vb_t[k:k + 1024])
                                for k in range(0, len(vy_t), 1024)])
            vloss = float(nn.BCEWithLogitsLoss()(vlogit, vy_t))
            vauc = roc_auc(vy, torch.sigmoid(vlogit).numpy())

        hist.train_loss.append(round(train_loss, 5))
        hist.val_loss.append(round(vloss, 5))
        hist.train_auc.append(round(train_auc, 5))
        hist.val_auc.append(round(vauc, 5))
        sched.step(vauc)

        if vauc > hist.best_val_auc:
            hist.best_val_auc, hist.best_epoch = vauc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose:
            print(f"    epoch {epoch + 1:2d}/{cfg.epochs}  loss {train_loss:.4f}"
                  f"/{vloss:.4f}  AUC {train_auc:.4f}/{vauc:.4f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
        if bad >= cfg.patience:
            if verbose:
                print(f"    early stop at epoch {epoch + 1}", flush=True)
            break

    hist.seconds = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


# ==========================================================================
# Graph network
# ==========================================================================
def _graph_of(sample, max_candidates: int = 24):
    """Node features, edges, candidate pairs and labels for one puzzle."""
    vecs = np.stack([side_vector(s) for d in sample.descriptions
                     for s in d.sides])
    prior = None
    if sample.table is not None:
        n_sides = vecs.shape[0]
        prior = sample.table.cost.reshape(n_sides, n_sides)
        prior = np.where(np.isfinite(prior), prior, 1e6)
    x, edge_index, pairs = build_graph(vecs, sample.side_types(),
                                       sample.n_pieces,
                                       max_candidates=max_candidates,
                                       prior=prior)
    pos = {(i * 4 + s, j * 4 + t) for (i, s, j, t) in sample.positives}
    pos |= {(b, a) for (a, b) in pos}
    labels = np.array([1.0 if (int(a), int(b)) in pos else 0.0
                       for a, b in pairs], dtype=np.float32)
    return x, edge_index, pairs, labels


def train_gnn(samples, split, cfg: TrainConfig | None = None,
              verbose: bool = True):
    """Train the graph network.  Returns ``(model, history)``.

    The natural batch here is one **puzzle**: a graph is only meaningful as a
    whole, and a puzzle is exactly the context the model is meant to exploit.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if verbose:
        print("    preparing graphs...", flush=True)
    graphs = {i: _graph_of(samples[i]) for i in split.train + split.val}

    model = PuzzleGNN(VECTOR_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=cfg.lr_patience)
    hist = TrainHistory()
    best_state, bad = None, 0
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(cfg.epochs):
        model.train()
        order = list(split.train)
        rng.shuffle(order)
        total, seen, ys, ps = 0.0, 0, [], []
        for gi in order:
            x, ei, pairs, labels = graphs[gi]
            if len(pairs) == 0:
                continue
            y = torch.as_tensor(labels)
            crit = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(labels))
            opt.zero_grad()
            logit = model(x, ei, torch.as_tensor(pairs))
            loss = crit(logit, y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(y)
            seen += len(y)
            ys.append(labels)
            ps.append(torch.sigmoid(logit).detach().numpy())
        train_loss = total / max(seen, 1)
        train_auc = roc_auc(np.concatenate(ys), np.concatenate(ps))

        model.eval()
        vys, vps, vloss, vseen = [], [], 0.0, 0
        with torch.no_grad():
            for gi in split.val:
                x, ei, pairs, labels = graphs[gi]
                if len(pairs) == 0:
                    continue
                logit = model(x, ei, torch.as_tensor(pairs))
                vloss += float(nn.BCEWithLogitsLoss()(
                    logit, torch.as_tensor(labels))) * len(labels)
                vseen += len(labels)
                vys.append(labels)
                vps.append(torch.sigmoid(logit).numpy())
        vloss = vloss / max(vseen, 1)
        vauc = roc_auc(np.concatenate(vys), np.concatenate(vps))

        hist.train_loss.append(round(train_loss, 5))
        hist.val_loss.append(round(vloss, 5))
        hist.train_auc.append(round(train_auc, 5))
        hist.val_auc.append(round(vauc, 5))
        sched.step(vauc)

        if vauc > hist.best_val_auc:
            hist.best_val_auc, hist.best_epoch = vauc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose:
            print(f"    epoch {epoch + 1:2d}/{cfg.epochs}  loss {train_loss:.4f}"
                  f"/{vloss:.4f}  AUC {train_auc:.4f}/{vauc:.4f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
        if bad >= cfg.patience:
            if verbose:
                print(f"    early stop at epoch {epoch + 1}", flush=True)
            break

    hist.seconds = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist

#!/usr/bin/env python
"""CSE480 Machine Vision -- Milestone 2 entry point.

Trains both models on identical data and compares them with the classical
matcher of Milestone 1 on the same unseen test puzzles::

    python main_milestone2.py                 # the whole thing
    python main_milestone2.py --puzzles 40 --epochs 20
    python main_milestone2.py --scaling       # accuracy vs puzzle size
    python main_milestone2.py --hard --figures  # hard tests + reconstructed images

Everything it produces goes to ``results/milestone2/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch                                                  # noqa: E402

from src.ml.dataset import (PairDataset, generate_dataset,     # noqa: E402
                            generate_real_dataset)
from src.ml.evaluate import compare_methods, evaluate_method  # noqa: E402
from src.ml.gnn import GNNConfig, HAS_PYG                     # noqa: E402
from src.ml.infer import (gnn_table, predicted_matches,       # noqa: E402
                          siamese_table)
from src.ml.siamese import SiameseConfig                      # noqa: E402
from src.ml.train import TrainConfig, train_gnn, train_siamese  # noqa: E402
from src.edge_matching import build_compatibility             # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "milestone2")
CACHE = os.path.join(ROOT, "results", "milestone2", "cache")


def _save(obj, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else
            float(o) if isinstance(o, np.floating) else
            int(o) if isinstance(o, np.integer) else str(o)))
    print(f"  wrote {os.path.relpath(path, ROOT).replace(os.sep, '/')}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", type=int, default=60,
                    help="how many labelled puzzles to generate")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", action="store_true",
                    help="train on the PROVIDED photographs (detection/) "
                         "instead of generated puzzles, as Milestone 2 task 1 "
                         "requires; this is what results/milestone2/ holds")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--scaling", action="store_true",
                    help="also report accuracy against puzzle size")
    ap.add_argument("--hard", action="store_true",
                    help="also run the low-texture sweep and the real photographs")
    ap.add_argument("--real-limit", type=int, default=8,
                    help="how many real photographs to evaluate on")
    ap.add_argument("--figures", action="store_true",
                    help="also draw the charts and the reconstructed images")
    args = ap.parse_args(argv)

    global OUT
    if not args.real:
        # The committed results are the --real ones, because Milestone 2 task 1
        # asks for the provided data; a generated-puzzle run is kept separate
        # so it cannot overwrite them.
        OUT = os.path.join(ROOT, "results", "milestone2_generated")
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    t_all = time.time()

    # ---- 1. dataset preparation --------------------------------------
    print("[1/5] dataset preparation")
    if args.real:
        detection = os.path.join(ROOT, "detection")
        print("  source: the PROVIDED photographs under detection/")
        samples, split = generate_real_dataset(detection, seed=args.seed,
                                               cache_dir=CACHE)
        if not samples:
            print("  no photographs could be labelled; is detection/ present?")
            return 1
    else:
        print("  source: generated puzzles (pass --real for the provided data)")
        samples, split = generate_dataset(n_puzzles=args.puzzles,
                                          seed=args.seed, cache_dir=CACHE)
    train_pairs = PairDataset(samples, split.train, mode="strip", seed=args.seed)
    val_pairs = PairDataset(samples, split.val, mode="strip", seed=args.seed + 1)
    test_pairs = PairDataset(samples, split.test, mode="strip", seed=args.seed + 2)
    data_info = {
        "source": ("provided photographs (detection/)" if args.real
                   else "generated puzzles"),
        "n_puzzles": len(samples),
        "split_puzzles": split.counts(),
        "split_note": "split by puzzle, never by pair, and shared by both models",
        "sizes": sorted({s.grid_shape for s in samples}, key=lambda g: g[0] * g[1]),
        "pairs": {"train": train_pairs.balance(), "val": val_pairs.balance(),
                  "test": test_pairs.balance()},
        "negatives_per_positive": train_pairs.k,
        "augmentation": ["illumination", "contrast", "colour balance", "noise"],
    }
    data_info["sizes"] = [list(g) for g in data_info["sizes"]]
    print(f"  {len(samples)} puzzles, split {split.counts()}")
    print(f"  train pairs {train_pairs.balance()}")
    _save(data_info, "dataset.json")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      learning_rate=args.lr, augment=not args.no_augment,
                      seed=args.seed)

    # ---- 2. Siamese CNN ----------------------------------------------
    print("\n[2/5] training the Siamese CNN")
    siamese, hist_s = train_siamese(samples, split, cfg)
    torch.save(siamese.state_dict(), os.path.join(OUT, "siamese_cnn.pt"))
    print(f"  {hist_s.diagnosis()}")

    # ---- 3. Graph network --------------------------------------------
    print("\n[3/5] training the graph neural network")
    gnn, hist_g = train_gnn(samples, split, cfg)
    torch.save(gnn.state_dict(), os.path.join(OUT, "graph_nn.pt"))
    print(f"  {hist_g.diagnosis()}")

    _save({
        "config": cfg.as_dict(),
        "siamese_cnn": {"architecture": SiameseConfig().as_dict(),
                        "parameters": siamese.n_parameters(),
                        "size_mb": round(siamese.size_bytes() / 1e6, 3),
                        "history": hist_s.as_dict(),
                        "diagnosis": hist_s.diagnosis()},
        "graph_nn": {"architecture": GNNConfig().as_dict(),
                     "uses_torch_geometric": HAS_PYG,
                     "parameters": gnn.n_parameters(),
                     "size_mb": round(gnn.size_bytes() / 1e6, 3),
                     "history": hist_g.as_dict(),
                     "diagnosis": hist_g.diagnosis()},
    }, "training.json")

    # ---- 4. comparison on the unseen test split ----------------------
    print("\n[4/5] evaluating on the unseen test puzzles")
    comparison = compare_methods(samples, split.test, siamese=siamese, gnn=gnn)
    comparison["training_seconds"] = {
        "siamese_cnn": round(hist_s.seconds, 1),
        "graph_nn": round(hist_g.seconds, 1),
        "classical": 0.0,
    }
    _save(comparison, "comparison.json")

    # the four per-pair outputs the brief asks each model to provide
    demo = samples[split.test[0]]
    _save({
        "puzzle": demo.source,
        "classical": predicted_matches(build_compatibility(demo.descriptions),
                                       demo.descriptions)[:12],
        "siamese_cnn": predicted_matches(siamese_table(siamese, demo.descriptions),
                                         demo.descriptions)[:12],
        "graph_nn": predicted_matches(gnn_table(gnn, demo.descriptions, demo.table),
                                      demo.descriptions)[:12],
    }, "predicted_matches.json")

    # ---- 5. how performance changes with puzzle size ------------------
    if args.scaling:
        print("\n[5/5] performance against puzzle size")
        by_size = {}
        for idx in split.test:
            by_size.setdefault(samples[idx].grid_shape, []).append(idx)
        scaling = {}
        for size, idxs in sorted(by_size.items(), key=lambda kv: kv[0][0] * kv[0][1]):
            key = f"{size[0]}x{size[1]}"
            scaling[key] = {}
            for name, fn, model in (
                    ("classical", lambda s: build_compatibility(s.descriptions), None),
                    ("siamese_cnn", lambda s: siamese_table(siamese, s.descriptions), siamese),
                    ("graph_nn", lambda s: gnn_table(gnn, s.descriptions, s.table), gnn)):
                r = evaluate_method(name, samples, idxs, fn, model=model)
                scaling[key][name] = {**r.matching, **r.reconstruction,
                                      **r.cost}
            print(f"  {key}: " + ", ".join(
                f"{k} nbr {v['neighbour_accuracy']:.2f}"
                for k, v in scaling[key].items()))
        _save(scaling, "scaling.json")
    else:
        print("\n[5/5] skipped (pass --scaling)")

    # ---- 6. conditions hard enough to separate the methods -------------
    if args.hard:
        from src.ml.hard_eval import (evaluate_on_real_photographs,
                                      evaluate_texture_sweep)
        print("\n[6/6] harder conditions")
        print("  fading the picture's texture out:")
        sweep = evaluate_texture_sweep(siamese, gnn)
        _save(sweep, "texture_sweep.json")

        detection = os.path.join(ROOT, "detection")
        if os.path.isdir(detection):
            print("  the dataset photographs:")
            real = evaluate_on_real_photographs(siamese, gnn, detection,
                                                limit=args.real_limit)
            _save(real, "real_photographs.json")
            for name, v in real.items():
                print(f"    {name:12s} neighbour {v['neighbour_accuracy']:.3f}"
                      f"  position {v['position_accuracy']:.3f}"
                      f"  ({v['n_images']} images)")

    # ---- 7. the pictures behind the numbers ---------------------------
    if args.figures:
        from src.ml.figures import (comparison_chart, real_photograph_panels,
                                    texture_chart, training_curves)
        print("\n[7/7] drawing the figures")
        figdir = os.path.join(OUT, "figures")
        os.makedirs(figdir, exist_ok=True)

        training_curves({"siamese_cnn": {"history": hist_s.as_dict()},
                         "graph_nn": {"history": hist_g.as_dict()}},
                        os.path.join(figdir, "training_curves.png"))
        comparison_chart(comparison,
                         os.path.join(figdir, "test_split_comparison.png"))
        sweep_path = os.path.join(OUT, "texture_sweep.json")
        if os.path.exists(sweep_path):
            with open(sweep_path, encoding="utf-8") as fh:
                texture_chart(json.load(fh),
                              os.path.join(figdir, "texture_sweep.png"))

        # the reconstructed images, on the actual dataset photographs
        detection = os.path.join(ROOT, "detection")
        panels = []
        if os.path.isdir(detection):
            panels = real_photograph_panels(siamese, gnn, detection, figdir,
                                            limit=args.real_limit)
            with open(os.path.join(figdir, "index.json"), "w",
                      encoding="utf-8") as fh:
                json.dump([{**p, "figure": os.path.relpath(p["figure"], ROOT)}
                           for p in panels], fh, indent=2)
        print(f"  wrote {len(panels) + 3} figures to results/milestone2/figures/")

    print(f"\ndone in {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

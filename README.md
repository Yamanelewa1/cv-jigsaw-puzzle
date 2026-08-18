# Jigsaw Puzzle Reconstruction

**CSE480 Machine Vision · Milestones 1 and 2 · Ain Shams University, Faculty of Engineering**

**Milestone 1** is a reusable image-processing library that reconstructs a
jigsaw puzzle from a photograph of its shuffled pieces using **classical
techniques only** — no machine learning, and no OpenCV/SciPy
image-processing calls. Every filter, threshold, edge detector, morphological
operator, connected-component labeller, contour tracer, descriptor, matcher
and search is implemented from scratch on top of NumPy array arithmetic.

**Milestone 2** replaces the hand-designed compatibility measure with two
learned ones — a **Siamese CNN** and a **graph neural network** — and compares
all three. Everything else in the pipeline, including the assembly algorithm,
is unchanged, so the comparison isolates the matcher. See
[`report/milestone_2_report.md`](report/milestone_2_report.md) and
`python main_milestone2.py`.

Pieces may be photographed at **arbitrary rotations**; the reconstruction
algorithm determines and resolves each piece's orientation during placement.

---

## Quick start

```bash
pip install -r requirements.txt

# the headline: run the pipeline on the real dataset photographs
python main.py
python main.py --run dataset --limit 10      # just the first ten

# reconstruct any single scrambled-puzzle image
python main.py --input some_photo.jpg --grid 5x7

# repopulate data/ from the dataset
python main.py --run data

# regenerate every figure and metric used in the report
python main.py --run all

# run the tests (pytest optional)
python tests/run_tests.py
python -m pytest tests            # equivalent, if pytest is installed
```

The end-to-end routine is also a two-line library call:

```python
from src import solve_puzzle

result = solve_puzzle(image, grid_shape=(4, 5))
result.reconstruction      # the reconstructed image
result.quality["quality"]  # numerical reconstruction quality in [0, 1]
```

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── main.py                     CLI: solve an image, or run any demo
│
├── src/
│   ├── __init__.py             package doc + the end-to-end PuzzleSolver
│   ├── enhancement.py          array primitives; noise, contrast, sharpening
│   ├── thresholding.py         global / Otsu / adaptive thresholding
│   ├── edge_detection.py       Sobel, Prewitt, full Canny
│   ├── segmentation.py         foreground mask, morphology, components
│   ├── contour_extraction.py   boundary tracing, piece cropping, orientation
│   ├── piece_description.py    corners, four sides, tab/blank/flat, colour
│   ├── edge_matching.py        pairwise side compatibility
│   ├── assembly.py             greedy best-first placement + rendering
│   └── evaluation.py           synthetic ground truth + all metrics
│
├── tests/                      one file per stage, plus a pytest-free runner
├── data/
│   ├── input/                  scrambled-puzzle photographs to solve
│   ├── ground_truth/           the 5x7 answer key for each (see below)
│   └── sample_pieces/          individual pieces cropped by the pipeline
├── results/                    every figure and metric the report cites
├── notebooks/demonstration.ipynb
└── report/
    ├── milestone_1_report.md   the report source
    ├── milestone_1_report.pdf  the submitted report, with a figure appendix
    └── build_report.py         rebuilds the PDF from the Markdown
```

Rebuild the report after re-running the demos with
`python report/build_report.py`.

---

## Getting the dataset

**The dataset is deliberately not committed.** The Roboflow YOLO export is
~425 MB packed and ~453 MB extracted, and GitHub refuses any single file over
100 MB, so a repository carrying it could not be pushed at all. `.gitignore`
excludes `detection/` and `*.rar`.

Everything the brief asks to be *submitted* — source, tests, results, figures
and the report — is committed and needs nothing else. So that the headline
demo still runs on **real photographs** in a fresh clone, eight of the
photographs and their answer keys are committed too:

```
data/input/          8 full-scramble photographs (PNG, 1920x1080)
data/ground_truth/   the answer key for each: every piece's true 5x7 cell
data/sample_pieces/  8 individual pieces cropped by the segmentation stage
```

`python main.py` uses `detection/` when it is present and falls back to
`data/` when it is not (`main.dataset_scenes`), so the only difference is
that the study covers 8 photographs instead of 50.

To run on the full 50, download the **YOLOv8 / YOLO-format export** of

> <https://universe.roboflow.com/hobby-projs/puzzle-vrkx6-9xh3l/dataset/1>

and unpack it into `detection/` so that the layout is:

```
detection/
├── data.yaml
├── images/{train,valid,test}/*.jpg
└── labels/{train,valid,test}/*.txt
```

then `python main.py --run data` refreshes `data/` from it. The folder is read
and never modified.

---

## The six tasks

### 1. Image enhancement — `src/enhancement.py`

| Requirement | Implementation |
|---|---|
| Mean (box) noise reduction | `mean_filter(size)`, the unweighted neighbourhood average. Separable like the Gaussian, so also two 1-D passes. Kept as the baseline: a flat kernel turns a step into a piecewise-linear ramp with hard corners, which is what the Gaussian avoids. |
| Gaussian noise reduction, kernel from size **and** σ | `gaussian_kernel_1d(size, sigma)` samples `exp(-x²/2σ²)` and normalises; size defaults to `2⌈3σ⌉+1`. `gaussian_blur` runs two separable 1-D passes (`O(k)` instead of `O(k²)` per pixel). |
| Median filtering, **loop justified** | `median_filter`. The median is a *rank* statistic — not a linear operator, so it has no convolutional or separable form and every output pixel genuinely needs its own window's order statistics. Rather than a per-pixel Python loop, all windows are materialised at once through a stride-trick view and one vectorised `np.median` performs the `H·W` selections in compiled code; the image is processed in stripes to bound memory. The only Python loop left is over the (≤3) colour channels. |
| Contrast: histogram equalisation, contrast stretching, **histogram computed by the library** | `histogram` / `cumulative_histogram` are used by both. Equalisation maps intensities through the normalised CDF; stretching reads its two percentiles off that same CDF (not `np.percentile`). |
| Sharpening by unsharp masking **or** Laplacian on the convolution routine | Both: `unsharp_mask` (`in + amount·(in − blur)`) and `laplacian_sharpen` (`in − α·∇²in`), the latter built on the shared `convolve2d`. |

### 2. Edge detection — `src/edge_detection.py`

`sobel` and `prewitt` return `(gx, gy, magnitude, orientation)`. `canny`
implements the complete detector: Gaussian smoothing → gradient →
non-maximum suppression (gradient direction quantised to the four available
discrete orientations, evaluated by whole-array shifts) → double thresholding
→ hysteresis edge linking. Linking is a binary geodesic reconstruction
evaluated with a single connected-component pass, so its cost does not depend
on how long the edge chains are and there is no recursion limit.

Defaults `σ = 1.4`, `low = 0.05`, `high = 0.15` on the normalised gradient
magnitude (the usual `high ≈ 3·low` ratio). σ = 1.4 removes the texture of the
dataset's black cloth while keeping tab/blank curvature intact; the effect of
each parameter is shown in `results/edge_visualisations/`.

### 3. Piece segmentation and contour extraction — `src/segmentation.py`, `src/contour_extraction.py`

* **Foreground mask** — `foreground_mask` supports global Otsu, local adaptive
  thresholding, and a `"background"` mode that Otsu-thresholds the *colour
  distance to the estimated background* instead of the luma. That last mode is
  the default for colour input, because a navy patch printed on a piece has
  almost the same luma as a black cloth but a very different chroma.
* **Morphology from scratch** — `erode`/`dilate` as intersections/unions of
  translated copies (no `H·W·|SE|` temporary), plus `opening`, `closing`,
  `fill_holes` and a self-dual `majority_smooth`.
* **Splitting touching pieces** — `distance_transform` (exact Euclidean, via
  the separable lower-envelope algorithm) and `split_touching`, a
  distance-transform watershed. Pieces that touch in a photograph share one
  connected component and *no* threshold separates them — the information
  needed is shape, not intensity. A blob whose area says it holds *k* pieces
  is thresholded on its (smoothed) distance map, sweeping the level down to
  the highest one yielding exactly *k* substantial cores, and those cores are
  grown back at equal speed until they meet on the watershed line. On the
  dataset photographs this is what lifts segmentation recall from **0.81 to
  0.97**.
* **Connected components from scratch** — `connected_components` is the
  classic two-pass union-find algorithm applied to *runs* rather than pixels:
  each row is decomposed into maximal horizontal runs with one vectorised
  `diff`, overlapping runs in consecutive rows are unioned (widened by one
  pixel for 8-connectivity), then labels are resolved and renumbered. Same
  labelling as the textbook pixel scan, but only `O(#runs)` work in Python.
* **Boundary tracing** — `trace_boundary` is Moore-neighbour tracing with
  **Jacob's stopping criterion** (stop when the start pixel is re-entered
  *from the same direction*), which the naive "stop on reaching the start"
  rule gets wrong on the thin neck of a tab.
* **Piece extraction** — `extract_pieces` crops each component to its bounding
  box and stores the mask, the contour, the centroid and a normalised
  orientation from the minimum-area rectangle (`min_area_rect`, a
  rotating-calipers search over convex-hull edges).

### 4. Piece-edge description — `src/piece_description.py`

Corners are the make-or-break step, since every profile and colour strip is
measured *between* two of them. Three strategies are tried in order:

1. **Body-edge model** (normal path). A piece is a rectangle plus bumps, so
   its four straight edges share one direction modulo 90°. That direction is
   recovered as `arg(Σ exp(4iφ))/4` over the tangent angles — a circular tab
   arc sweeps all directions and cancels itself out — the contour is rotated
   into that frame, and each edge is located as the **mode** of the
   perpendicular coordinate of the boundary points whose tangent runs along
   it. The corners are the four line intersections, snapped onto the contour;
   a snap further than 8 % of the body rejects the model.
2. **Curvature search** — convex local maxima of the turning angle, with a
   combinatorial choice of the best four.
3. **Minimum-area rectangle**, snapped onto the contour.

Why not curvature first: the tip of a tab of radius ρ turns through 90° over
an arc of only ≈1.6ρ pixels, which is *sharper* than the piece's own corner at
any scale large enough to be noise-free — a curvature search really does
mistake tab tips for corners.

Two further passes use the fact that the pieces of one puzzle are **not
independent samples** — they were cut from one grid with one tool:

* `refine_descriptions` takes the median body size over the pieces whose own
  corner estimate is self-consistent, and re-fits the ones that disagree with
  `fit_body_rectangle`, which finds the maximum-coverage rectangle of the
  agreed size. A single skewed piece otherwise yields a skewed descriptor
  *and* a visibly warped piece in the rendered reconstruction.
* `calibrate_side_types` learns the flat/tab boundary from the puzzle's own
  amplitude distribution instead of trusting a fixed constant. An `R×C` grid
  exposes exactly `2(R+C)` flat sides, so the widest gap in the sorted
  amplitudes that yields a grid-consistent count is chosen. On a dataset
  photograph this recovers **24 of the 24** expected flat sides where the
  fixed default found 14.

Each side then gets

* a **type** — `flat`, `tab` (outward) or `blank` (inward), from its peak
  deviation relative to the calibrated threshold;
* a **shape profile** — the signed perpendicular deviation from the chord,
  normalised by the chord length and resampled at equal arc length;
* a **colour strip** — RGB sampled at three depths just inside the side, offset
  along the **chord** normal (the local normal runs almost along the boundary
  in the throat of a blank and would sample background), validated against the
  mask eroded by two pixels.

### 5. Piece-edge matching — `src/edge_matching.py`

For an ordered pair of sides `(a, b)`:

> **D(a, b) = w_s · D_shape(a, b) + w_c · D_colour(a, b) + w_l · D_length(a, b)**

with `D = +∞` for inadmissible pairs. Hard constraints ("does the shape
fit"): a **flat** side is a border of the whole puzzle and can never be an
interior seam; a **tab** must meet a **blank**; a side never matches another
side of the same piece.

When two pieces sit side by side one side is traversed in the opposite
direction to the other and their outward normals oppose, so a perfect fit
satisfies `p_a(t) = −p_b(1−t)`. The residual of that identity, in RMS, is the
shape cost; the colour strips must satisfy `C_a(t) = C_b(1−t)` and their
squared differences give the colour cost:

```
D_shape (a,b) = sqrt( 1/M   · Σ_i    ‖ p_a(t_i) + p_b(1−t_i) ‖² )
D_colour(a,b) = sqrt( 1/3MD · Σ_i,d  ‖ C_a(t_i,d) − C_b(1−t_i,d) ‖² )
D_length(a,b) = |L_a − L_b| / max(L_a, L_b)
```

Both profiles are normalised by their own chord length, so the terms are
dimensionless and piece-size independent. Corner localisation is accurate to
a few pixels, and a corner found early on one piece and late on its neighbour
shifts the whole profile — so both `D_shape` and `D_colour` are **minimised
over integer sample shifts of up to 5 % of the side**, evaluated on the
overlap.

**Illumination normalisation** (`colour_norm="meanstd"`). On the dataset
photographs the pieces lie all over a table, each lit slightly differently, so
two strips facing each other across a true seam differ by an offset *and* a
gain even where the picture continues perfectly. Subtracting each strip's own
mean and dividing by its own spread removes both and compares only the pattern
along the strip. It is neutral on synthetic puzzles (uniform lighting by
construction) and clearly helps on real ones — the matcher's top-1 rate rises
from 0.19 to 0.25 and its best-buddy precision from 0.26 to 0.40 — so it is
off by default and on for the dataset.

**Weights: `w_s = 1.0`, `w_c = 1.0`, `w_l = 0.5`** — shape and colour
contribute equally, length is a half-weight tie breaker. This is justified by
the two terms having comparable dynamic range on this data (see
`results/evaluation_results/matching_study.json` and `weight_study.json`), so
neither needs rescaling to be heard. They are complementary rather than
redundant: shape disambiguates uniformly coloured regions, colour
disambiguates the many geometrically interchangeable tab/blank pairs of a
machine-cut puzzle. Using either alone measurably degrades the result.

### 6. Assembly algorithm — `src/assembly.py`

Greedy best-first placement on a fixed `R×C` grid (inferred from the piece
count and the number of border pieces when not supplied). The grid is
anchored with a corner piece rotated so its two flats face North and West.
At each step every empty frontier cell, every unused piece and each of its
four rotations is scored by the mean dissimilarity to its already-placed
neighbours, subject to the border/flat and tab/blank constraints — so **each
piece's rotation is resolved as it is placed**.

**Tie-breaking rule.** Candidates are ordered by

```
(not a best-buddy seam, mean seam cost, −#matched neighbours,
 −confidence margin, piece index, rotation)
```

*Best-buddy* seams — where the two sides are each other's cheapest partner in
the entire puzzle — are committed first, because a mutual first choice is far
stronger evidence than a merely cheap one. Then cheapest cost; then the cell
constrained by more placed neighbours; then the larger margin to the
second-best candidate for that cell; the last two make the result
deterministic.

**Dead ends.** The search never aborts. It relaxes constraints in three
documented stages — (i) drop the border/flat requirement at a fixed
`BORDER_PENALTY`, (ii) allow illegal tab/tab and blank/blank seams at
`DEADEND_PENALTY`, (iii) place whatever is left in reading order — and marks
those placements as *forced*.

**Best arrangement guaranteed.** The greedy pass is repeated from every
candidate seed and the best result is kept, ranked by `(pieces placed, forced
placements, mean seam cost)`, so `assemble` always returns the best
arrangement obtained, complete or not.

---

## Reconstruction quality

`src/evaluation.py` provides both families of measure.

**Reference-free** — returned by the end-to-end routine with every image:
`quality = 1 − mean(seam cost) / mean(all admissible costs)`, clipped to
`[0, 1]`. The denominator is what an arrangement pairing sides at random
would pay, so the score answers *how much better than chance is this?*

**Ground-truth based** — the dataset photographs come with no answer key, so
`evaluation.py` also contains the synthetic puzzle **generator** that cuts any
picture into interlocking pieces (a piece is its rectangular body ∪ its tab
circles ∖ its neighbours' blank circles, so mating sides are exactly
complementary), shuffles and rotates them onto a canvas, and remembers the
answer. That makes measurable: `direct_accuracy` (right cell, right rotation,
maximised over the four global quarter-turns), `neighbour_accuracy` (fraction
of true adjacencies reproduced — invariant to any global transform),
`rotation_accuracy`, `matching_accuracy` (how often the compatibility measure
alone ranks the true partner first, isolating the matcher from the search)
and `image_metrics` (MSE / PSNR / SSIM against the original picture, SSIM
implemented from scratch).

### Ground truth on the real dataset

The dataset labels every piece with its **identity** (1-35), and those ids turn
out to be the row-major positions of the finished 5x7 puzzle: id `k` sits at
`((k-1)//7, (k-1)%7)`. The evidence is the flat sides — counting them per id
across photographs gives **zero flats for all fifteen ids the hypothesis calls
interior**, and at least one for the ids it calls border. That turns the
identity labels into a real answer key, so reconstruction accuracy is measured
on the actual photographs, not only on synthetic puzzles
(`main.dataset_true_cells`).

Scored against the annotated boxes, over all **50** full-scramble photographs
(`python main.py`, per-image rows in
`results/evaluation_results/dataset_study.json`):

| Measure | Value |
|---|---|
| Piece recall (annotated pieces isolated) | **0.973** |
| Piece precision (components that are pieces) | **0.963** |
| F1 | 0.965 |
| Pieces isolated | 35.4 / 35 on average |
| Flat sides found | **23.1** vs 24 the 5x7 grid must expose |
| Corner pieces found | 3.2 / 4 |
| Time per photograph | 32 s at native 1920x1080 |

Splitting touching pieces is what makes this work: without it recall is
**0.81**, with it **0.94** (and 0.97 at the reduced 1280 px working size used
for the segmentation-only study). Description is essentially correct — 23.1
flat sides found against 24 expected — so the pieces, their corners, their
sides and their types are all recovered from real photographs.

The same 50 photographs, scored against the answer key of the previous
section:

| Measure | Real photographs | Synthetic (for contrast) |
|---|---|---|
| Neighbour accuracy | **0.220** | 0.970 |
| Position accuracy | **0.131** | 0.989 |
| Matcher top-1 | 0.324 | 0.90 |
| Best single image | 0.328 neighbour | 1.000 |

**The reconstruction does not succeed on this dataset.** Roughly one adjacency
in five is right, against about one in twenty by chance -- a real signal, but
nowhere near a solved puzzle. That is the honest result, and the cause is
identifiable rather than mysterious.

The numbers above are with the photograph preset (`main.DATASET_SOLVER`);
the two settings that separate it from the library defaults were each
measured over all 50 answer-keyed photographs, from cached descriptors so
that only the stage under test varies:

| Configuration | Matcher top-1 | Neighbour acc. | Position acc. |
|---|---|---|---|
| colour SSD, hard border rule | 0.267 | 0.189 | 0.119 |
| **+ MGC** photometric term | 0.324 | 0.192 | 0.132 |
| **+ soft border rule** (the preset) | 0.324 | **0.220** | **0.131** |

Both changes are real but small: +16 % neighbour accuracy overall, better on
26 of the 50 photographs and worse on 18. The synthetic puzzles stay at 100 %
under the same settings, so neither is a trade of clean-input accuracy for
noisy-input accuracy.

**Why it is still only 0.22.** There are two distinct limits, and only the
second turned out to be fixable.

*The compatibility measure.* Asking of every side whether its cheapest partner
lies on a genuinely adjacent piece -- which needs no knowledge of any
rotation, so it isolates the matcher -- gives a top-1 rate of **0.324** against
**0.10** for chance, where the same measure reaches **0.90** on synthetic
puzzles. Attempts to close that gap:

| Attempt | Varied | Outcome |
|---|---|---|
| More search | restarts 1 to 60 | saturates at 4 |
| Richer colour descriptor | 7 sampling-depth variants | best raises best-buddy precision 0.52 to 0.57, but *lowers* reconstruction |
| Wider alignment search | shift 0 % to 20 % of a side | optimum is the existing 5 %; both directions worse |
| Per-side cost normalisation | divide each side's row by its own 2nd-best / low quantile / z-score | top-1 0.343 to 0.381, reconstruction flat |
| Beam search | width 20-150, top-3 to top-5 | no gain on real photos, and a regression on clean input |
| Cluster merging (Kruskal-style) | commit cheapest merges globally instead of growing from a seed | *worse* (0.173 vs 0.213), so the seeded greedy is not the weak link |
| Removing non-puzzle objects | oracle: keep only pieces matched to an annotation | **no change at all** (0.213 to 0.214) |
| **MGC** (gradient continuity) | Gallagher's Mahalanobis gradient compatibility | matcher +21 % (top-1 0.267 to 0.324); reconstruction +11 % on position accuracy |

The pattern is consistent: every change that improves the *matcher* leaves the
*reconstruction* nearly where it was. At a top-1 of ~0.3, no search recovers a
35-piece puzzle, and the measure is the reason.

*The border rule.* The second limit was a genuine defect rather than a
property of the data. The flat/tab/blank classifier gets a piece's flat count
right on only **79.6 %** of pieces here (against ~100 % on synthetic puzzles),
and only **52 %** of the true corner pieces are recognised as corners. The
assembler was treating "a rim cell must show a flat outwards" as a *hard*
filter, so on these photographs it rejected correct placements more often than
it prevented wrong ones. Charging `BORDER_PENALTY` instead
(`assemble(border_mode="soft")`) is what produced the 0.192 to 0.220 step
above, and costs nothing on clean input.

Three properties of this particular puzzle explain it, all properties of the
data rather than of the algorithms:

1. **The picture is almost entirely white.** The HIWIN advert is white and
   pale grey over most of its area, so colour strips along most seams are
   nearly identical. The weight study above measured that colour carries the
   great majority of the discriminative power (top-1 0.880 alone, versus
   0.078 for shape alone) -- on a white puzzle that dominant term has almost
   nothing to work with.
2. **The tabs are machine-cut and nearly identical**, so the shape term --
   already the weaker by an order of magnitude -- cannot compensate.
3. **Each piece is lit and angled differently** on the table. Illumination
   normalisation recovers part of that (top-1 0.193 to 0.251); perspective
   differences across the frame remain.

The same code reconstructs a 35-piece puzzle perfectly when the pieces carry
distinguishable picture content, which places the limitation in the input
rather than in the method. The clearest way to close it is a descriptor that
reads the fine printed texture across a seam instead of its mean colour --
normalised cross-correlation of the boundary strips, or gradient continuity
(Gallagher's MGC) -- since on a mostly-white picture the residual texture is
exactly the signal that survives.

### Validation on synthetic puzzles

Reconstruction accuracy also needs a case where the answer is known *and* the
input is clean, to show the assembly itself is correct. `python main.py --run
validate` cuts seven puzzles from a source picture, six with pieces at
arbitrary rotations (full table in `results/evaluation_results/benchmark.json`):

| Puzzle | Pieces | Rotated | Neighbour acc. | Position acc. | SSIM | Time |
|---|---|---|---|---|---|---|
| 2x3 | 6 | yes | 1.00 | 1.00 | 0.81 | 1.7 s |
| 3x4 | 12 | yes | 1.00 | 1.00 | 0.76 | 5.2 s |
| 3x4 | 12 | yes | 1.00 | 1.00 | 0.80 | 5.0 s |
| 4x5 | 20 | yes | 1.00 | 1.00 | 0.79 | 12.5 s |
| 4x6 | 24 | yes | 0.79 | 0.92 | 0.77 | 17.7 s |
| 5x7 | 35 | yes | 1.00 | 1.00 | 0.79 | 40.2 s |
| 5x7 | 35 | no | 1.00 | 1.00 | 0.80 | 35.5 s |

**Six of seven perfect**, mean neighbour accuracy 0.970. Comparing the rotated
and unrotated 35-piece rows shows arbitrary orientation costs essentially
nothing — the rotation is resolved during placement.

How much each term of the compatibility measure contributes, swept over three
puzzles (`results/evaluation_results/weight_study.json`):

| `w_shape` | `w_colour` | `w_length` | Top-1 match rate | Neighbour acc. |
|---|---|---|---|---|
| 1 | 0 | 0 | 0.078 | 0.483 |
| 0 | 1 | 0 | 0.880 | 1.000 |
| 1 | 1 | 0 | **0.900** | 1.000 |
| **1** | **1** | **0.5** (default) | 0.890 | 1.000 |

Shape alone is weak on a machine-cut puzzle — every tab is die-cut to nearly
the same silhouette — but it still improves the matcher on top of colour,
because it disambiguates exactly what colour cannot: seams crossing a
uniformly coloured region.


## Design notes and pitfalls

Three failures that shaped the implementation, all visible in the tests:

* **Opening-then-closing destroys tab/blank complementarity.** Opening rounds
  only convex features and closing only concave ones, so a tab and the blank
  it fits get distorted by *different* amounts and the shape term stops
  discriminating. `majority_smooth` (a binary median) is self-dual and does
  not have this problem.
* **A raw 8-connected contour is a staircase** whose arc length is up to ~8 %
  longer than the underlying curve, by an amount that depends on the piece's
  rotation. Since sides are resampled by arc length, that alone misaligns the
  profiles of two mating sides photographed at different angles; the contour
  is smoothed before it is split.
* **Sampling colour along the local normal fails inside a blank**, where the
  normal runs almost along the boundary and the strip fills up with
  background. The chord normal always points from the cut line into the body.
* **Touching pieces are one connected component**, and no threshold splits
  them — the cue is shape, not intensity. A distance-transform watershed
  raised segmentation recall on the real photographs from 0.81 to 0.97.
* **A per-piece constant is worse than a per-puzzle one.** A fixed flat/tab
  threshold found 14 of 24 flat sides on a dataset photograph; reading the
  boundary off the puzzle's own bimodal amplitude distribution, constrained by
  the fact that an `R x C` grid has exactly `2(R+C)` flats, found 24.

## Testing

144 tests, one file per stage plus the end-to-end routine:

```
tests/test_enhancement.py        primitives, filters, histograms, sharpening
tests/test_thresholding.py       incl. a brute-force check that Otsu really
                                 maximises between-class variance
tests/test_edge_detection.py     kernels, NMS thinning, hysteresis linking
tests/test_segmentation.py       morphology, components, distance transform,
                                 watershed splitting, tracing, warping
tests/test_piece_description.py  orientation, corners, side types, strips
tests/test_edge_matching.py      the formula, the table, best buddies
tests/test_assembly.py           placement, constraints, dead ends, rendering
```

`python tests/run_tests.py` runs them without pytest installed.

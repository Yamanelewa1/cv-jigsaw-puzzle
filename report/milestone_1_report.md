# Milestone 1 — Jigsaw Puzzle Reconstruction with Classical Computer Vision

**CSE480 Machine Vision · Ain Shams University, Faculty of Engineering, Mechatronics Engineering Department · Summer 2026**

---

## 0. Requirement checklist

Every item the brief asks for, where it is implemented, and the test that pins it down.

| Brief requirement | Implementation | Test |
|---|---|---|
| Noise reduction: **mean** filter | `enhancement.mean_filter` (§2.1) | `test_mean_filter_equals_the_plain_neighbourhood_average` |
| Noise reduction: Gaussian, kernel derived from size **and** sigma | `enhancement.gaussian_kernel_1d` / `gaussian_kernel` / `gaussian_blur` | `test_gaussian_kernel_normalised_and_symmetric` |
| Noise reduction: median filter, **any loop justified** | `enhancement.median_filter` (§2.1) | `test_median_filter_removes_impulses_gaussian_does_not` |
| Contrast: histogram equalisation and stretching, **histogram computed by the library** | `enhancement.histogram`, `cumulative_histogram`, `histogram_equalization`, `contrast_stretch` | `test_histogram_equalization_flattens_the_histogram` |
| Sharpening: unsharp masking **or** Laplacian on the convolution routine | both: `enhancement.unsharp_mask`, `laplacian_sharpen` on `convolve2d` | `test_unsharp_mask_increases_edge_contrast` |
| Thresholding: global, Otsu, adaptive (mean or Gaussian) | `thresholding.global_threshold`, `otsu_threshold`, `adaptive_threshold` | `test_otsu_maximises_between_class_variance` |
| Sobel and Prewitt returning magnitude **and** orientation | `edge_detection.sobel`, `prewitt` | `test_gradient_of_a_vertical_edge_is_horizontal` |
| Complete Canny: smoothing, gradient, NMS, double threshold, hysteresis linking | `edge_detection.canny` and its five stages | `test_canny_outlines_a_rectangle_without_filling_it` |
| Parameter choices stated and their effect illustrated | §3 and `results/edge_visualisations/` | `test_canny_sigma_controls_the_amount_of_detail` |
| Foreground mask from the thresholding routines | `segmentation.foreground_mask` | `test_foreground_mask_separates_pieces_from_the_background` |
| Connected components, **from scratch** | `segmentation.connected_components` (two-pass union-find over runs) | `test_connected_components_counts_and_labels_blobs` |
| Trace the boundary of each piece | `contour_extraction.trace_boundary` (Moore + Jacob) | `test_trace_boundary_walks_a_square_once` |
| Extract to bounding box, store mask, contour, normalised orientation | `contour_extraction.extract_pieces`, `normalize_piece` | `test_extract_pieces_crops_masks_and_contours` |
| Locate four corners, divide boundary into four sides | `piece_description.find_corners`, `split_sides` | `test_find_corners_is_accurate_for_almost_every_piece` |
| Classify every side tab / blank / flat from geometry | `piece_description.classify_side` | `test_classify_side_flat_tab_and_blank` |
| Sample a colour strip along the interior of each side | `piece_description.sample_color_strip` | `test_colour_strip_never_samples_the_background` |
| One number per side pair, shape fit **and** colour continuity | `edge_matching.side_compatibility`, `build_compatibility` | `test_table_matches_the_scalar_formula` |
| **Exact formula stated**, with shape-versus-colour weighting | §6.1 and §6.2 | `test_weights_change_the_cost_predictably` |
| Greedy best-first assembly, grid seeded from a corner/border piece | `assembly.assemble`, `infer_grid_shape` | `test_assembly_fills_the_grid_and_uses_each_piece_once` |
| Rotation resolved during placement | rotation is part of every candidate; `Placement.rotation` | `test_rotated_puzzle_is_reconstructed_exactly` |
| **Tie-breaking rule specified** | §7.1 | `test_border_pieces_show_their_flats_outwards` |
| **Dead ends and unplaceable pieces handled** | §7.2, three relaxation stages | `test_assembly_returns_a_full_arrangement_even_with_useless_costs` |
| **Best arrangement returned even when incomplete** | §7.4, restarts ranked and kept | `test_restarts_never_return_a_worse_arrangement` |
| End-to-end routine returning image **and** a quality number | `src.solve_puzzle` / `PuzzleSolver`, `evaluation.seam_quality` | `test_solve_puzzle_wrapper_reports_quality` |

---

## 1. Overview

This milestone delivers a reusable image-processing library that reconstructs a jigsaw puzzle from a photograph of its shuffled pieces using classical techniques alone. Every operation named in the brief is implemented from scratch on top of NumPy array arithmetic: no OpenCV or SciPy filtering, morphology, labelling, feature or matching code is used anywhere in `src/`. Pillow appears only in `main.py`, to decode and encode image files.

The library exposes each operation as a documented, independently testable function, and provides an end-to-end routine — `src.solve_puzzle` / `src.PuzzleSolver` — that accepts a scrambled puzzle image and returns the reconstructed image together with a numerical measure of reconstruction quality. Pieces may be presented at arbitrary orientations; each piece's rotation is determined and resolved during placement.

The pipeline is:

> enhance → threshold → segment → split touching → trace contours → describe pieces → match sides → assemble → render + score

**What is measured on what.** The primary target is the **provided dataset**: the 50 photographs (43 training, 7 validation) that show all 35 pieces of the jigsaw scattered on a dark cloth. Those are real inputs with real difficulties, and §9 reports the pipeline's behaviour on every one of them.

The dataset carries no explicit answer key — nobody labels which cell of the finished picture each photographed piece belongs to. §9.1 shows that one is nevertheless recoverable: the class ids of the annotations *are* the row-major positions of the finished 5×7 puzzle, which turns them into ground truth and lets reconstruction accuracy be measured on the actual photographs. Synthetic puzzles are used for one purpose only — to demonstrate that the assembly stage is correct when its input is clean (§8) — and every number is labelled with which of the two it comes from.

---

## 2. Task 1 — Image enhancement

All routines live in `src/enhancement.py`, which also holds the shared primitives (dtype and colour conversion, padding, the sliding-window view, `convolve2d`, `resize_bilinear`, `integral_image`) that the rest of the package is built on.

### 2.1 Noise reduction

**Mean.** `mean_filter(size)` replaces each pixel by the unweighted average of its `size x size` neighbourhood. It is the maximum-likelihood estimate of a constant patch under additive Gaussian noise, so it attenuates that noise by a factor of `size`, and the flat kernel is separable in the same way the Gaussian is, so it also runs as two 1-D passes. It is the baseline rather than the tool: a flat kernel has a sinc frequency response with large side lobes, and in the spatial domain it turns a step into a piecewise-linear ramp whose slope jumps at both ends — hard corners along an edge, where the Gaussian's response is smooth everywhere. A test measures exactly that, comparing the largest second difference of the two step responses.

**Gaussian.** `gaussian_kernel_1d(size, sigma)` samples `exp(-x² / 2σ²)` and normalises it; the kernel is derived from both arguments, and when `size` is omitted it defaults to `2⌈3σ⌉ + 1`, which truncates less than 0.3 % of the mass. The 2-D kernel is the outer product of two 1-D kernels, and `gaussian_blur` exploits that separability with two 1-D passes — `O(k)` per pixel instead of `O(k²)`. A test asserts that the separable result equals a full 2-D convolution to 1e-10.

**Median — and why a loop is needed.** The median is a *rank* statistic. Unlike the Gaussian it is not a linear operator, so it cannot be written as a convolution and has no separable or recursive form; every output pixel genuinely requires the order statistics of its own `size²` neighbours, i.e. an explicit pass over the window. What the implementation avoids is making that pass a *Python* loop over pixels (≈10⁶ iterations for a 1 MP image). Instead all windows are materialised at once as a stride-trick view (`sliding_window_view`, which shares memory and copies nothing) and a single vectorised `np.median` performs the `H·W` independent selections in compiled code. The image is processed in horizontal stripes so the temporary stays bounded (~64 MB), and the only Python-level loop that remains is over the at most three colour channels — a colour median must be taken per band or the filter invents colours that are in no pixel of the image.

**Measured effect** (`results/evaluation_results/enhancement_study.json`), on a dataset crop corrupted with Gaussian noise (σ = 0.05) plus 2 % salt and 2 % pepper:

| Filter | PSNR | SSIM |
|---|---|---|
| none | 17.46 dB | 0.325 |
| Gaussian σ = 1.5 | 25.92 dB | 0.741 |
| Gaussian σ = 2.5 | 23.81 dB | 0.772 |
| **median 3×3** | **31.18 dB** | 0.864 |
| median 5×5 | 30.46 dB | 0.911 |

This is the textbook result made concrete: a linear filter *smears* an impulse over its whole support, whereas the median *deletes* it. That is why the pipeline's default chain applies the median first and the Gaussian second.

### 2.2 Contrast adjustment

`histogram` quantises to `bins` levels and counts with `np.bincount`; `cumulative_histogram` is its running sum. Both contrast operators are driven by that histogram, as the brief requires.

**Histogram equalisation** uses the normalised CDF as its transfer function, `s = CDF(r)`, which pushes the intensity distribution towards uniform. For colour input the default equalises the luma and re-applies the original chroma ratios, so hue is preserved; `per_channel=True` equalises R, G and B independently (stronger, but it shifts colours).

**Contrast stretching** is the linear map `(in − lo)/(hi − lo)` clipped to `[0, 1]`, where `lo` and `hi` are read off the *cumulative histogram built by this library* rather than from `np.percentile`. Percentiles rather than the raw min/max make it robust to a handful of hot or dead pixels.

**Measured effect.** On a deliberately low-contrast patch (intensity range 0.42–0.51), equalisation raises the intensity standard deviation from 0.022 to 0.142 — a 6.4× increase in visible contrast. Its histogram *entropy*, however, stays at 2.39 bits, and that is worth stating plainly: equalisation is a monotone mapping through a 256-bin CDF, so it can spread the levels the image already distinguishes but cannot create new ones. Contrast stretching, being a continuous linear map, reaches 5.57 bits on the same patch. On a full photograph, which is dominated by a large dark background region, the CDF has a step at zero that no monotone mapping can spread, and equalisation changes entropy only from 5.35 to 5.11 bits while stretching reaches 5.60.

### 2.3 Sharpening

Both requested forms are provided. `unsharp_mask` computes `in + amount·(in − blur(in))`, adding back a multiple of the high-pass residual, with an optional threshold that suppresses the boost where the residual is small so flat noisy regions are left alone. `laplacian_sharpen` computes `in − α·∇²in` using the discrete Laplacian (4- or 8-neighbour) applied through the shared `convolve2d` routine — the second derivative responds to exactly the high-frequency content that sharpening should boost.

### 2.4 Thresholding (task 1d)

`src/thresholding.py` provides all three required families.

* **Global** — a fixed level, plus `isodata_threshold`, the iterative Ridler–Calvard rule that starts at the mean and repeatedly moves to the average of the two class means.
* **Otsu** — maximises the between-class variance `σ_b²(t) = w₀w₁(μ₀ − μ₁)²`, evaluated for all candidate levels at once from cumulative sums of the histogram. One refinement is worth recording: when the two modes are well separated the criterion is *flat* across the whole empty valley between them, so a plain `argmax` returns the valley's left edge, a threshold pressed right up against the darker mode. This implementation returns the **midpoint of the maximising plateau** — the same optimum, but as far from both modes as possible, and therefore far more tolerant of noise. A test verifies by brute force that the returned level really does maximise the criterion, and another checks that it then agrees with isodata on a clean bimodal image.
* **Adaptive** — `pixel > local_statistic − c`, with the mean variant evaluated in `O(1)` per pixel through an integral image (so the cost is independent of the block size) and a Gaussian-weighted variant for smoother boundaries. A test constructs a constant-contrast pattern under a strong illumination ramp and confirms that adaptive thresholding recovers it where global Otsu cannot.

On a dataset photograph, Otsu returns 0.346 and isodata 0.349 — good agreement, as expected for a clean pieces-on-cloth scene.

---

## 3. Task 2 — Edge detection

`src/edge_detection.py`. `sobel` and `prewitt` apply their respective derivative pairs and return `(gx, gy, magnitude, orientation)`, with orientation in radians in `(−π, π]` measured with `x` right and `y` down. Both operators factor into a smoothing kernel times a differencing kernel; Sobel's `[1 2 1]` smoother weights the centre row twice as heavily as Prewitt's `[1 1 1]`, which makes it slightly less noise-sensitive and less "boxy".

`canny` implements the complete detector:

1. **Gaussian smoothing** at scale σ;
2. **gradient** computation (Sobel);
3. **non-maximum suppression** — the continuous gradient direction is quantised to the four available discrete orientations (0°, 45°, 90°, 135°) and a pixel survives only if its magnitude is at least that of both neighbours along that direction. This is implemented by shifting the whole array once per orientation, with no per-pixel loop;
4. **double thresholding** into strong and weak sets;
5. **hysteresis edge linking** — the transitive closure "keep weak pixels 8-connected to a strong pixel". This is a binary geodesic reconstruction, and it is evaluated with a *single* connected-component labelling pass rather than an iterative flood, so its cost does not depend on how long the edge chains are and there is no recursion-depth limit.

### Parameter choices and their effect

**σ = 1.4** is the default. It is large enough to erase the woven texture of the dataset's black cloth (which otherwise litters the background with spurious edges) and small enough to leave the curvature of a tab or blank intact. `results/edge_visualisations/` shows σ ∈ {0.8, 1.4, 2.5} on the same crop: at 0.8 the printed picture inside each piece is fully detected along with the cloth grain; at 2.5 essentially only the piece silhouettes survive, and the tab necks begin to round off.

**low = 0.05, high = 0.15** on the *normalised* gradient magnitude, i.e. the usual `high ≈ 3·low` ratio. On the reference crop these keep 3.58 % of pixels as edges; loosening to (0.02, 0.06) raises that to 4.17 % and admits cloth texture, while tightening to (0.10, 0.25) drops it to 3.09 % and starts breaking the silhouette into arcs. A `use_percentiles` mode interprets the two levels as percentiles of the non-zero magnitudes instead, which makes the detector exposure-independent across the dataset.

A worked illustration on representative pieces — the smoothed image, the gradient magnitude and orientation, the thinned ridge, the strong and weak sets, and the linked result — is written to `results/edge_visualisations/contact_sheet.png`.

---

## 4. Task 3 — Piece segmentation and contour extraction

### 4.1 Foreground mask

`segmentation.foreground_mask` offers global Otsu, local adaptive thresholding, and a `"background"` mode that is the default for colour input. That mode estimates the background colour as the median of a thin frame around the image — the outermost pixels of a puzzle photograph are essentially always table or cloth, and the median is insensitive to the occasional piece that touches the frame — then Otsu-thresholds the per-pixel **Euclidean colour distance** to it.

Thresholding colour distance rather than luma is what makes segmentation survive dark picture content on a dark table: a navy patch printed on a piece has almost the same luma as black cloth but a very different chroma. When the pipeline used luma alone, dark printed regions were classified as background and cut pieces into fragments; the colour-distance criterion removed that failure entirely.

### 4.2 Morphology

Implemented from scratch: `erode` and `dilate` as intersections and unions of translated copies of the mask — one boolean pass per structuring-element element, avoiding the `H·W·|SE|` temporary that a window view would need — plus `opening`, `closing`, `fill_holes` (background components not touching the frame are holes; found with one labelling pass, not an iterative flood) and `majority_smooth`.

`majority_smooth` deserves its own note, because it fixes a subtle bug. Opening rounds only *convex* features and closing rounds only *concave* ones, so applying them in sequence distorts a tab and the blank it fits into by **different** amounts — and the whole shape-matching stage rests on those two being mirror images. A majority (binary median) filter is self-dual: it treats foreground and background symmetrically, so complementary features stay complementary.

### 4.3 Connected components, from scratch

`connected_components` is the classic **two-pass algorithm with union-find**, applied to *runs* rather than to individual pixels:

1. each row is decomposed into maximal horizontal runs of foreground pixels, found with one vectorised `diff`, and every run gets a provisional label;
2. runs in consecutive rows that overlap are neighbours, so their labels are unioned — with 8-connectivity the overlap test is widened by one pixel on each side so diagonal contacts count;
3. a second pass replaces every provisional label by the root of its equivalence class, renumbers the roots `1..n`, and paints the runs into the output.

This produces exactly the labelling of the textbook pixel-wise scan while leaving only `O(#runs)` work in Python — a few thousand operations for a typical mask instead of a million. Union–find uses path compression, so the pass is effectively linear. A 1280×720 mask labels in about 0.03 s.

`component_stats` and `filter_components` then reject clutter. The area filter is expressed relative to the **median** component area rather than as an absolute pixel count, which is the robust way to drop the screwdrivers, bolts and cable ties present in the dataset photographs without hard-coding anything about resolution.

### 4.4 Splitting touching pieces

In every one of the dataset photographs several pieces are physically in contact, and two touching pieces are a **single connected component**. No threshold separates them: the information needed is shape, not intensity. This was the largest single source of error in the first version of the pipeline, which recovered only 28–32 of the 35 pieces.

The classical answer is a watershed on the distance transform, and that is what `split_touching` implements:

1. `distance_transform` computes the *exact* Euclidean distance from every foreground pixel to the background, as two 1-D passes of the separable lower-envelope algorithm (each row's transform is the lower envelope of the parabolas `(q − v)² + f(v)`, built in one sweep). It is exact, unlike the chamfer approximations usually taught beside it.
2. A component whose area is at least 1.45× the expected piece area is assumed to hold `round(area / target)` pieces. Its distance map is smoothed and thresholded, sweeping the level from high to low, and the **highest** level at which exactly that many substantial cores survive is kept — a high level retains only the thick middle of each piece and drops the narrow isthmus where two pieces touch. The smoothing matters: the tip of a tab is a local maximum of the raw distance map too, and without it the sweep counts tabs as pieces.
3. Those cores are grown back over the blob at equal speed, each iteration letting every label claim the still-unassigned neighbours of the pixels it owns, so they meet on the geodesic influence-zone boundary — which for a pair of pieces joined at a narrow contact is exactly the join.

The expected piece area defaults to the median component area, which is the correct estimate whenever most pieces are already isolated. On the dataset photographs this single stage lifts segmentation recall from **0.81 to 0.97**.

### 4.5 Estimating "the area of one piece" robustly

Both the watershed above and the area filter that follows it are expressed relative to an estimate of how large one puzzle piece is, so that no pixel count is hard-coded. That estimate was originally the plain median of the component areas, and it has a failure mode that cost 32 of 35 pieces on one photograph.

The watershed occasionally sheds a **swarm of slivers** along its influence-zone boundary — dozens of components of a few hundred pixels each. A median counts components, so once the slivers outnumber the pieces the median follows the slivers: on `0718-30` it fell from ~7700 px to ~4700 px, the `[0.45, 1.7] × reference` window then straddled the gap between slivers and pieces, and `filter_components` kept **3 components out of 35**.

The obvious repair — an *area-weighted* median, which slivers cannot move because they carry no area — fails in the opposite direction: a few unresolved blobs of four merged pieces carry a great deal of area and drag the estimate up, so the same window then deletes the real pieces. Neither statistic is safe alone.

`segmentation.reference_area` therefore cuts the debris first and takes a plain median of what survives, with the cut anchored on the **upper** end of the distribution (10 % of the 90th percentile) rather than on the median — because the median is precisely the statistic the debris has already corrupted. A percentile rather than the maximum keeps one enormous blob from setting the scale. It reduces to the plain median whenever every component is already about the same size, which is the case on synthetic puzzles and on clean photographs.

Across the 50 photographs this removes the catastrophic failures: the worst case goes from **3 pieces recovered to 35**, and 48 of 50 photographs now land within ±2 of the true count.

### 4.6 Boundary tracing

`contour_extraction.trace_boundary` is **Moore-neighbour tracing with Jacob's stopping criterion**. From the current boundary pixel we walk clockwise around its 8-neighbourhood, starting from the background pixel we came from, until the next foreground pixel is found; that pixel becomes the new boundary point. The stopping rule is *"stop when the start pixel is entered a second time from the same direction"*, not the naive *"stop when you reach the start"* — the naive rule truncates contours that legitimately pass through the start pixel twice, which is common on the thin neck of a jigsaw tab. Getting this wrong was in fact the first bug found during development: the tracer wandered until it hit its step limit and returned contours of 100 000 points for a 180×211 piece.

### 4.7 Piece extraction

`extract_pieces` crops every labelled component to its bounding box (with a small margin so the traced contour never touches the crop border) and stores, in a `Piece`: the RGB crop with background zeroed, the boolean mask, the traced contour in crop coordinates, the centroid, the piece area, and a **normalised orientation** — the rotation of the piece's minimum-area rectangle, computed by `min_area_rect`, a rotating-calipers search over the edges of the convex hull (Andrew's monotone chain). `normalize_piece` uses that angle to rotate a piece upright with bilinear inverse warping.

---

## 5. Task 4 — Piece-edge description

`src/piece_description.py`. Corner localisation is the make-or-break step of the whole project: every side profile and every colour strip is measured *between* two corners, so a corner that is a few percent off shifts an entire descriptor and the matcher pays for it. Three strategies are tried in order.

**1. Body-edge model (the normal path).** A jigsaw piece is a rectangle plus bumps, so its four straight body edges are mutually parallel or perpendicular and share one direction *modulo 90°*. `dominant_orientation` recovers it as

> θ = arg( Σᵢ exp(4i·φᵢ) ) / 4

over the tangent angles φᵢ: mapping every angle to `exp(4iφ)` folds the four edge directions onto the same phasor, while a circular tab arc sweeps all directions uniformly and cancels itself out. The contour is rotated into that frame, where the body edges become axis-aligned, and each edge is located as the **mode** of the perpendicular coordinate of the boundary points whose tangent runs along it. The mode matters: the flat part of an edge dumps a tall one-pixel-wide spike into the histogram, whereas a tab arc spreads its mass over its whole radius, so the mode finds the edge even where the arc contributes more points in total — which a median would not. Both edges of an axis are taken as the two strongest peaks of the *same* histogram, separated by at least 35 % of the range; an earlier version split the points at the centroid instead and failed on pieces with a large tab, because the tab drags the centroid across the true midline. The corners are the four line intersections, snapped onto the contour, and a snap further than 8 % of the body rejects the model outright.

**2. Curvature search (fallback).** The contour is resampled and the turning angle between the chords to the neighbours ±k samples away is measured; convex local maxima are corner candidates, and the best four are chosen by a small combinatorial search that rewards a large enclosed quadrilateral with near-right angles.

**3. Minimum-area rectangle (last resort), snapped onto the contour.**

Why curvature is *not* tried first is worth recording, because it is counter-intuitive: the tip of a tab of radius ρ turns through 90° over an arc of only ≈1.6ρ pixels, which is **sharper** than the piece's own corner at any scale large enough to be noise-free. A curvature search really does mistake tab tips for corners, and it did so on roughly a third of pieces before the body-edge model was introduced. With the model in place, the corner quadrilateral is within 60°–120° at every vertex and within a 0.6 side-length ratio for **more than 90 %** of pieces, and the pieces that fall through to a fallback cost one descriptor, not the reconstruction.

Each of the four sides is then characterised:

* **Type.** The chord between the two corners is the reference line; `profile[i]` is the perpendicular distance of sample `i` from that chord, signed so that positive means outward (away from the piece centroid), divided by the chord length so pieces of different sizes are comparable. If the peak absolute deviation is below `flat_tol = 0.055` the side is a **flat**; otherwise the sign of the extreme decides **tab** (outward) or **blank** (inward). The threshold sits an order of magnitude below the ≈0.30 amplitude of a real tab and comfortably above the rasterisation noise of a straight cut; measured amplitudes on a 4×5 puzzle are 0.010–0.018 for flats and 0.27–0.38 for tabs and blanks, a gap of more than 15×.
* **Shape profile.** The same normalised deviation, resampled at equal arc length to a fixed number of samples (96 by default).
* **Colour signature.** RGB sampled at three depths just inside the side, offset along the **chord** normal rather than the local tangent normal. That choice matters: in the throat of a blank the local normal runs almost along the boundary, so stepping along it lands on the background and the strip fills up with black — this was the second major bug found in development, and it made the colour term worthless until it was fixed. The chord normal always points from the cut line into the body of the piece, and because two mating sides have exactly opposite chord normals it samples corresponding material on both. Validity is tested against the mask **eroded by two pixels** (the outermost ring of a segmented piece is anti-aliased, and the morphological closing can pull in another pixel of genuine background); a sample that is still invalid is retried at smaller depths and finally snapped to the nearest safely-interior pixel.

### 5.1 Two whole-puzzle passes

The pieces of one puzzle are **not independent samples** — they were cut from one grid with one tool — and two passes exploit that.

**`refine_descriptions`.** Corner detection works on one piece at a time and, on a handful of pieces, still lands on a skewed quadrilateral. But the population knows something no single piece does: they share a body size. The pass takes the median body size over the pieces whose own estimate is self-consistent, flags those that disagree by more than 10 %, and re-fits them with `fit_body_rectangle`, which rotates the mask into the body frame and finds the rectangle of the agreed size covering the most piece area (both orientations are tried, since a piece may be lying on its side). Because a tab only adds area outside the body while a blank only removes area inside it, that maximum-coverage window sits on the body. A repair is kept only if it covers more of the piece than the original. Without it, one skewed piece produces a skewed descriptor *and* a visibly warped piece in the rendered reconstruction.

**`calibrate_side_types`.** A fixed `flat_tol` has to hold across resolutions, camera angles and how crisply a puzzle is cut, and it does not: on a dataset photograph the fixed 5.5 % default found **14** flat sides where the 5×7 grid demands 24. Two facts pin the boundary down instead — the amplitudes are strongly bimodal (flats near zero, tabs and blanks near 0.3), and an `R × C` grid exposes exactly `2(R + C)` flat sides. Every gap between consecutive sorted amplitudes is a candidate threshold; the widest gap whose flat count matches `2(R + C)` for some factor pair of `N` wins, with the widest gap overall as the fallback when segmentation lost pieces and no count can match. On the same photograph this recovers **24 of 24**, and 4 corner pieces of 4.

Classification is verified end-to-end: on a 3×4 puzzle the library finds exactly 4 corner pieces (two adjacent flats), 6 edge pieces (one flat) and 2 interior pieces (no flats), which is precisely what a 3×4 grid must contain, and `infer_grid_shape` reproduces `(3, 4)` from those counts alone.

---

## 6. Task 5 — Piece-edge matching

`src/edge_matching.py`. The compatibility of an ordered pair of sides `(a, b)` is a single number, lower being better.

### 6.1 The exact formula

> **D(a, b) = w_s · D_shape(a, b) + w_c · D_colour(a, b) + w_l · D_length(a, b)**

with `D(a, b) = +∞` for inadmissible pairs.

**Admissibility — "does the shape fit".** These are hard constraints, not penalties:

* a **flat** side is a border of the whole puzzle, so it can never form an interior seam — any pair involving a flat is inadmissible;
* a **tab** must meet a **blank**: `tab–tab` and `blank–blank` are inadmissible;
* a side never matches another side of the same piece.

**Shape — "does the outline continue".** When two pieces are placed next to each other, one side is traversed in the opposite direction to the other, and their outward normals point in opposite directions. A perfect fit therefore satisfies `p_a(t) = −p_b(1−t)`, and the RMS residual of that identity is the shape cost:

> D_shape(a, b) = √( (1/M) · Σᵢ ( p_a(tᵢ) + p_b(1−tᵢ) )² )

Because both profiles are normalised by their own chord length, the term is dimensionless and independent of piece size. A length-consistency term penalises pairing sides of visibly different physical size:

> D_length(a, b) = |L_a − L_b| / max(L_a, L_b)

**Colour — "does the picture line up".** Each side carries a strip of colours `C(t, d)` sampled at `D = 3` depths just inside the piece. With the same reversal, the cost is the sum of squared colour differences, divided by the number of terms so it stays in `[0, 1]`:

> D_colour(a, b) = √( (1/3MD) · Σᵢ,d ‖ C_a(tᵢ, d) − C_b(1−tᵢ, d) ‖² )

**Shift tolerance.** Corner localisation is accurate to a few pixels, and a corner found a little early on one piece and a little late on its neighbour shifts the whole profile. Both `D_shape` and `D_colour` are therefore **minimised over integer sample shifts of up to 5 % of the side**, evaluated on the overlapping part. The whole `(4N)²` table is computed with matrix algebra — expanding `‖u − v‖² = ‖u‖² − 2u·v + ‖v‖²` turns each shift into one Gram matrix product — so a 35-piece puzzle's 19 600 pairs are scored in well under a second.

**Illumination normalisation** (`colour_norm="meanstd"`). The synthetic puzzles are lit uniformly by construction; the dataset photographs are not. The pieces lie all over a table, each under slightly different light and seen from a slightly different angle, so two strips facing each other across a *true* seam differ by a colour offset and a gain even where the picture continues perfectly. Subtracting each strip's own mean and dividing by its own spread removes both and compares only the pattern along the strip. Measured on the photographs, it raises the matcher's top-1 rate from 0.193 to 0.251 and the precision of its best-buddy pairs from 0.264 to 0.400; on synthetic puzzles it is neutral (top-1 0.918 vs 0.917). It is therefore off by default and enabled for the dataset.

### 6.2 How much weight goes to shape versus colour

**The defaults are `w_s = 1.0`, `w_c = 1.0`, `w_l = 0.5`** — shape and colour contribute equally, and length is a half-weight tie breaker.

The justification is measured, not assumed. On the 4×5 reference puzzle (`results/evaluation_results/matching_study.json`), averaged over the 62 true seams versus all admissible pairs:

| Term | True seams | All admissible pairs | Separation |
|---|---|---|---|
| D_shape | 0.0235 | 0.0566 | 2.4× |
| D_colour | 0.0581 | 0.1939 | 3.3× |
| D_length | 0.0050 | 0.0054 | 1.1× |

Both shape and colour are on the same order of magnitude for true and for random pairs, so **neither needs rescaling to be heard** — that is what makes equal weights the natural choice rather than an arbitrary one. `D_length` is essentially constant within a puzzle (all pieces are cut to the same size), which is why it carries a *half* weight: it contributes nothing on clean data and acts purely as a safety net when segmentation produces a size outlier, which does happen on the real photographs.

A sweep over eight weight combinations on three puzzles (`results/evaluation_results/weight_study.json`) shows the two terms are complementary rather than redundant:

| w_shape | w_colour | w_length | Top-1 match rate | Neighbour acc. | Position acc. |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0.078 | 0.483 | 0.476 |
| 0 | 1 | 0 | 0.880 | 1.000 | 1.000 |
| 1 | 1 | 0 | **0.900** | 1.000 | 1.000 |
| **1** | **1** | **0.5** | 0.890 | 1.000 | 1.000 |
| 2 | 1 | 0.5 | 0.893 | 1.000 | 1.000 |
| 1 | 2 | 0.5 | 0.894 | 1.000 | 1.000 |
| 1 | 0.5 | 0.5 | 0.893 | 1.000 | 1.000 |

Read honestly, this says three things. **Shape alone is weak** on a machine-cut puzzle — a top-1 match rate of 0.078 — because every tab is die-cut to nearly the same silhouette and the geometry barely distinguishes them; it is a real but weak signal (2.4× separation) rather than a decisive one. **Colour alone is strong** (0.880). **Adding shape to colour still helps**, lifting top-1 from 0.880 to 0.900, because shape disambiguates exactly the cases colour cannot: seams crossing a uniformly coloured region. Every configuration that includes the colour term reconstructs all three puzzles perfectly, so the choice among them is not critical for the final result — but the combination gives the best matcher, and a better matcher is what buys margin on harder inputs. The differences among the weighted variants are within run-to-run noise, so the defaults are chosen as the simplest ones inside that band.

`best_buddies` additionally reports the side pairs that are each other's *mutual* cheapest partner — 30 such pairs on the 4×5 puzzle. They are the highest-confidence evidence available before any search, and the assembly stage uses them.

---

## 7. Task 6 — Assembly algorithm

`src/assembly.py`. The compatibility table is coupled to a greedy best-first search over a fixed `R × C` grid. When the grid size is not supplied, `infer_grid_shape` deduces it: an `R × C` puzzle has `R·C` pieces of which `2(R + C) − 4` are border pieces, so `R + C` follows from the observed border count and `R·C = N` pins the pair down.

**Seeding.** The grid is anchored with a *corner* piece — two adjacent flat sides — placed at cell `(0, 0)` and rotated so that its two flats face North and West. If no corner piece exists (a puzzle without a border, or a failed flat classification), the seed is the piece of the most confident seam, placed at the centre so the grid can grow in every direction.

**Iteration.** Every empty cell touching at least one placed piece is a frontier candidate. For each such cell, every unused piece and each of its four rotations is scored by the total edge dissimilarity to the already-placed neighbours, normalised by the number of matched neighbours, and rejected outright if a hard constraint is violated: a cell on the outer boundary of the grid must show a flat on that side, an interior seam may not show a flat, and every seam must be an admissible tab/blank pair. **This is where each piece's rotation is resolved** — the rotation is part of the candidate, chosen by the same cost.

### 7.1 Tie-breaking rule

Candidates are compared on the tuple

> `(not a best-buddy seam, mean seam cost, −number of matched neighbours, −confidence margin, piece index, rotation)`

in that order.

*Best-buddy seams first.* If the two sides of a seam are each other's cheapest partner in the entire puzzle, that mutual first choice is far stronger evidence than a merely cheap cost, and committing such seams first stops the walk from spending an irreplaceable piece on a locally attractive but wrong cell. This single rule was the largest single improvement to reconstruction accuracy in development.

*Then mean seam cost* — cheapest first, the basic greedy criterion.

*Then more matched neighbours* — a placement constrained by three already-placed neighbours rests on three times the evidence of one constrained by a single neighbour.

*Then a larger confidence margin* — the gap to the second-best candidate for the same cell. A placement that is uniquely good is safer than one where two pieces score almost the same.

*Then piece index and rotation*, purely so that the result is deterministic and reproducible.

### 7.2 Handling of dead ends and unplaceable pieces

The search never aborts. If no candidate satisfies the hard constraints, they are relaxed in three documented stages:

1. **drop the border/flat requirement**, charging `BORDER_PENALTY = 5.0` per violated side;
2. **allow illegal `tab–tab` and `blank–blank` seams**, charging `DEADEND_PENALTY = 10.0` per seam;
3. **place whatever is left in reading order.**

Both penalties are set far above any legitimate seam cost (which is ≈0.1–0.5), so a forced placement is never preferred to a legal one. Every placement made through a relaxed stage is flagged `forced=True` and counted in `Assembly.n_forced`, so the caller can always tell how much of the arrangement is trustworthy. A test drives this path deliberately, replacing the whole cost table with `+∞`, and confirms that all pieces are still placed exactly once.

### 7.3 A supplied grid is authoritative

When the caller states the grid — and for the dataset it is known to be 5×7 — that layout is used verbatim even if the piece count disagrees with it. On a real photograph segmentation may recover 34 or 36 pieces of a 35-piece puzzle, and honouring the known layout (leaving one cell empty, or one spare piece unplaced, and saying so in the log) is far more useful than silently re-deriving a 2×17 grid from the wrong count, which is what an earlier version did. `infer_grid_shape` is used only when no grid is supplied.

### 7.4 The best-arrangement guarantee

One greedy walk is only as good as its first few decisions. The pass is therefore repeated from **every candidate seed** — every corner piece in every orientation that puts its two flats on the outside — and the best arrangement obtained is returned, ranked by

> `(pieces placed, forced placements, mean seam cost)`

A better result is never discarded, so `assemble` always returns the best arrangement it obtained, complete or not. A test asserts that running with eight restarts never yields a worse arrangement than running with one.

### 7.5 Rendering

Once the arrangement is known, `render_assembly` draws it back into a single image. Each piece is mapped onto its grid cell with the **similarity transform** (rotation + uniform scale + translation) that best sends its four corners onto the four corners of the cell. Two correspondences already determine a similarity exactly, and an earlier version used only the two corners of the North-facing side — but then a single mis-located corner carried half the leverage and visibly skewed the piece in the output. The least-squares fit over all four reduces that to a quarter and averages the corner noise out. Because the cut is complementary, the tabs land precisely in the neighbours' blanks and the seams close without any blending trick. A final cosmetic pass fills the hairline cracks that survive where two neighbouring pieces were scaled from slightly different corner estimates; it never invents content where a whole piece is missing.

### 7.6 The border rule as a filter or as a penalty

Stage (i) of §7.2 encodes a genuine fact about a rectangular jigsaw: a cell on the rim of the grid must present a flat side outwards, and an interior cell must not. Enforcing it as a **hard filter** is a large saving — it removes most candidates before their seams are ever scored — and it is exactly right whenever the flat/tab/blank classification can be trusted.

On the dataset photographs it cannot. §9.3 measures the classifier at **79.6 %** correct on a piece's flat count, with only **52 %** of the true corner pieces recognised as corners, against ~100 % on synthetic puzzles. Under that error rate a hard filter is actively harmful, and asymmetrically so: a *missed* flat makes the true border piece inadmissible at every border cell, so the correct placement is not merely ranked low, it is unavailable, and the search must put something else there. Rejecting a correct candidate costs the whole downstream walk, while admitting a wrong one only costs its own seam.

`assemble(border_mode=...)` therefore selects between the two readings:

* `"hard"` (the default) — stage 0 filters, as above. Correct for clean input, and the setting under which the synthetic results of §8.3 are obtained.
* `"soft"` — stage 0 is skipped, so a border violation is only ever charged `BORDER_PENALTY` and can be outvoted by strong seam evidence. This is what `DATASET_SOLVER` selects.

The distinction is one of *confidence in a constraint*, not of the constraint's truth: the rule is still believed, and still charged for, but it is no longer allowed to veto evidence that comes from a more reliable measurement. Measured over the 50 photographs it lifts neighbour accuracy 0.192 → 0.220, and it leaves the synthetic puzzles at 100 %, so it is not a trade.

A consequence worth recording: a placement is now counted as *forced* when it actually had to break a rule — when it was charged `BORDER_PENALTY` or `DEADEND_PENALTY` — rather than merely because it was scored in a relaxed stage. Under `"soft"` every stage is relaxed, so the old stage-based test would have reported every placement as forced and made the restart ranking of §7.4 meaningless.

---

## 8. Reconstruction quality, and validation on clean input

`src/evaluation.py` provides two families of measure. This section defines them and validates the assembly stage on synthetic puzzles, where the input is clean and the answer is known by construction; §9 then applies the same measures to the real photographs.

### 8.1 Reference-free — what the end-to-end routine returns

> `quality = 1 − mean(seam cost paid) / mean(cost of all admissible seams)`, clipped to `[0, 1]`

The denominator is what an arrangement that paired sides at random would pay on average, so the score answers *how much better than chance is this arrangement?* It requires no ground truth and is therefore available for every input, including the dataset photographs. A perfect 4×5 reconstruction scores 0.51 with a mean seam cost of 0.135; the same pieces shuffled at random score near 0. Also reported: the number of seams, how many of them are illegal, the median and maximum seam cost, and the number of forced placements.

### 8.2 Ground truth, and where it comes from

The dataset photographs come with no answer key — nobody labels which cell of the finished picture each photographed piece belongs to. `evaluation.py` therefore also contains a synthetic puzzle **generator**. A piece is defined as its rectangular body ∪ its own tab circles ∖ its neighbours' blank circles; because the very same circle is added to one piece and subtracted from the other, mating sides are **exactly complementary**, which is what a real die-cut jigsaw does. Every cut gets its own radius, neck depth and along-edge offset, so no two seams share a silhouette — without that variation the shape term would carry no information whatsoever, since every tab would fit every blank equally well. (This was the third significant finding of development: the first generator cut identical circular tabs, and the shape term measured 0.023 for true seams and 0.024 for random ones — perfectly useless.) The generator then shuffles the pieces, rotates them by arbitrary angles, and lays them out on a canvas with enough spacing that they never touch.

That makes the following measurable:

* **`direct_accuracy`** — the fraction of pieces in the correct cell, and the fraction correct in both cell *and* rotation, maximised over the four global quarter-turns of the grid, because nothing in a bag of pieces says which way is up and a puzzle solved sideways is still solved;
* **`neighbour_accuracy`** — the fraction of true adjacencies reproduced. Invariant to any global transform, which makes it the fairest single number for a jigsaw solver;
* **`rotation_accuracy`** — the fraction of pieces whose orientation relative to the grid is correct, allowing one global quarter-turn shared by all pieces;
* **`matching_accuracy`** — how often the compatibility measure *alone*, before any search, ranks the true partner side first. This isolates the matcher from the search and is what §6.2 reports;
* **`image_metrics`** — MSE, PSNR and SSIM against the original picture, with SSIM implemented from scratch on the library's own Gaussian filter, and the four global quarter-turns tried so that a correctly-but-sideways solved puzzle is not scored as a failure.

### 8.3 Validation results (synthetic input)

Seven synthetic puzzles, six with pieces at arbitrary rotations (`python main.py --run validate`; full table in `results/evaluation_results/benchmark.json`):

| Puzzle | Pieces | Rotated | Pieces found | Neighbour acc. | Position acc. | Rotation acc. | Quality | SSIM | Time |
|---|---|---|---|---|---|---|---|---|---|
| 2×3 | 6 | yes | 6/6 | 1.00 | 1.00 | 1.00 | 0.47 | 0.81 | 1.7 s |
| 3×4 | 12 | yes | 12/12 | 1.00 | 1.00 | 1.00 | 0.47 | 0.76 | 5.2 s |
| 3×4 | 12 | yes | 12/12 | 1.00 | 1.00 | 1.00 | 0.47 | 0.80 | 5.0 s |
| 4×5 | 20 | yes | 20/20 | 1.00 | 1.00 | 1.00 | 0.51 | 0.79 | 12.5 s |
| 4×6 | 24 | yes | 24/24 | 0.79 | 0.92 | 0.92 | 0.44 | 0.77 | 17.7 s |
| 5×7 | 35 | yes | 35/35 | 1.00 | 1.00 | 1.00 | 0.48 | 0.79 | 40.2 s |
| 5×7 | 35 | no | 35/35 | 1.00 | 1.00 | 1.00 | 0.53 | 0.80 | 35.5 s |

**Six of seven puzzles are reconstructed perfectly**, including both 35-piece puzzles — the size of the real dataset's jigsaw. Mean neighbour accuracy is **0.970** and mean position accuracy **0.989**. Comparing the rotated and unrotated 35-piece runs shows that **arbitrary orientation costs essentially nothing**: the rotation is resolved during placement, and the only measurable difference is a slightly lower reference-free quality (0.48 vs 0.53), reflecting the small extra descriptor noise that resampling a rotated piece introduces.

The one imperfect run (4×6) still places 92 % of pieces correctly; it fails on a cluster of pieces whose picture content is nearly uniform, where the colour term is uninformative and the shape term alone is not decisive — exactly the regime §6.2 predicts.

The SSIM figures near 0.8 on *perfect* reconstructions are a resampling artefact, not a placement error: the reconstruction is drawn on a grid whose cell size is the *measured* median piece body, which differs from the original cell size by a pixel or two, so the two images drift out of pixel registration across the canvas.

**Scaling.** Run time grows roughly with the square of the piece count (1.7 s at 6 pieces, 12.5 s at 20, 40 s at 35). Two costs dominate: scoring all `(4N)²` side pairs, and re-running the greedy pass once per candidate seed. Accuracy does not degrade with size in this range — the 35-piece puzzles are solved perfectly — because the extra pieces also bring extra constraints, and the border/flat rules become more restrictive as the perimeter grows.

---

## 9. Results on the provided dataset

The `detection/` dataset is the YOLO export of [`hobby-projs/puzzle-vrkx6-9xh3l`](https://universe.roboflow.com/hobby-projs/puzzle-vrkx6-9xh3l/dataset/1) on Roboflow Universe: photographs of the 35 pieces of one jigsaw on a dark cloth, each piece annotated with its **identity** (1–35) and a bounding box. Most images show a single piece, but **43 training and 7 validation images show all 35 pieces scattered** — those 50 are the real scrambled puzzles this milestone targets, and `python main.py` runs the whole pipeline on every one of them.

### 9.1 Recovering an answer key from the annotations

The annotations label identity, not position, so at first sight reconstruction cannot be scored on real data at all. It can: the ids turn out to be the **row-major positions of the finished 5×7 puzzle**, id `k` sitting at `((k−1) // 7, (k−1) % 7)`.

The evidence is the flat sides, and it does not presuppose any reconstruction. Counting flats per id across photographs and laying the result out row-major gives

```
   1:2.0   2:0.0   3:1.0   4:1.0   5:2.0   6:1.0   7:1.0
   8:1.0   9:0.0  10:0.0  11:0.0  12:0.0  13:0.0  14:1.0
  15:1.0  16:0.0  17:0.0  18:0.0  19:0.0  20:0.0  21:1.0
  22:1.0  23:0.0  24:0.0  25:0.0  26:0.0  27:0.0  28:1.0
  29:1.0  30:0.0  31:1.0  32:1.0  33:0.0  34:1.0  35:2.0
```

Every one of the **fifteen** ids the hypothesis calls interior shows zero flats, and the ids it calls border show at least one, with the two detected corners at the ends of the first and last rows. Agreement is 29 of 35 ids, and every disagreement is a border id occasionally *missing* a flat — the failure mode flat detection actually has — never an interior id gaining one, which is what a wrong layout would produce. `main.dataset_true_cells` uses that mapping, so §9.3 reports reconstruction accuracy measured on the real photographs.

### 9.2 Segmentation and description

Scored against the annotated boxes, over all **50** full-scramble photographs (`python main.py`; per-image rows in `results/evaluation_results/dataset_study.json`):

| Measure | Value |
|---|---|
| Piece recall (annotated pieces isolated) | **0.973** |
| Piece precision (components that are pieces) | **0.963** |
| F1 | 0.965 |
| Images where every piece was found (recall = 1.00) | **41 / 50** |
| Pieces isolated | 35.1 / 35 on average |
| Flat sides found | **23.1** vs the 24 a 5×7 grid must expose |
| Corner pieces found | 3.2 / 4 |
| Time per photograph | 32 s at native 1920×1080 |

**Splitting touching pieces (§4.4) is what makes this work.** Without it, pieces in contact merge into one component and recall sits at **0.81**; with it, **0.94** here, and 0.97 in a segmentation-only study at the reduced 1280 px working size. Working at native resolution matters too: at 1280 px the piece body is ~75 px and the descriptors are measurably noisier than at 1920 px, where it is ~114 px and the flat/tab amplitude histogram separates cleanly.

**Description is essentially correct on real photographs.** Averaged over the 50 images the library finds 23.1 flat sides against the 24 a 5×7 grid must expose, and 3.2 corner pieces of 4; on a representative photograph it finds exactly 24 and exactly 4, using the calibrated threshold of §5.1 where the fixed default found 14 and 0. So the pieces, their corners, their four sides and their types are all recovered from the real data.

### 9.3 Reconstruction

The same 50 photographs, scored against the answer key of §9.1:

| Measure | Real photographs | Synthetic, for contrast (§8.3) |
|---|---|---|
| Neighbour accuracy | **0.220** | 0.970 |
| Position accuracy | **0.131** | 0.989 |
| Matcher top-1 | 0.324 | 0.90 |
| Best single image | 0.328 neighbour | 1.000 |

**The reconstruction does not succeed on this dataset.** Roughly one adjacency in five is correct, against about one in twenty by chance — a real signal, but nowhere near a solved puzzle. That is the honest result, and its cause is identifiable rather than mysterious.

Those figures use the photograph preset (`main.DATASET_SOLVER`). It differs from the library defaults in exactly two settings, each measured over all 50 answer-keyed photographs from cached descriptors so that only the stage under test varies:

| Configuration | Matcher top-1 | Neighbour acc. | Position acc. |
|---|---|---|---|
| colour SSD, hard border rule | 0.267 | 0.189 | 0.119 |
| **+ MGC** photometric term (§6.3) | 0.324 | 0.192 | 0.132 |
| **+ soft border rule** (§7.6) | 0.324 | **0.220** | **0.131** |

Both are real but small: +16 % neighbour accuracy, better on 26 of the 50 photographs and worse on 18. The synthetic puzzles stay at 100 % under the same settings, so neither trades clean-input accuracy for noisy-input accuracy.

**Why it is still only 0.22.** There are two distinct limits, and only the second turned out to be fixable.

*The compatibility measure* is the first and the larger. Asking of every side whether its cheapest partner lies on a genuinely adjacent piece — a measure that needs no knowledge of any piece's rotation, so it isolates the matcher — gives a top-1 rate of **0.324** against **0.10** for chance, where the same measure reaches **0.90** on synthetic puzzles. Everything tried against it is recorded in `results/evaluation_results/`:

| Attempt | Varied | Outcome |
|---|---|---|
| More search | restarts 1 to 60 | saturates at 4 |
| Richer colour descriptor | 7 sampling-depth variants | best lifts best-buddy precision 0.52 → 0.57 but *lowers* reconstruction |
| Wider alignment search | shift 0 % to 20 % of a side | the existing 5 % is the optimum; both directions worse |
| Per-side cost normalisation | each side's row divided by its own 2nd-best / low quantile / z-score | top-1 0.343 → 0.381, reconstruction flat |
| Beam search | width 20–150, top-3 to top-5 | no gain, and a regression on clean input |
| Cluster merging (Kruskal-style) | commit the cheapest merges globally rather than growing from a seed | *worse* (0.173 vs 0.213) — the seeded greedy is not the weak link |
| Removing non-puzzle objects | oracle: keep only the pieces matched to an annotation | **no change at all** (0.213 → 0.214) |
| **MGC** (gradient continuity, §6.3) | Gallagher's Mahalanobis Gradient Compatibility | matcher +21 % (top-1 0.267 → 0.324); position accuracy +11 % |

The pattern is consistent, and it is what establishes the diagnosis: every change that improves the *matcher* leaves the *reconstruction* nearly where it was. At a top-1 of ~0.3 no search recovers a 35-piece puzzle, and two of these rows rule out the obvious alternative explanations directly — a fundamentally different search strategy did worse, and *perfectly* removing every non-puzzle object from the frame changed nothing.

*The border rule* was the second limit, and unlike the first it was a defect rather than a property of the data. The flat/tab/blank classifier gets a piece's flat count right on only **79.6 %** of pieces here (against ~100 % on synthetic puzzles), and only **52 %** of the true corner pieces are recognised as corners. The assembler nevertheless treated "a rim cell must show a flat outwards" as a *hard* filter, so on these photographs it rejected correct placements more often than it prevented wrong ones. Charging `BORDER_PENALTY` instead (§7.6) produced the 0.192 → 0.220 step above at no cost on clean input.

Three properties of this particular puzzle account for it, and all three are properties of the data rather than of the algorithms:

1. **The picture is almost entirely white.** The HIWIN advert is white and pale grey over most of its area, so the colour strips along most seams are nearly identical. §6.2 measured that colour carries the great majority of the discriminative power on a *colourful* puzzle (top-1 0.880 for colour alone versus 0.078 for shape alone); on a white puzzle that dominant term has almost nothing to work with.
2. **The tabs are machine-cut and nearly identical**, so the shape term — already the weaker of the two by an order of magnitude — cannot compensate.
3. **Each piece is photographed under its own lighting and viewing angle.** Illumination normalisation (§6) recovers part of this, lifting top-1 from 0.193 to 0.251, but perspective differences across the frame remain.

The pipeline therefore segments, describes and assembles the dataset's pieces correctly, while reconstructing *this particular puzzle* from *these particular photographs* is beyond what a shape-plus-colour compatibility measure supports. The same code reconstructs a 35-piece puzzle perfectly when the pieces carry distinguishable picture content (§8.3), which places the limitation in the input rather than in the method.

### 9.4 What would close the gap

* **A discriminative descriptor — but not a two-dimensional one.** The obvious classical candidate, gradient continuity (Gallagher's MGC, §6.3), has been implemented and measured: it helps the matcher by a fifth and the reconstruction slightly. The natural next step would be a descriptor carrying more than a one-dimensional strip — patch-based correlation over a two-dimensional band either side of the seam, using the printed texture's spatial structure rather than a single line of it. **That has now been measured too, and it does not work**: widening the strip along the side and deepening it into the piece both make the matcher *worse*, monotonically.

  | Seam sampling | MGC top-1 | Best-buddy precision |
  |---|---|---|
  | 96 samples × 3 depths (the default) | **0.335** | **0.531** |
  | 160 × 3 | 0.331 | 0.499 |
  | 224 × 3 | 0.334 | 0.506 |
  | 96 × 6 (2-D band) | 0.317 | 0.479 |
  | 160 × 6 | 0.328 | 0.492 |
  | 160 × 8 | 0.328 | 0.492 |

  The interpretation is physical rather than algorithmic. The printed detail that a patch descriptor would exploit is 1–2 px wide at this resolution, and the two halves of a true seam were photographed at different angles and distances across the table, so they cannot be registered to that precision. Sampling more of the seam therefore adds misregistered detail, i.e. noise, faster than it adds signal. A patch descriptor would need the photographs rectified first, which is the next item.
* **Photometric calibration.** Estimating and dividing out the illumination field across the table before describing pieces would remove the per-piece gain that normalisation only approximates.
* **A stronger search**, but only *after* the measure improves. Beam search was implemented and measured here and did not help (see above); loopy belief propagation or hierarchical merging would tolerate a weaker measure better, but the evidence is that no search recovers information the measure never captured.

## 10. Testing

The suite is 144 tests, one file per stage, run either with pytest or with the bundled pytest-free runner (`python tests/run_tests.py`).

| File | What it pins down |
|---|---|
| `test_enhancement.py` | convolution against a hand-computed sum; separable blur equals 2-D; the median deletes impulses where the Gaussian smears them; histograms count every pixel; equalisation flattens; stretching spans the range |
| `test_thresholding.py` | Otsu really maximises between-class variance (brute-force check); adaptive mean equals the local box mean; adaptive beats global under an illumination ramp |
| `test_edge_detection.py` | kernel algebra; a vertical edge gives a horizontal gradient; NMS thins a ridge to its crest; hysteresis keeps connected weak pixels and drops isolated ones; Canny outlines a rectangle without filling it |
| `test_segmentation.py` | erosion/dilation sizes on a square; `majority_smooth` is self-dual; hole filling; 4- vs 8-connectivity on a diagonal; component statistics; the distance transform against brute force; the watershed splits two joined discs and leaves separate ones alone; Moore tracing walks a square exactly once; convex hull and minimum-area rectangle on a rotated rectangle; `reference_area` survives both a swarm of watershed slivers and a few oversized blobs, where a plain median follows the slivers (§4.5) |
| `test_piece_description.py` | dominant orientation straightens a rectangle at any angle; corners form a usable quadrilateral always and an accurate one for ≥90 % of pieces; tab/blank/flat classification on analytic profiles; profiles are scale-invariant; a 3×4 puzzle yields exactly 4 + 6 + 2 pieces by flat count |
| `test_edge_matching.py` | the admissibility truth table; the shape distance is zero for a perfect fit; shift tolerance absorbs a misalignment; illumination normalisation is invariant to gain and offset; the vectorised table matches the scalar formula entry by entry; the table is symmetric; best buddies really are mutual |
| `test_assembly.py` | the grid fills with each piece used once; flats face outwards; rotated *and* unrotated puzzles reconstruct exactly; the dead-end path still places everything; restarts never return a worse arrangement; the reconstruction resembles the original and not an unrelated picture |

---

## 11. Conclusion

Every requirement of the brief is implemented from scratch and independently testable, and the end-to-end routine accepts a scrambled puzzle, resolves each piece's rotation during placement, and returns the reconstructed image together with a numerical quality score.

On the **provided dataset** the pipeline isolates the pieces reliably — recall 0.97 at precision 0.96 across all 50 full-scramble photographs, with the distance-transform watershed of §4.4 responsible for lifting that from 0.81 — and describes them correctly, recovering all 24 flat sides and all 4 corner pieces that a 5×7 grid must expose. It does **not** reconstruct the picture: the compatibility measure runs at roughly three times chance on these photographs, and §9.3 shows exactly why, on a mostly-white advert photographed piece by piece across a table. On **synthetic puzzles**, where the picture carries distinguishable content, the identical code solves six of seven perfectly, including both 35-piece cases.

Six findings are worth carrying forward.

1. **The descriptor is where the difficulty lives, not the search.** Once corners came from the body-edge model and colour was sampled along the chord normal, a plain greedy search with best-buddy ordering solved every clean puzzle; and where the descriptor is weak, no search recovers it. Two negative results in §9.3 make this concrete rather than rhetorical: replacing the seeded greedy with global cluster merging did *worse*, and removing every non-puzzle object from the frame with an oracle changed nothing at all.
2. **Shape and colour are complementary but far from equal.** On a machine-cut puzzle colour alone reaches a 0.88 top-1 match rate and shape alone 0.08 — so a puzzle whose picture is uniform removes most of the available information, which is precisely the dataset's situation.
3. **Complementarity is fragile.** Opening-then-closing, an asymmetric contour smoother, or a mis-placed corner each destroy the mirror relationship between a tab and its blank, and each silently reduced the shape term to noise until it was found.
4. **The population is more informative than the piece.** Three of the largest improvements — repairing outlier corners against the median body size, calibrating the flat threshold against the `2(R+C)` flats a grid must have, and sizing the watershed split from the median piece area — come from treating the puzzle as one object cut by one tool rather than as a bag of independent samples.

The clearest next step is a descriptor that reads the fine printed texture across a seam instead of its mean colour; §9.4 sets out that and two alternatives.

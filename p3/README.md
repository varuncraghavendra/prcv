# 2D Object Recognition

## Folder structure

```
project_3/
├── main.cpp
├── include/
│   ├── image_processing.hpp
│   ├── object_recognition.hpp
│   └── utilities.hpp
├── src/
│   ├── image_processing.cpp
│   ├── object_recognition.cpp
│   └── utilities.cpp
├── data/
│   ├── train/          ← training crops saved by pressing t
│   ├── test/           ← test crops saved by pressing x
│   ├── database_normal.csv
│   └── database_eigen.csv
├── models/
│   └── eigenspace.yml
├── output/             ← screenshots saved by pressing p
└── README.md
```

---

## Build

```bash
sudo apt update
sudo apt install libopencv-dev pkg-config

g++ -std=c++17 \
    main.cpp src/image_processing.cpp src/object_recognition.cpp src/utilities.cpp \
    $(pkg-config --cflags --libs opencv4) \
    -I./include \
    -o object_recognition

./object_recognition
```

---

## Controls

| Key | Action |
|-----|--------|
| `1` | Threshold view |
| `2` | Cleaned mask view |
| `3` | Region map view (stable colors) |
| `4` | Annotated color view (default) |
| `t` | Save current object as **training** data + rebuild eigenspace |
| `x` | Save current object as **test** data |
| `f` | Toggle Normal ↔ Eigenspace recognition mode |
| `m` | Print confusion matrix for the current mode |
| `p` | Save current view to `output/` |
| `q` | Quit |

---

## Task-by-task implementation notes

### Task 1 — Threshold the input video
Function: `thresholdInputVideo()` in `src/image_processing.cpp`.

- Gaussian blur (7×7) smooths the frame before scoring.
- Each pixel gets a **custom score**: `0.60 × (255 − gray) + 0.40 × saturation`.
  Strongly colored pixels (high saturation) are made darker, moving them away from
  the white background, as required.
- Bright, low-saturation pixels (paper, glare) are suppressed to 0 or heavily reduced.
- The threshold is set **dynamically** using **k-means (k=2)** on a 1/16 sub-sample
  of score values. The midpoint between the two cluster centers is used as the cut-off.
- A glare mask zeroes any pixel with gray > 200 and saturation < 30.

### Task 2 — Cleanup the binary image
Function: `cleanupBinaryImage()` in `src/image_processing.cpp`.
**No OpenCV morphology functions are used — all pixel loops are direct C++.**

**We chose morphological CLOSING (erosion then dilation) rather than erosion alone.**

Why closing and not just shrinking?
- Pure erosion (shrinking) kills noise but also shrinks real objects and opens holes
  inside objects that have thin or poorly-lit regions (e.g. a pen end-on, a shiny coin).
- Closing = erosion with radius r, then dilation with the same radius r:
  1. Erosion destroys isolated small speckles that are smaller than the kernel (noise removed).
  2. Dilation restores the surviving foreground roughly to its original size and **fills
     small interior holes** left by the thresholding step.
- The result is a clean, solid object mask — better for connected-component segmentation
  and more stable oriented bounding boxes and feature values.

### Task 3 — Segment the image into regions
Function: `segmentIntoRegions()` in `src/image_processing.cpp`.

- Uses `connectedComponentsWithStats` (OpenCV).
- Ignores regions smaller than `minArea` (default 1200 px).
- Ignores regions larger than 35% of the image area (likely background).
- Ignores regions touching the image boundary.
- Ignores regions wider or taller than 90% of the frame.
- Remaining candidates are ranked by a score: area − 0.7 × distance-from-center.
  This prefers a large, central object over a large off-center one.
- Regions are renumbered sequentially after filtering.
- **Color-flicker prevention**: a `colorMemory` vector is maintained across frames.
  Each region's centroid is matched to the nearest centroid from the previous frame
  (within 50 px). A matching region keeps the same color; truly new regions get a
  fresh random color. This eliminates the jumping colors seen with a fixed palette.

### Task 4 — Compute features for each major region
Function: `computeFeaturesForRegion()` in `src/object_recognition.cpp`.

Features computed (all region-based, not boundary-based):

| Feature | Invariance |
|---------|-----------|
| `percentFilled` = region area / oriented-box area × 100 | translation, scale, rotation |
| `aspectRatio` = max(w,h) / min(w,h) of oriented box | translation, scale, rotation |
| Hu moments 0–6 (via `cv::HuMoments`) | translation, scale, rotation |

OpenCV `cv::moments()` computes raw moments (m00, m10, m01, …) and central moments
(mu20, mu02, mu11, …). `cv::HuMoments` derives seven non-linear combinations of the
normalized central moments η that are invariant to translation, scale, and rotation.
We use all seven Hu moments as features.

The principal axis angle is computed from: `θ = 0.5 × atan2(2 μ₁₁, μ₂₀ − μ₀₂)`.

**Live feature display (Task 4 extension)**: the annotated view (key `4`) shows
`fill=XX.X%  AR=Y.YY` in the bottom-left corner in real time, so you can rotate and
translate the object and watch the values stay stable.

Drawn overlays:
- Green oriented bounding box (rotates with the object).
- Red principal axis line through the centroid.
- Blue filled circle at the centroid.
- White classification label near the centroid.

### Task 5 — Collect training data
Press **`t`** to save a training example.

1. The system prompts for a label (e.g. `pen`).
2. The oriented-box crop is saved to `data/train/<label>_<N>.png`.
3. The feature vector is appended to `data/database_normal.csv`
   (header row: `label, percent_filled, aspect_ratio, hu0 … hu6`).
4. The eigenspace is rebuilt from all training crops and saved to
   `models/eigenspace.yml`; PCA coefficients are written to `data/database_eigen.csv`.

### Task 6 — Classify new images
Two modes, toggled with **`f`**:

**Normal mode** — scaled Euclidean distance on the hand-crafted feature vector:
```
d = sqrt( Σ [(x_i − x_ref_i) / σ_i]² )
```
where σ_i is the standard deviation of feature i across the database.

**Eigenspace mode** — scaled Euclidean distance on PCA coefficients:
The query image is preprocessed, projected into the eigenspace, and matched with
scaled Euclidean distance against stored coefficient rows in `database_eigen.csv`.

The nearest-neighbor label is printed on the video and reported in the terminal.

### Task 7 — Evaluate performance / confusion matrix
Press **`m`** to evaluate all images in `data/test/`.

For each image the pipeline runs: threshold → cleanup → segment → features → classify.
True labels are recovered from the filename (e.g. `coin_2.png` → `coin`).
Results are tallied into a confusion matrix printed to the terminal.

With 5 object classes the output is a **5×5 confusion matrix** (true label rows,
predicted label columns).

### Task 8 — One-shot classification using eigenspace (PCA / eigenspace method)
*CNN / ResNet18 is NOT used. The eigenspace (PCA) option is fully implemented.*

Pre-processing pipeline (steps A–D from the assignment):
1. Rotate the original frame so the region's primary axis is aligned with the X-axis
   (rotation by −θ). (`extractOrientedBoxCrop` in `src/utilities.cpp`)
2. Extract the ROI crop corresponding to the axis-aligned bounding box.
3. Resize to **64 × 64** grayscale.
4. Apply histogram equalization to reduce lighting sensitivity.

Training (building the eigenspace):
- All oriented crops in `data/train/` are stacked as rows.
- `cv::PCA` (DATA_AS_ROW) computes the mean image row and eigenvectors.
- Up to 25 principal components are kept (capped at n_samples − 1).
- The mean row and eigenvectors are saved to `models/eigenspace.yml`.
- Each training crop is projected to get its coefficient vector; these are written to
  `data/database_eigen.csv` with a header row (`label, eigen_0, eigen_1, …`).

Classification:
- The query crop is prepared with the same A–D steps.
- It is flattened, mean-subtracted, then multiplied by the eigenvector matrix:
  `coeff = (row − mean) × eigenvectors^T`
- The resulting vector is matched against stored rows using scaled Euclidean distance.
- The nearest neighbor wins.

---

## What to do: data collection, evaluation, and benchmarking

### Step 1 — Collect training data (one sample per class minimum)
1. Run `./object_recognition`.
2. Place the first object (e.g. a coin) in front of the camera on a plain background.
   Wait for the green bounding box to appear stably around it.
3. Press **`t`**, type the label (`coin`), press Enter.
   The system saves the crop, updates the CSV, and rebuilds the eigenspace.
4. Repeat for each class. The assignment uses 5 classes (phone, pen, airpods, coin,
   lipbalm). You may add more. More training samples per class = better accuracy.

### Step 2 — Collect test data (at least 3 images per class)
1. Place the same object at a **different position and orientation** than during training.
2. Press **`x`**, type the label, press Enter. The system saves to `data/test/`.
3. Collect at least 3 test samples per class (15 images total for 5 classes).
   Vary: position (left, center, right), orientation (0°, 45°, 90°), distance.

> Alternatively you can copy labeled still images directly into `data/test/`
> following the naming scheme `<label>_<N>.png` (e.g. `pen_0.png`, `pen_1.png`).

### Step 3 — Generate the normal-features confusion matrix
1. Make sure the system is in **Normal mode** (bottom of the terminal shows `[Normal]`,
   or press `f` to switch back to it).
2. Press **`m`**. The pipeline classifies every image in `data/test/` and prints a
   confusion matrix like:

### Step 4 — Generate the eigenspace confusion matrix
1. Press **`f`** to enter Eigenspace mode (terminal will confirm).
   If the model is not built yet, the system builds it automatically.
2. Press **`m`** again. You will see a second confusion matrix labeled
   `Eigenspace one-shot`.

### Step 5 — Benchmark and compare
Compare the two matrices:
- Diagonal entries = correct classifications.
- Off-diagonal entries = misclassifications.
- Compute per-class accuracy = diagonal / row sum.
- Overall accuracy = sum of diagonal / total test images.

Typical observations:
- **Normal features** (percent_filled + aspect_ratio + Hu moments) perform well when
  objects have distinctive shapes and the threshold is clean.
- **Eigenspace** uses raw appearance (texture + shape), so it can distinguish objects
  that have similar silhouettes but different surface patterns (e.g. phone vs. airpods case).
- Eigenspace can be sensitive to lighting changes unless histogram equalization is used.
- Normal features are more robust to minor illumination variation.

---


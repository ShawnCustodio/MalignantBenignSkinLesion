# Benign vs. Malignant Skin Lesion Classification (MATLAB)

Image-processing + ML pipeline on ISIC dermoscopic images: preprocessing →
mask-based lesion isolation → handcrafted features (ABCD + texture) →
Random Forest vs. superpixel-graph GCN → stratified CV + XAI.

## Requirements
MATLAB R2023b+ with Image Processing, Statistics and Machine Learning, and
Deep Learning toolboxes (all present in R2026a here).

## Data setup
The images are **not** committed. Expected layout:

```
SkinCancer/
├── Benign/      ISIC_xxxxxxx.jpg + ISIC_xxxxxxx_Segmentation.png
└── malignant/   ISIC_xxxxxxx.jpg + ISIC_xxxxxxx_Segmentation.png
```

Tell the code where it is, any one of:
1. `setenv('SKIN_DATA_DIR', 'D:\path\to\SkinCancer')` before running, or
2. copy/symlink it to `data/SkinCancer/` inside the repo (git-ignored), or
3. edit the default path in `config.m`.

## Running
Everything at once (~25 min):
```matlab
cd <repo>
run_all
```
Or step by step (each step reuses the saved outputs of the previous ones):
```matlab
cd <repo>
setup               % adds src/ and scripts/ to the path
s00_check_data      % verify pairs, sizes, masks; writes results/dataset_index.csv
s01_preprocess      % resize, hair removal, illumination + color correction, CLAHE (~1 min)
s02_features        % 63 handcrafted features from the hand-drawn masks
s03_segment         % fully automatic pipeline: finds the lesion itself, scores vs hand-drawn masks
s04_build_graphs    % superpixel graph per lesion (manual + automatic masks)
s05_classify        % Random Forest vs GCN, repeated stratified CV, heatmaps (~10 min)
```

### Step 3 – automatic segmentation (`src/segment/segmentLesion.m`)
Preprocessing runs with **no** hand-drawn mask (`preprocessLesion(I, [], cfg)`).
The lesion is found from a "darkness + saturation" map (PCA of L\* and chroma),
an Otsu threshold scaled by 0.5, rejection of dark vignetting corners, and
morphological clean-up. Mean Dice vs the hand-drawn masks: **0.93** (min 0.76).
The 0.5 scale was chosen on these same 65 images, so this is slightly optimistic.
Outputs: `results/segmentation_scores.csv`, `results/features_auto.csv`,
`results/figures/segmentation_auto_vs_manual.png` (green = hand-drawn, magenta = automatic, worst first).

### Step 4 – graphs (`src/graph/buildLesionGraph.m`)
~150 SLIC superpixels inside each lesion = nodes; touching superpixels = edges.
15 node features: color relative to skin, color spread, gradient, texture,
distance from center, border flag, size, darkness vs the rest of the lesion,
**color difference to neighbors**, blue-gray share, black share.

### Step 5 – classification + explanation (`src/models`, `src/xai`)
* **Random Forest**: `TreeBagger`, 300 trees, on the 63 features; permutation importance.
* **GCN**: 2 graph-convolution layers (16 units) → mean pooling → linear, trained with
  a custom Deep Learning Toolbox loop (Adam, dropout 0.5, weight decay).
* Same stratified 5-fold splits for both, repeated 5×; all predictions are out-of-fold.
* **Region heatmaps**: because the GCN averages node embeddings and then applies a
  linear layer, the malignant score is exactly the mean of per-region scores, so each
  superpixel can be colored red (→ malignant) or blue (→ benign).
  `results/heatmaps_<src>/<id>.png` has one per lesion.
* **Ablation – GCN_noEdges**: the identical network with the edges removed (A = I), so
  each region only sees itself. GCN vs GCN_noEdges isolates the value of the spatial graph.
* **Significance**: exact McNemar test on per-image predictions (`significance_tests.csv`).
* **Failure analysis**: image quality of correct vs wrong images (`quality_vs_errors.csv`)
  and `figures/failure_cases_<src>.png` (lesions both models get wrong).

### Step 1 – preprocessing (`src/preprocess/preprocessLesion.m`)
| Stage | Method | Why |
|---|---|---|
| Resize | shorter side → 512 px, aspect kept | images range 717–2592 px tall; squashing would distort shape features |
| Hair / ruler removal | DullRazor: black top-hat (12 orientations) → keep thin, unbranched curves → `regionfill` inpainting | stricter inside the lesion so pigment network/dots are **not** erased |
| Illumination | divide L* by a heavy Gaussian blur of the skin (lesion filled in first) | removes vignetting / uneven light without touching color |
| Color constancy | Shades-of-Gray (p = 6) on skin pixels | removes yellow/pink camera casts |
| Denoise | bilateral filter | edge-preserving smoothing → `P.smoothed` (**use for color features**) |
| Contrast | CLAHE on L* → `P.enhanced` | **use for texture / graph features only** (CLAHE distorts color) |

### Step 2 – features (`s02_features`, `src/features/extractFeatures.m`)
63 features per lesion, all scale-free (magnification differs between images):

| Group | Features |
|---|---|
| Shape | areaFraction, compactness, solidity, extent, eccentricity, aspectRatio, convexity |
| Asymmetry | shape XOR after mirroring about both principal axes; color ΔE between halves |
| Border | radial-distance CV, box-counting fractal dimension, border gradient mean + variation across 8 sectors |
| Color | mean/std/skew of L\*a\*b\*, saturation, lesion-vs-skin contrast, share of 6 dermoscopic colors (Lab rules, relative to skin), color count |
| Pigmentation | L\* entropy, center-vs-periphery lightness |
| Texture | GLCM contrast/correlation/energy/homogeneity/entropy at 1, 3, 5 px; rotation-invariant uniform LBP (10 bins) |

Outputs: `results/features.csv`, `results/feature_ranking.csv` (per-feature AUC
and Wilcoxon p, **exploratory only – computed on all images**),
`results/figures/top_features_boxplots.png`.

Step 1 outputs: `results/preprocessed/<id>.mat` (struct `P`), `results/image_quality.csv`
(brightness, sharpness, vignetting, hair fraction… for the image-quality analysis),
`results/figures/preprocessing_stages.png`, `results/figures/before_after.png`.

## Results (5 × repeated stratified 5-fold CV, 65 images)

| Masks | Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---|---|---|---|---|---|
| hand-drawn | Random Forest | 0.60 ± 0.07 | 0.69 | 0.51 | 0.65 | 0.64 |
| hand-drawn | GCN | **0.68 ± 0.06** | 0.69 | **0.67** | **0.70** | **0.71** |
| hand-drawn | GCN, no edges (ablation) | 0.64 ± 0.05 | 0.68 | 0.60 | 0.67 | 0.68 |
| automatic | Random Forest | 0.57 ± 0.04 | 0.67 | 0.45 | 0.63 | 0.63 |
| automatic | GCN | **0.65 ± 0.05** | 0.70 | 0.59 | 0.68 | 0.70 |
| automatic | GCN, no edges (ablation) | 0.62 ± 0.06 | 0.66 | 0.57 | 0.65 | 0.67 |

Exact McNemar tests (images one model gets right and the other wrong):

| Masks | Comparison | 1st right / 2nd wrong | reverse | p |
|---|---|---|---|---|
| hand-drawn | GCN vs RF | 15 | 9 | 0.31 |
| hand-drawn | GCN vs GCN no edges | 3 | 3 | 1.00 |
| automatic | GCN vs RF | 16 | 5 | **0.03** |
| automatic | GCN vs GCN no edges | 5 | 0 | 0.06 |

Key findings
* The region-based GCN beats the handcrafted-feature Random Forest in every setting,
  mainly through specificity; significant only with automatic masks (p = 0.03).
* Most of the gain comes from the region-level description itself: removing the edges
  costs only 2–4 points and is not significant. Spatial neighbor information helps a
  little (trend with automatic masks, p = 0.06), but 65 images cannot confirm it.
* Fully automatic masks (Dice 0.93) cost about 3 accuracy points.
* Regions the GCN marks as malignant have more texture/edges, more internal color
  variation, more blue-gray and larger color differences to their neighbors.
* GCN errors occur on blurrier images (sharpness p = 0.03 both mask types); hair and
  vignetting were not associated with errors.

Limitations: tiny dataset (wide error bars); GCN settings fixed a priori (no nested CV);
segmentation threshold and dermoscopic-color thresholds are heuristic.

## Layout
| Folder | Purpose |
|---|---|
| `config.m`, `setup.m` | paths, class names, CV folds, seed |
| `src/io` | dataset index, image/mask reading |
| `src/preprocess` | illumination/contrast, color normalization, hair removal (DullRazor) |
| `src/features` | area, asymmetry, border, color, texture (GLCM/LBP) |
| `src/segment` | automatic lesion segmentation + Dice scoring |
| `src/graph` | superpixel graph construction (nodes = SLIC regions inside mask) |
| `src/models` | Random Forest (`TreeBagger`), GCN (Deep Learning Toolbox custom loop) |
| `src/eval` | stratified folds, metrics |
| `src/xai` | RF permutation importance, GCN node-level attribution |
| `scripts/` | numbered pipeline steps `s00_…`, `s01_…` |
| `results/` | generated tables/figures (git-ignored) |

Use the same `makeFolds` partition for both models so the comparison is fair.

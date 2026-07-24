# AGENTS.md

Engineering notes for agents working on `ingest-scans`. Read before touching
detection or orientation.

## What this is

Single-file CLI (`ingest.py`) that splits flatbed scans of loose printed photos
into one PNG per photo. Originally ChatGPT-generated; the detection and
orientation stages were rewritten (see history below).

## Environment (important, non-obvious)

- Python **3.14**, venv at `.venv/`.
- **`opencv-python` 5.x**. OpenCV 5 is the source of two nasty surprises:
  - `cv2.CascadeClassifier` is **gone** from the Python build. `cv2.data`
    exists but `cv2/data/` ships only an `__init__.py` — no Haar XMLs. Any
    Haar-based code will `AttributeError`.
  - Orientation therefore uses **`cv2.FaceDetectorYN`** (YuNet DNN) instead.
    Model vendored at `assets/face_detection_yunet_2023mar.onnx` (~230 KB, from
    the `opencv_zoo` repo). Loading it prints a harmless
    `setPreferableTarget ... not supported by the new graph engine` warning.
- Real scan corpus: `~/Sync/Titi_Scans_07-24/` (84 PNGs, ~50–60 MB each,
  5100×7016 typical). Output convention: `~/Sync/Extract/` (+ `debug/`).

## The data (drives every design choice)

Family-album scans: **white-bordered prints on a near-white scanner lid**
(corner LAB L≈243). Consequences:

- Background-colour subtraction (the original approach) sees ~zero difference →
  detected almost nothing (31 photos across 84 scans; 53 scans returned nothing).
- Prints are scattered, various sizes, small skew — **not** a clean grid, so
  projection-profile splitting doesn't generalise.
- Some scans have **horizontal scanner banding** (low-amplitude gray stripes)
  that naive texture detection flags as foreground → must denoise first.
- Adjacent/stacked prints are separated only by thin white gaps and often linked
  by drop-shadow lines → components weld together without an explicit sever step.

## Detection algorithm (current, in `build_detection_mask` + `find_photo_detections`)

1. `bilateralFilter` to kill banding while keeping real photo edges.
2. Foreground = **faint border edges** (`Canny(12,40)`, low is safe post-denoise)
   OR **interior signal** (saturation `>18` or darkness `>35`), morph-gradient'd
   to an outline.
3. Dilate + close → filled photo regions.
4. **Open** with a kernel `~1.4%` of image height — this is the key step; it
   severs thin bridges so each print becomes its own connected component.
5. `connectedComponentsWithStats` → per component **axis-aligned `boundingRect`**
   (`CC_STAT_*`). Not `minAreaRect` — see lesson below.
6. Accept filter: `min_area < area_frac < 0.94`, `aspect < 6.5`.
   (The `<0.94` also rejects the near-full-page blob that appears on failure.)
7. `merge_overlapping_boxes`: union any two boxes overlapping by >0.12 of the
   smaller, to a fixed point. Collapses slivers and sky/subject splits into one
   box per photo. Genuinely separate prints keep their white gap, so they don't
   merge.

Output is **not cropped or deskewed** — `crop_bounding_box` returns
`image[y0:y1, x0:x1]` padded by `--margin`. The requirement is only to split the
scan into whole prints; orientation happens afterward on each split.

Result: **~270 photos across the 84 scans** (~3/scan, matches expectation).

### Lesson: do not use minAreaRect here

The first version fit `minAreaRect` per component and perspective-warped it.
On irregular/partial components it produced **skewed diamonds over sub-regions**
of a single photo (a vampire's face, a patch of water), splitting one photo into
2–3 rotated slivers. Axis-aligned `boundingRect` + overlap-merge fixed this
(`titi_universalstudio87_campvivaldi`, `titi_voilier` were the reference
failures). Photos on a flatbed are near axis-aligned, so upright boxes are fine.
Genuinely one-photo-with-a-big-internal-bright-gap (voilier top: water separated
from cabin by boat structure) can still over-split; erring toward over-split is
acceptable (extra file to discard) — over-*merge* would lose a photo.

### Do NOT

- `fill_holes`/flood-fill the mask: inter-photo gaps become enclosed "holes" and
  fill in, welding the whole page into one component. Tried it, whole page → 1 blob.
- `RETR_EXTERNAL` contours: when foreground bridges across the page the outer
  contour swallows every print. Use connected components instead.
- Push detection resolution high; `--processing-size 1200` is tuned. Absolute
  thresholds (Canny, sat/dark) were calibrated at ~1100–1200 px.

## Orientation (`load_face_detector`, `face_orientation_score`, `auto_orient_photo`)

YuNet run on all 4 rotations; highest score (face count + area + confidence)
wins, but only if clearly better than the others (guards against spurious flips).
`score_threshold=0.9` — at 0.6 it detected up to 88 "faces" in texture and
corrupted the rotation choice. Keep it high.

## Verifying changes

Always visual. Fast loop:

```bash
.venv/bin/python ingest.py ~/Sync/Titi_Scans_07-24 ~/Sync/Extract --dry-run --debug
# then montage the overlays and look at them:
```

The montage-all-overlays snippet (7-wide grid of `*_detections.png`) is how the
whole corpus was judged in one image — reach for that over reading logs. Sanity
checks worth running on the log: max `faces=` (should be small, group photos
aside) and the spread of `orientation +N` values.

Dead code removed in the rewrite: `sample_background_colour`, `contour_to_quad`,
`points_are_reasonable`, `polygon_iou`. `build_detection_mask` still accepts an
unused `background_threshold` param to keep the CLI flag stable.

## Known remaining gaps

- Two genuinely touching prints can still merge into one box (hardest case).
- A few very light, low-detail prints under-detect; lower `--min-area` per-scan.
- Not every scan hits 100%; a manual cleanup pass on a handful is expected.

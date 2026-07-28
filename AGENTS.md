# AGENTS.md

Engineering notes for agents working on `descan`. Read before touching
detection or orientation.

## Before pushing

Run `./scripts/gate.sh` (ruff format check + lint) and make it pass before every
push.
Skip only for docs-only changes (`*.md`). CI (`.github/workflows/ci.yml`) runs
the same two checks on push and PR.

## What this is

A small suite for turning flatbed scans of loose prints into one PNG per photo.
Scripts live in `src/`; run them directly (`python src/ingest.py ...`), which
puts `src/` on the path so the others can `import ingest`. `ingest.py` was
originally ChatGPT-generated; its detection and orientation stages were then
rewritten (history below). The rest are manual clean-up tools.

- `ingest.py` — automatic detect + split + orient (the core; this file is what
  the notes below are about).
- `autocrop_review.py` — GUI to trim leftover white lid margins.
- `manualcrop.py` — GUI to draw boxes for missed/mis-split photos.
- `rename_photos.py` — date-prefixed renaming from `.xmp`/EXIF via exiftool.

All share `ingest.py`'s image IO (`read_image`, `write_png`, `find_input_files`)
and its YuNet orientation.

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

Deliberately simple — this is the third iteration, and simpler beat clever:

1. `bilateralFilter` to suppress scanner banding.
2. Foreground = **not background**: `(background_L - L) > 12` (darker than the
   lid) OR `chroma > 10` (coloured). `background_L` = median lightness of the
   3px scan border (border is almost always lid).
3. `MORPH_CLOSE` (~1%) to bridge speckle, then **`fill_holes`** — flood the
   background in from a *padded* frame, turn on whatever it can't reach. This
   solidifies each photo (sky, white shirts included) into one blob and is what
   killed the over-split problem.
4. `MORPH_OPEN` (~1.2%) to drop thin links between adjacent prints and noise.
5. `connectedComponentsWithStats` → per component **axis-aligned `boundingRect`**
   (`CC_STAT_*`), never `minAreaRect` — see lesson.
6. Accept filter: `min_area < area_frac < 0.94`, `aspect < 6.5`.
   (`<0.94` also rejects the near-full-page blob that appears on failure.)
7. `merge_overlapping_boxes`: union boxes overlapping by >0.12 of the smaller.
   Separate prints keep their white gap, so they don't merge.
8. **`expand_boxes_to_lid`**: the mask hugs coloured/dark content, so the box
   clips the print's white border and pale regions (sky, white shirts, pale
   walls) — the main "over-crop" complaint. Each side is pushed outward in small
   steps and stops when the strip just beyond is mostly flat white lid
   (`is_lid = L>p92-12 & chroma<7`, stop at >80% lid), when it reaches another
   box, or at the image edge. Because it stops at the white gap it can never
   merge separate prints. This is a *stopping* use of whiteness; using flatness
   as a positive foreground signal instead welds everything (lid banding reads
   as texture — tried it, whole scan → 1 box).
9. **`merge_content_bridged_boxes`**: a photo with a low-content band through it
   (plain sand, a dark wall) splits into adjacent boxes. Rejoin an adjacent pair
   only if the *whole union rectangle* is almost all non-lid
   (`mean(is_lid) < 0.10`) — one solid photo. Deliberately strict: it fires on
   clearly dark over-splits (`titi_noel_2001`) but abstains when a photo has
   large bright regions that look like a gap (`titi_nyc8`'s bright sky stays
   split — accepted), and it never welds pale/faded prints, which read as
   lid-like (`titi_mariage`: welding it to 1 box lost photos — the failure this
   guard exists to prevent). Losing a photo is far worse than an over-split
   (just an extra file to discard), so the whole merge is biased to abstain.

Output is **not cropped or deskewed** — `crop_bounding_box` returns
`image[y0:y1, x0:x1]` padded by `--margin`. Orientation runs afterward per split.
The separation mask (steps 1-7) and the extent (step 8) are deliberately split:
the mask is good at *counting/separating* photos but tight; expansion recovers
the full extent without risking a merge. Reference over-crop fixes:
`titi_plage`, `titi_violon` (was cutting the pale-walled violin shots in half).

Result: **~220 photos across the 84 scans**.

### Lesson: minAreaRect and the emap approach were both wrong

- The 2nd version built a foreground from **Canny edges + interior gradient**,
  dilate/close/open, then fit **`minAreaRect`** and perspective-warped. On
  irregular/partial components minAreaRect produced **skewed diamonds over
  sub-regions** of one photo (a vampire's face, a patch of water) → 2–3 rotated
  slivers per print (`titi_universalstudio87_campvivaldi`, `titi_voilier` were
  the reference failures). Photos on a flatbed are near axis-aligned; upright
  `boundingRect` is correct and simpler.
- `fill_holes` was avoided in v2 for fear of welding pages, but the real welds
  came from over-aggressive `MORPH_CLOSE` *before* filling, plus a **seed bug**
  (flooding from `(0,0)` when a photo touches that corner fills the whole page).
  Pad a background frame and seed from there and it is safe: inter-photo gaps
  run to the border so the flood reaches them and they stay open.

### Do NOT

- `RETR_EXTERNAL` contours: when foreground bridges across the page the outer
  contour swallows every print. Use connected components.
- flood `fill_holes` from `(0,0)` without a padded background frame (seed bug
  above), or after a large `MORPH_CLOSE` that rings the whole page.
- Push detection resolution high; `--processing-size 1200` is tuned. The
  absolute thresholds (L delta, chroma) were calibrated at ~1200 px.

## Orientation (`load_face_detector`, `face_orientation_score`, `auto_orient_photo`)

YuNet run on all 4 rotations; highest score (face count + area + confidence)
wins, but only if clearly better than the others (guards against spurious flips).
`score_threshold=0.9` — at 0.6 it detected up to 88 "faces" in texture and
corrupted the rotation choice. Keep it high.

## Verifying changes

Always visual. Fast loop:

```bash
.venv/bin/python src/ingest.py ~/Sync/Titi_Scans_07-24 ~/Sync/Extract --dry-run --debug
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

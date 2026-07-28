# AGENTS.md

Working notes for `descan`. Read before touching detection or orientation.
History and rejected approaches live in [docs/findings.md](docs/findings.md).

## Before pushing

Run `./scripts/gate.sh` (ruff format check + lint); it must pass. Skip only for
docs-only changes (`*.md`). CI runs the same two checks.

## Conventions

- Commit messages: terse, no `Co-Authored-By` trailer.
- Keep all docs (README, AGENTS, docstrings) concise and terse.

## Layout

Installable package `descan` under `src/descan/` (`pip install -e ".[dev]"`).
Modules cross-import as `from descan.ingest import ...`; all share `ingest.py`'s
image IO (`read_image`, `write_png`, `find_input_files`) and its YuNet
orientation. The YuNet model is package data at `src/descan/assets/`.

`cli.py` is the unified `descan <command>` dispatcher (and `--version`).
`[project.scripts]` also maps each module's `main()` to a direct command:

- `descan-split` = `ingest.py` — auto detect + split + orient (the core; the
  notes below are it).
- `descan-trim` = `autocrop_review.py` — GUI, trim white lid margins.
- `descan-crop` = `manualcrop.py` — GUI, draw boxes for missed/mis-split photos.
- `descan-rename` = `rename_photos.py` — dated renaming from `.xmp`/EXIF.
- `descan-montage` = `montage.py` — tile a directory of images into one grid.

## Environment gotchas

- Python 3.14, venv at `.venv/`.
- `opencv-python` 5.x: `cv2.CascadeClassifier` is **gone** (`cv2/data/` ships no
  Haar XMLs). Orientation uses `cv2.FaceDetectorYN` (YuNet), model vendored at
  `assets/face_detection_yunet_2023mar.onnx`. Loading prints a harmless
  `setPreferableTarget ... not supported` warning.

## The data

Loose white-bordered prints on a near-white scanner lid (corner LAB L≈243):
scattered, various sizes, slight skew (not a grid). Scans can carry faint
horizontal banding; adjacent prints are separated only by thin white gaps.

## Detection (`build_detection_mask` + `find_photo_detections`)

Separate first, then recover extent:

1. `bilateralFilter` (suppresses banding).
2. Foreground = not-background: `(background_L - L) > 12` OR `chroma > 10`;
   `background_L` = median lightness of the 3px border.
3. `MORPH_CLOSE` (~1%), then `fill_holes` (flood bg from a **padded** frame — the
   seed must be background) to solidify each photo, pale interior included.
4. `MORPH_OPEN` (~1.2%) to sever thin links between prints.
5. `connectedComponentsWithStats` → axis-aligned `boundingRect` per component
   (never `minAreaRect`).
6. Accept: `min_area < area_frac < 0.94`, `aspect < 6.5`.
7. `merge_overlapping_boxes`: union pairs overlapping >0.12 of the smaller.
8. `expand_boxes_to_lid`: grow each box outward until the strip beyond is mostly
   white lid (`is_lid = L>p92-12 & chroma<7`, >80%), a neighbour, or the edge.
   Recovers clipped borders/sky; stops at the gap so it can't merge.
9. `merge_content_bridged_boxes`: rejoin adjacent boxes only if the whole union
   is almost all non-lid (`mean(is_lid) < 0.10`). Strict, biased to abstain —
   losing a photo is worse than an over-split.

Output is not cropped/deskewed: `crop_bounding_box` returns `image[y0:y1, x0:x1]`
padded by `--margin`. Orientation runs afterward per split.

## Do NOT

- Use `minAreaRect` (rotated slivers) or `RETR_EXTERNAL` contours (swallow the
  page when foreground bridges). Use connected components + `boundingRect`.
- `fill_holes` from `(0,0)` without a padded background frame, or after a large
  `MORPH_CLOSE` that rings the page — both weld the whole scan.
- Use flatness as a positive foreground signal — banding reads as texture and
  welds everything.
- Push detection resolution up; `--processing-size 1200` is tuned (absolute
  L/chroma thresholds calibrated there).

## Orientation

YuNet on all 4 rotations; best score (face count + area + confidence) wins only
if clearly better than the rest. `score_threshold=0.9` (lower hallucinates faces
in texture and flips wrongly).

## Verifying

Visual. `descan-split <scans> <out> --dry-run --debug`, then `descan-montage
<out>/debug` to tile the `*_detections.png` overlays into one grid and look —
faster than logs. Log sanity: max `faces=` small; spread of `orientation +N`.

## Known gaps

- Two genuinely touching prints can merge into one box.
- Very light / low-detail prints under-detect; lower `--min-area`.
- Not 100%; expect a small manual pass.
